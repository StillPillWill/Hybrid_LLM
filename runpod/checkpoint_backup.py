"""Pull RunPod checkpoints into a rolling backup and an archive folder.

Run with --loop for a 15-minute poller, or --once for Task Scheduler.
The RunPod SSH gateway requires an interactive shell, so this uses an SSH
PTY and a marker-delimited base64 stream instead of scp/sftp.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


SSH_TARGET = os.environ.get("LLMY_RUNPOD_SSH_TARGET", "")
SSH_KEY = Path(os.environ.get("LLMY_SSH_KEY", Path.home() / ".ssh" / "id_ed25519"))
REMOTE_FILE = os.environ.get(
    "LLMY_REMOTE_CHECKPOINT",
    "/workspace/llmy/runs/a5000_moe24/latest.pt",
)
E_ROOT = Path(os.environ.get("LLMY_BACKUP_ROOT", "runpod_backups/a5000_moe24"))
F_ROOT = Path(os.environ.get("LLMY_ARCHIVE_ROOT", "runpod_archive/a5000_moe24"))
POLL_SECONDS = 900
TRANSFER_TIMEOUT_SECONDS = 300
CHUNK_TIMEOUT_SECONDS = 180
CHUNK_RETRIES = 3
# Keep each SSH response below the gateway output limit. Valid short responses
# are accepted, and the next request resumes at the exact byte offset.
CHUNK_BYTES = 32 * 1024 * 1024
TRANSFER_RETRIES = 3
LOCK_PATH = E_ROOT / ".checkpoint_backup.lock"


def ssh_stream(command: str, *, timeout: int = TRANSFER_TIMEOUT_SECONDS) -> bytes:
    args = [
        "ssh", "-tt", "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "IdentitiesOnly=yes",
        "-o", "PreferredAuthentications=publickey", "-i", str(SSH_KEY),
        SSH_TARGET,
    ]
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        data, _ = proc.communicate(
            ("stty -echo; stty cols 1000 rows 1000; " + command + "; exit\n").encode(),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise
    if proc.returncode not in (0, None):
        raise RuntimeError(f"ssh exited with {proc.returncode}: {data[-500:]!r}")
    return data


def remote_signature(path: str | None = None) -> tuple[int, int, int]:
    remote_path = shlex.quote(path or REMOTE_FILE)
    raw = ssh_stream(f"stat -c '%Y %s %i' -- {remote_path}")
    matches = re.findall(rb"(\d{9,}) (\d+) (\d+)", raw)
    if not matches:
        raise RuntimeError(f"could not read checkpoint signature: {raw[-1000:]!r}")
    stamp, size, inode = matches[-1]
    # Include the inode as well as mtime and size.  The trainer normally
    # publishes latest.pt with an atomic replace, so this catches a new
    # checkpoint even when it has the same byte length.
    return int(stamp), int(size), int(inode)


def validate_checkpoint(path: Path) -> None:
    """Validate the complete PyTorch ZIP, including CRCs for every member."""
    try:
        with zipfile.ZipFile(path) as archive:
            if not archive.namelist():
                raise zipfile.BadZipFile("empty checkpoint archive")
            bad_member = archive.testzip()
    except Exception as exc:
        raise RuntimeError(f"checkpoint ZIP validation failed: {exc}") from exc
    if bad_member is not None:
        raise RuntimeError(f"checkpoint CRC validation failed at {bad_member}")


def pull_checkpoint(destination: Path, expected_size: int, source: str | None = None) -> None:
    """Pull a checkpoint in bounded chunks so the SSH gateway cannot truncate it."""
    begin, end = b"__RUNPOD_CHUNK_BEGIN__", b"__RUNPOD_CHUNK_END__"
    remote_source = shlex.quote(source or REMOTE_FILE)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".partial")
    if temp.exists():
        existing_size = temp.stat().st_size
        if existing_size > expected_size:
            temp.unlink()
    offset = temp.stat().st_size if temp.exists() else 0
    with temp.open("ab") as out:
        while offset < expected_size:
            chunk_size = min(CHUNK_BYTES, expected_size - offset)
            block_bytes = 1024 * 1024
            aligned_offset = offset - (offset % block_bytes)
            prefix_bytes = offset - aligned_offset
            blocks = (prefix_bytes + chunk_size + block_bytes - 1) // block_bytes
            read = (
                f"dd if={remote_source} bs=1048576 skip={aligned_offset // block_bytes} "
                f"count={blocks} status=none"
            )
            if prefix_bytes:
                read += f" | tail -c +{prefix_bytes + 1}"
            read += f" | head -c {chunk_size}"
            command = (
                f"printf '{begin.decode()}\\n'; "
                f"({read}) | base64 -w 0; "
                f"printf '\\n{end.decode()}\\n'"
            )
            for attempt in range(1, CHUNK_RETRIES + 1):
                try:
                    raw = ssh_stream(command, timeout=CHUNK_TIMEOUT_SECONDS)
                    begin_offset = raw.rfind(begin)
                    end_offset = raw.rfind(end)
                    if begin_offset < 0 or end_offset <= begin_offset:
                        raise RuntimeError("markers missing or truncated")
                    encoded = re.sub(
                        rb"[^A-Za-z0-9+/=]",
                        b"",
                        raw[begin_offset + len(begin):end_offset],
                    )
                    decoded = base64.b64decode(encoded, validate=True)
                    decoded_size = len(decoded)
                    if decoded_size <= 0 or decoded_size > chunk_size:
                        raise RuntimeError(
                            f"chunk size mismatch: got {decoded_size}, expected at most {chunk_size}"
                        )
                    out.write(decoded)
                    out.flush()
                    offset += decoded_size
                    print(
                        f"[backup] chunk {offset / expected_size:.0%} "
                        f"({offset / 1e9:.2f}/{expected_size / 1e9:.2f} GB)",
                        flush=True,
                    )
                    break
                except Exception as exc:
                    if attempt == CHUNK_RETRIES:
                        raise RuntimeError(
                            f"chunk at {offset} failed after {CHUNK_RETRIES} attempts: {exc}"
                        ) from exc
                    print(
                        f"[backup] chunk at {offset / expected_size:.0%} failed "
                        f"(attempt {attempt}/{CHUNK_RETRIES}): {exc}; retrying",
                        flush=True,
                    )
    if temp.stat().st_size != expected_size:
        raise RuntimeError(f"checkpoint size mismatch: got {temp.stat().st_size}, expected {expected_size}")
    # Size alone does not prove that the remote checkpoint was read
    # consistently.  PyTorch checkpoints are ZIP archives; reject a truncated
    # or mixed-version archive before exposing it as a completed .pt file.
    try:
        with zipfile.ZipFile(temp) as archive:
            names = archive.namelist()
            if not names:
                raise zipfile.BadZipFile("empty checkpoint archive")
            root = names[0].split("/", 1)[0]
            archive.read(f"{root}/byteorder")
            archive.read(f"{root}/version")
            archive.read(f"{root}/data.pkl")
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"checkpoint archive validation failed: {exc}") from exc
    temp.replace(destination)


def backup_once() -> bool:
    E_ROOT.mkdir(parents=True, exist_ok=True)
    F_ROOT.mkdir(parents=True, exist_ok=True)
    for stale in E_ROOT.glob("*.ssh.raw"):
        stale.unlink(missing_ok=True)
    for transfer_attempt in range(1, TRANSFER_RETRIES + 1):
        snapshot = f"/workspace/llmy/runs/a5000_moe24/.checkpoint_backup_{os.getpid()}_{int(time.time())}_{transfer_attempt}.pt"
        try:
            # A hard link pins the current inode without consuming another
            # 3-GB copy. The trainer can atomically replace latest.pt while
            # this snapshot remains immutable for the duration of the pull.
            ssh_stream(f"ln -- {shlex.quote(REMOTE_FILE)} {shlex.quote(snapshot)}")
            stamp, size, inode = remote_signature(snapshot)
            current_tag = f"remote{stamp}_{inode}"

            existing = sorted(E_ROOT.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            matching = [p for p in existing if current_tag in p.name and p.stat().st_size == size]
            if matching:
                try:
                    validate_checkpoint(matching[0])
                except Exception as exc:
                    print(f"[backup] removing invalid existing copy {matching[0]}: {exc}", flush=True)
                    matching[0].unlink(missing_ok=True)
                else:
                    print(f"[backup] unchanged and valid ({size / 1e9:.2f} GB, remote {current_tag}); skipping", flush=True)
                    return False

            # Never resume a partial belonging to an older remote inode.
            for stale in E_ROOT.glob("checkpoint_*.pt.partial"):
                if current_tag not in stale.name:
                    stale.unlink(missing_ok=True)
            partials = sorted(E_ROOT.glob(f"checkpoint_*_{current_tag}.pt.partial"))
            if partials:
                destination = partials[-1].with_suffix("")
                print(f"[backup] resuming {destination}.partial (attempt {transfer_attempt})", flush=True)
            else:
                now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                destination = E_ROOT / f"checkpoint_{now}_{current_tag}.pt"
                print(f"[backup] pulling {size / 1e9:.2f} GB -> {destination} (attempt {transfer_attempt})", flush=True)

            try:
                pull_checkpoint(destination, size, snapshot)
                after_signature = remote_signature(snapshot)
                if after_signature != (stamp, size, inode):
                    raise RuntimeError(
                        f"remote checkpoint snapshot changed: {stamp, size, inode} -> {after_signature}"
                    )
                validate_checkpoint(destination)

                # Publish to F atomically and validate the archive there too.
                archive = F_ROOT / destination.name
                archive_temp = archive.with_suffix(archive.suffix + ".partial")
                archive_temp.unlink(missing_ok=True)
                shutil.copy2(destination, archive_temp)
                validate_checkpoint(archive_temp)
                archive_temp.replace(archive)
            except Exception as exc:
                print(f"[backup] rejected transfer attempt {transfer_attempt}: {exc}", flush=True)
                destination.unlink(missing_ok=True)
                destination.with_suffix(destination.suffix + ".partial").unlink(missing_ok=True)
                continue

            for old in sorted(E_ROOT.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)[2:]:
                old.unlink(missing_ok=True)
            print(
                f"[backup] rolling copies={len(list(E_ROOT.glob('checkpoint_*.pt')))}; "
                f"archive saved and validated at {archive}",
                flush=True,
            )
            return True
        finally:
            try:
                ssh_stream(f"rm -f -- {shlex.quote(snapshot)}", timeout=60)
            except Exception as exc:
                print(f"[backup] warning: could not remove remote snapshot: {exc}", flush=True)

    raise RuntimeError(f"could not obtain a stable, CRC-valid checkpoint after {TRANSFER_RETRIES} attempts")


def acquire_lock() -> int | None:
    """Allow only one full-size checkpoint transfer at a time."""
    E_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            stale = time.time() - LOCK_PATH.stat().st_mtime > POLL_SECONDS
            lock_text = LOCK_PATH.read_text(errors="ignore")
            match = re.search(r"pid=(\d+)", lock_text)
            if match and int(match.group(1)) != os.getpid():
                lock_pid = int(match.group(1))
                if os.name == "nt":
                    probe = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {lock_pid}", "/NH"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    process_exists = re.search(
                        rf"^\s*\S+\s+{lock_pid}\s",
                        probe.stdout,
                        flags=re.MULTILINE,
                    ) is not None
                    stale = stale or not process_exists
                else:
                    try:
                        os.kill(lock_pid, 0)
                    except ProcessLookupError:
                        stale = True
                    except PermissionError:
                        pass
            if stale:
                LOCK_PATH.unlink()
                return acquire_lock()
        except FileNotFoundError:
            return acquire_lock()
        print("[backup] another backup is already running; skipping", flush=True)
        return None
    os.write(fd, f"pid={os.getpid()}\n".encode())
    return fd


def main() -> int:
    global SSH_TARGET, SSH_KEY, REMOTE_FILE, E_ROOT, F_ROOT, LOCK_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=POLL_SECONDS)
    parser.add_argument(
        "--ssh-target",
        default=SSH_TARGET or None,
        required=not bool(SSH_TARGET),
        help="RunPod SSH user and host, or set LLMY_RUNPOD_SSH_TARGET",
    )
    parser.add_argument(
        "--ssh-key",
        default=str(SSH_KEY),
        help="SSH private key path, or set LLMY_SSH_KEY",
    )
    parser.add_argument("--remote-file", default=REMOTE_FILE)
    parser.add_argument("--e-root", default=str(E_ROOT))
    parser.add_argument("--f-root", default=str(F_ROOT))
    args = parser.parse_args()
    SSH_TARGET = args.ssh_target
    SSH_KEY = Path(args.ssh_key)
    REMOTE_FILE = args.remote_file
    E_ROOT = Path(args.e_root)
    F_ROOT = Path(args.f_root)
    LOCK_PATH = E_ROOT / ".checkpoint_backup.lock"
    lock_fd = acquire_lock()
    if lock_fd is None:
        return 0
    try:
        while True:
            try:
                backup_once()
            except Exception as exc:
                print(f"[backup] ERROR: {exc}", file=sys.stderr, flush=True)
                # In one-shot/task-scheduler mode, surface transfer failures to
                # the caller.  Returning success here made Task Scheduler
                # believe a failed backup completed and prevented its retry /
                # failure reporting mechanisms from doing anything useful.
                if args.once:
                    return 1
            if args.once:
                return 0
            time.sleep(max(60, args.interval))
    finally:
        os.close(lock_fd)
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
