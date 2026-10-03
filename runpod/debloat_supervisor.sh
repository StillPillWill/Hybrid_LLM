#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

debloat_seconds="${LLMY_DEBLOAT_SECONDS:-7200}"
run_dir="${LLMY_RUN_DIR:-runs/a5000_finemath4plus_v1}"
config_path="${LLMY_CONFIG:-runpod/config_a5000_moe24.yaml}"
data_dir="${LLMY_DATA_DIR:-data/streamed_fineweb}"
log_file="$run_dir/supervisor.log"

mkdir -p "$run_dir"
touch "$log_file"
exec >> "$log_file" 2>&1
child_pid=0

forward_term() {
    if [[ "$child_pid" != "0" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM "$child_pid" 2>/dev/null || true
    fi
}
trap forward_term TERM INT

while true; do
    start_line="$(wc -l < "$log_file" 2>/dev/null || printf '0')"
    resume_flag=()
    reset_optimizer_flag=()

    if [[ -f "$run_dir/.supervisor_initialized" ]]; then
        resume_flag+=(--resume)
        if [[ "${LLMY_RESET_OPTIMIZER_ONCE:-0}" == "1" && ! -f "$run_dir/.optimizer_reset_done" ]]; then
            reset_optimizer_flag+=(--reset-optimizer)
            touch "$run_dir/.optimizer_reset_done"
            echo "[Supervisor] Resetting optimizer state once; model weights and counters will resume."
        fi
        echo "[Supervisor] Restarting from the latest checkpoint; CUDA cleanup is scheduled in ${debloat_seconds}s."
    else
        touch "$run_dir/.supervisor_initialized"
        echo "[Supervisor] Starting a fresh run; CUDA cleanup is scheduled in ${debloat_seconds}s."
    fi

    LLMY_DIFF_MOE=0 \
    LLMY_MOE_TOKEN_CHUNK=64 \
    LLMY_VOCAB_CHUNK=16 \
    LLMY_LONG_SEQUENCE_LOOP_THRESHOLD=4096 \
    LLMY_LONG_SEQUENCE_MAX_LOOPS=1 \
    LLMY_ENABLE_COMPILED_CE=1 \
    LLMY_ENABLE_COMPILED_MLA=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LLMY_MAX_PREFETCH_SHARDS=8 \
        python3 -u runpod/train_a5000_ultimate.py \
            --config "$config_path" \
            --data-dir "$data_dir" \
            --run-dir "$run_dir" \
            --device cuda "${resume_flag[@]}" "${reset_optimizer_flag[@]}" &
    child_pid=$!

    (
        sleep "$debloat_seconds"
        if kill -0 "$child_pid" 2>/dev/null; then
            echo "[Supervisor] Stopping trainer PID $child_pid for scheduled CUDA cleanup."
            kill -TERM "$child_pid" 2>/dev/null || true
        fi
    ) &
    timer_pid=$!

    wait "$child_pid"
    status=$?
    kill "$timer_pid" 2>/dev/null || true
    wait "$timer_pid" 2>/dev/null || true
    child_pid=0

    if tail -n +$((start_line + 1)) "$log_file" | grep -q "\[Completed\] 1 Billion Tokens Pretraining Finished!"; then
        echo "[Supervisor] Training completed."
        exit "$status"
    fi

    echo "[Supervisor] Trainer exited with status $status; restarting from the latest checkpoint when available."
    sleep 5
done
