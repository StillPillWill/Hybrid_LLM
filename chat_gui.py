"""Local browser chat UI for the self-contained sampler.

The UI keeps the full system/user/assistant conversation and sends it through
the tokenizer chat template before generation. Assistant messages are rendered
as Markdown in the browser, including fenced code, lists, links, tables, and
block quotes. It uses only Python's standard library on the server side.

Run locally:
    python chat_gui.py --checkpoint runs/a5000_moe24/latest.pt --port 7860
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLMY Chat</title>
<style>
:root { color-scheme: dark; --bg:#0d1117; --panel:#161b22; --line:#30363d;
        --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --user:#1f6feb; }
* { box-sizing:border-box; }
body { margin:0; min-height:100vh; background:var(--bg); color:var(--text);
       font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif; }
.app { max-width:1100px; margin:0 auto; min-height:100vh; display:flex; flex-direction:column; }
header { padding:20px 24px 14px; border-bottom:1px solid var(--line); display:flex;
         align-items:center; justify-content:space-between; gap:16px; }
h1 { margin:0; font-size:20px; letter-spacing:.02em; }
.status { color:var(--muted); font-size:12px; text-align:right; }
main { flex:1; padding:24px; }
#messages { display:flex; flex-direction:column; gap:18px; padding-bottom:140px; }
.message { display:flex; gap:12px; align-items:flex-start; }
.message.user { justify-content:flex-end; }
.bubble { max-width:min(820px,90%); padding:12px 16px; border:1px solid var(--line);
          border-radius:14px; background:var(--panel); overflow-wrap:anywhere; }
.user .bubble { background:var(--user); border-color:#388bfd; }
.role { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em;
        margin-bottom:4px; }
.user .role { color:#dbeafe; }
.bubble p:first-child { margin-top:0; } .bubble p:last-child { margin-bottom:0; }
.bubble pre { margin:10px 0; padding:12px; overflow:auto; background:#0b0f14;
              border:1px solid var(--line); border-radius:8px; }
.bubble code { font:13px ui-monospace,SFMono-Regular,Consolas,monospace; }
.bubble :not(pre)>code { padding:2px 5px; border-radius:4px; background:#21262d; }
.bubble blockquote { margin:10px 0; padding-left:12px; border-left:3px solid var(--accent); color:#c9d1d9; }
.bubble table { border-collapse:collapse; margin:10px 0; width:100%; }
.bubble th,.bubble td { border:1px solid var(--line); padding:6px 8px; text-align:left; }
.bubble th { background:#21262d; }
.bubble a { color:var(--accent); }
.composer { position:fixed; left:50%; bottom:0; transform:translateX(-50%); width:min(1052px,100%);
            padding:14px 24px 20px; background:linear-gradient(transparent,var(--bg) 18%); }
.composer-inner { display:flex; gap:10px; padding:10px; border:1px solid var(--line);
                  border-radius:14px; background:var(--panel); }
textarea { flex:1; resize:none; min-height:48px; max-height:180px; border:0; outline:0;
           background:transparent; color:var(--text); font:inherit; padding:7px; }
button { border:1px solid var(--line); border-radius:9px; padding:9px 14px; color:var(--text);
         background:#21262d; cursor:pointer; font-weight:600; }
button:hover { border-color:var(--accent); } button:disabled { opacity:.55; cursor:wait; }
.send { background:var(--accent); color:#07111f; border-color:var(--accent); }
.tools { display:flex; gap:8px; margin-top:8px; align-items:center; }
.tools label { color:var(--muted); font-size:12px; }
.tools input { width:260px; border:1px solid var(--line); border-radius:6px; padding:5px 7px;
               background:var(--panel); color:var(--text); }
.empty { color:var(--muted); text-align:center; padding:12vh 20px; }
@media (max-width:700px) { main { padding:14px; } header { padding:15px; }
  .composer { padding:10px; } .tools input { width:170px; } }
</style>
</head>
<body>
<div class="app">
  <header><h1>LLMY Chat</h1><div id="status" class="status">Loading model…</div></header>
  <main><div id="messages"><div class="empty">Start a conversation below.</div></div></main>
  <div class="composer">
    <div class="composer-inner">
      <textarea id="prompt" placeholder="Message the model…" autofocus></textarea>
      <button id="send" class="send">Send</button>
    </div>
    <div class="tools">
      <label for="system">System</label>
      <input id="system" value="You are a helpful assistant." aria-label="System prompt">
      <button id="clear">Clear</button>
    </div>
  </div>
</div>
<script>
const messagesEl = document.getElementById('messages');
const promptEl = document.getElementById('prompt');
const systemEl = document.getElementById('system');
const sendEl = document.getElementById('send');
const statusEl = document.getElementById('status');
let conversation = [];

function esc(s) { return String(s).replace(/[&<>"']/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function inline(s) {
  const stash = [];
  const hold = x => { const id = `\u0000${stash.length}\u0000`; stash.push(x); return id; };
  s = s.replace(/`([^`]+)`/g, (_, x) => hold('<code>' + esc(x) + '</code>'));
  s = s.replace(/!\[([^\]]*)\]\((https?:\/\/[^)\s]+)\)/g,
    (_, alt, src) => hold('<img alt="' + esc(alt) + '" src="' + esc(src) + '">'));
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+|mailto:[^)\s]+)\)/g,
    (_, label, href) => hold('<a href="' + esc(href) + '" target="_blank" rel="noopener">' + esc(label) + '</a>'));
  s = esc(s);
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
       .replace(/__([^_]+)__/g, '<strong>$1</strong>')
       .replace(/\*([^*]+)\*/g, '<em>$1</em>')
       .replace(/_([^_]+)_/g, '<em>$1</em>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => stash[Number(i)]);
}
function markdown(md) {
  const code = [];
  let text = String(md || '').replace(/\r\n?/g, '\n').replace(/```([^\n]*)\n([\s\S]*?)```/g,
    (_, lang, body) => { const id = code.length; code.push('<pre><code class="language-' +
      esc(lang || 'text') + '">' + esc(body.replace(/\n$/, '')) + '</code></pre>'); return `\u0001${id}\u0001`; });
  const lines = text.split('\n'), out = []; let list = null;
  const close = () => { if (list) { out.push('</' + list + '>'); list = null; } };
  const cells = line => line.trim().replace(/^\||\|$/g, '').split('|').map(x => x.trim());
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const c = line.match(/^\u0001(\d+)\u0001$/);
    if (c) { close(); out.push(code[Number(c[1])]); continue; }
    if (!line.trim()) { close(); continue; }
    if (line.includes('|') && i + 1 < lines.length &&
        /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(lines[i + 1])) {
      close(); const headers = cells(line); i += 1; const rows = [];
      while (i + 1 < lines.length && lines[i + 1].includes('|') && lines[i + 1].trim()) rows.push(cells(lines[++i]));
      out.push('<table><thead><tr>' + headers.map(x => '<th>' + inline(x) + '</th>').join('') +
        '</tr></thead><tbody>' + rows.map(row => '<tr>' + headers.map((_, j) => '<td>' + inline(row[j] || '') + '</td>').join('') + '</tr>').join('') +
        '</tbody></table>'); continue;
    }
    let m = line.match(/^(#{1,6})\s+(.+)$/);
    if (m) { close(); const n = m[1].length; out.push(`<h${n}>${inline(m[2])}</h${n}>`); continue; }
    m = line.match(/^>\s?(.*)$/); if (m) { close(); out.push('<blockquote>' + inline(m[1]) + '</blockquote>'); continue; }
    m = line.match(/^[-*+]\s+(.+)$/); if (m) { if (list !== 'ul') { close(); out.push('<ul>'); list = 'ul'; } out.push('<li>' + inline(m[1]) + '</li>'); continue; }
    m = line.match(/^\d+[.)]\s+(.+)$/); if (m) { if (list !== 'ol') { close(); out.push('<ol>'); list = 'ol'; } out.push('<li>' + inline(m[1]) + '</li>'); continue; }
    if (/^([-*_])\s*\1\s*\1+$/.test(line.trim())) { close(); out.push('<hr>'); continue; }
    close(); out.push('<p>' + inline(line) + '</p>');
  }
  close(); return out.join('');
}
function render() {
  messagesEl.innerHTML = '';
  if (!conversation.length) { messagesEl.innerHTML = '<div class="empty">Start a conversation below.</div>'; return; }
  for (const msg of conversation) {
    if (msg.role === 'system') continue;
    const row = document.createElement('div'); row.className = 'message ' + msg.role;
    const bubble = document.createElement('div'); bubble.className = 'bubble';
    bubble.innerHTML = '<div class="role">' + esc(msg.role) + '</div>' +
      (msg.role === 'assistant' ? markdown(msg.content) : '<p>' + esc(msg.content).replace(/\n/g, '<br>') + '</p>');
    row.appendChild(bubble); messagesEl.appendChild(row);
  }
  window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
}
async function send() {
  const content = promptEl.value.trim(); if (!content || sendEl.disabled) return;
  if (!conversation.length && systemEl.value.trim()) conversation.push({role:'system', content:systemEl.value.trim()});
  conversation.push({role:'user', content}); promptEl.value = ''; render(); sendEl.disabled = true; statusEl.textContent = 'Generating…';
  try {
    const res = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({messages:conversation})});
    const data = await res.json(); if (!res.ok) throw new Error(data.error || 'Request failed');
    conversation.push(data.message); render();
    const s = data.stats || {}; statusEl.textContent = `${s.tps ? s.tps.toFixed(1) + ' tok/s · ' : ''}ready`;
  } catch (e) { statusEl.textContent = 'Error'; conversation.push({role:'assistant', content:'**Error:** ' + e.message}); render(); }
  finally { sendEl.disabled = false; promptEl.focus(); }
}
sendEl.onclick = send;
promptEl.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });
document.getElementById('clear').onclick = () => { conversation = []; render(); statusEl.textContent = 'ready'; promptEl.focus(); };
fetch('/api/health').then(r => r.json()).then(x => statusEl.textContent = x.ready ? 'ready' : 'loading…').catch(() => statusEl.textContent = 'offline');
</script>
</body>
</html>"""


class SamplerEngine:
    def __init__(self, checkpoint: str | None, config: str | None, device_name: str, *,
                 cuda_graph: bool, backend_module: str):
        self.sampler = importlib.import_module(backend_module)
        self.device = torch.device(device_name)
        ckpt = Path(checkpoint) if checkpoint else self.sampler.find_checkpoint()
        if not ckpt or not ckpt.is_file():
            raise FileNotFoundError("No checkpoint supplied and no local checkpoint was found.")
        if self.device.type == "cuda":
            major, _ = torch.cuda.get_device_capability(self.device)
            dtype = torch.bfloat16 if major >= 8 and torch.cuda.is_bf16_supported() else torch.float16
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            dtype = torch.float32
        self.dtype = dtype
        self.tokenizer = self.sampler.get_tokenizer()
        self.model, self.step = self.sampler.load_model(ckpt, self.device, dtype, config)
        self.sampler.set_fast_mhc(self.model, False)
        self.sampler.configure_adaptive_compute(self.model, SimpleNamespace(
            min_loops=None, max_loops=None, halt_threshold=None,
            halt_temperature=None, fixed_loops=None, adaptive=False,
        ))
        self.model = self.sampler.apply_all_triton_optimizations(self.model)
        self.cuda_graph = cuda_graph and self.device.type == "cuda"
        self.lock = threading.Lock()

    def reply(self, messages: list[dict[str, str]], *, max_tokens: int, temp: float,
              top_k: int, top_p: float, repetition: float) -> dict[str, Any]:
        with self.lock:
            result = self.sampler.sample(
                self.model, self.tokenizer, "", self.device, self.dtype,
                max_tokens=max_tokens, temp=temp, top_k=top_k, top_p=top_p,
                rep=repetition, show_prompt_ppl=False,
                use_cuda_graph=self.cuda_graph, messages=messages,
                emit_output=False,
            )
        return result or {"text": "", "tps": 0.0}


def _valid_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a non-empty list")
    result = []
    for item in value:
        if not isinstance(item, dict) or item.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("messages must contain only system, user, and assistant roles")
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be non-empty text")
        result.append({"role": str(item["role"]), "content": content})
    if result[-1]["role"] != "user":
        raise ValueError("the final message must be from the user")
    return result


def make_handler(engine: SamplerEngine, defaults: argparse.Namespace):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: Any, content_type: str = "application/json"):
            body = payload if isinstance(payload, bytes) else (
                json.dumps(payload).encode("utf-8") if content_type == "application/json" else payload.encode("utf-8")
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/api/health":
                self._send(200, {"ready": True, "step": engine.step})
            elif self.path in {"/", "/index.html"}:
                self._send(200, HTML, "text/html")
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/api/chat":
                self._send(404, {"error": "not found"}); return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size > 2_000_000: raise ValueError("request is too large")
                payload = json.loads(self.rfile.read(size))
                messages = _valid_messages(payload.get("messages"))
                result = engine.reply(
                    messages,
                    max_tokens=max(1, min(int(payload.get("max_tokens", defaults.max_tokens)), 2048)),
                    temp=float(payload.get("temp", defaults.temp)),
                    top_k=max(0, int(payload.get("top_k", defaults.top_k))),
                    top_p=float(payload.get("top_p", defaults.top_p)),
                    repetition=float(payload.get("repetition", defaults.repetition)),
                )
                self._send(200, {"message": {"role": "assistant", "content": result.get("text", "")}, "stats": result})
            except Exception as exc:
                self._send(400, {"error": str(exc)})

        def log_message(self, *_args):
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Markdown-rendering local chat UI")
    parser.add_argument("-c", "--checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition", type=float, default=1.1)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--backend", default="sample",
                        help="sampler module to load (sample or sample_standalone)")
    parser.add_argument("--no-browser", action="store_true",
                        help="keep the server headless instead of opening a browser tab")
    args = parser.parse_args()
    engine = SamplerEngine(
        args.checkpoint, args.config, args.device,
        cuda_graph=args.cuda_graph, backend_module=args.backend,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine, args))
    url = f"http://{args.host}:{args.port}"
    print(f"Chat UI ready at {url}", flush=True)
    if not args.no_browser and (os.name == "nt" or os.environ.get("DISPLAY")):
        threading.Timer(0.5, lambda: webbrowser.open_new_tab(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
