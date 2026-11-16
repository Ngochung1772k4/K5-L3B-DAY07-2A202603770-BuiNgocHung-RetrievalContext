"""Interactive Web UI for Day 7 Agentic RAG (Basic + Advanced Multimodal Challenge).

Run:
    .venv\\Scripts\\python.exe -m src.ui_app --port 8765
Then open http://127.0.0.1:8765 in your browser.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from src.adv_agent import make_adv_retriever, run_adv_agent
from src.adv_ingest import ADV_CHUNKS_PATH, ADV_MANIFEST_PATH
from src.config import AGENT_CASES_PATH, CHUNKS_PATH, RESULTS_DIR, ROOT
from src.ollama_agent import run_ollama_agent
from src.retrievers import Bm25Retriever, ChromaDenseRetriever, HybridRrfRetriever, TfidfRetriever

ADV_CASES_PATH = ROOT / "data" / "adv_cases.json"

_STATE: dict[str, Any] = {}


def get_state() -> dict[str, Any]:
    if not _STATE:
        basic_chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        adv_chunks = json.loads(ADV_CHUNKS_PATH.read_text(encoding="utf-8"))
        _STATE["basic_chunks"] = basic_chunks
        _STATE["basic_by_id"] = {c["id"]: c for c in basic_chunks}
        _STATE["adv_chunks"] = adv_chunks
        _STATE["adv_by_id"] = {c["id"]: c for c in adv_chunks}
        _STATE["basic_cases"] = json.loads(AGENT_CASES_PATH.read_text(encoding="utf-8"))
        _STATE["adv_cases"] = json.loads(ADV_CASES_PATH.read_text(encoding="utf-8"))
        _STATE["adv_manifest"] = json.loads(ADV_MANIFEST_PATH.read_text(encoding="utf-8"))
    return _STATE


def get_basic_retriever(mode: str):
    st = get_state()
    key = f"basic_ret_{mode}"
    if key not in st:
        chunks = st["basic_chunks"]
        if mode == "baseline":
            st[key] = TfidfRetriever(chunks, use_filters=False)
        else:
            dense = ChromaDenseRetriever(chunks, use_filters=True)
            bm25 = Bm25Retriever(chunks, use_filters=True)
            st[key] = HybridRrfRetriever(dense, bm25, fetch_k=12)
    return st[key]


def get_adv_retriever():
    st = get_state()
    if "adv_retriever" not in st:
        st["adv_retriever"] = make_adv_retriever(st["adv_chunks"])
    return st["adv_retriever"]


def _load_json_safe(path: Path) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def build_overview_payload() -> dict[str, Any]:
    st = get_state()
    adv_by_id = st["adv_by_id"]
    basic_by_id = st["basic_by_id"]

    def _enrich_traces(
        traces: list[dict[str, Any]] | None,
        reviews: list[dict[str, Any]] | None,
        cases_list: list[dict[str, Any]],
        chunks_map: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not traces:
            return []
        rev_map = {r["query_id"]: r for r in (reviews or [])}
        case_map = {c["id"]: c for c in cases_list}
        out = []
        for t in traces:
            qid = t["query_id"]
            c = case_map.get(qid, {})
            r = rev_map.get(qid, {})
            cited_details = []
            for cid in t.get("citations") or []:
                ch = chunks_map.get(cid)
                if ch:
                    cited_details.append(
                        {
                            "id": cid,
                            "page": ch["page"],
                            "region_type": ch.get("metadata", {}).get("region_type", "text"),
                            "crop_path": ch.get("metadata", {}).get("crop_path"),
                            "caption": ch.get("metadata", {}).get("caption"),
                            "text": ch["text"],
                        }
                    )
            ctx_details = []
            for cid in t.get("context_ids") or []:
                ch = chunks_map.get(cid)
                if ch:
                    ctx_details.append(
                        {
                            "id": cid,
                            "page": ch["page"],
                            "region_type": ch.get("metadata", {}).get("region_type", "text"),
                            "crop_path": ch.get("metadata", {}).get("crop_path"),
                            "caption": ch.get("metadata", {}).get("caption"),
                            "text": ch["text"],
                        }
                    )
            out.append(
                {
                    **t,
                    "expected_answer": c.get("expected_answer"),
                    "absence_reason": c.get("absence_reason"),
                    "gold_pages": c.get("gold_pages")
                    or [g["page"] for g in (c.get("gold") or []) if "page" in g],
                    "evidence_region": c.get("evidence_region"),
                    "review": r,
                    "cited_chunks": cited_details,
                    "context_chunks": ctx_details,
                }
            )
        return out

    return {
        "manifest": st["adv_manifest"],
        "presets": {
            "basic": st["basic_cases"],
            "advanced": [c for c in st["adv_cases"] if c["split"] in ("dev", "test")],
        },
        "retrieval_dev": _load_json_safe(RESULTS_DIR / "retrieval-dev" / "summary.json"),
        "retrieval_test": _load_json_safe(RESULTS_DIR / "retrieval-test" / "summary.json"),
        "runs": {
            "adv_test_v2": {
                "title": "Advanced Test v2 (Held-out, 27/30 = 90.0%)",
                "summary": _load_json_safe(RESULTS_DIR / "adv-agent-test" / "summary.json"),
                "traces": _enrich_traces(
                    _load_json_safe(RESULTS_DIR / "adv-agent-test" / "traces.json"),
                    _load_json_safe(RESULTS_DIR / "adv-agent-test" / "reviews.json"),
                    st["adv_cases"],
                    adv_by_id,
                ),
            },
            "adv_test_v1": {
                "title": "Advanced Test v1 (Archive, 24/30 = 80.0%)",
                "summary": _load_json_safe(RESULTS_DIR / "adv-agent-test-v1" / "summary.json"),
                "traces": _enrich_traces(
                    _load_json_safe(RESULTS_DIR / "adv-agent-test-v1" / "traces.json"),
                    _load_json_safe(RESULTS_DIR / "adv-agent-test-v1" / "reviews.json"),
                    st["adv_cases"],
                    adv_by_id,
                ),
            },
            "adv_dev": {
                "title": "Advanced Dev (18 câu, 15/18 = 83.3%)",
                "summary": _load_json_safe(RESULTS_DIR / "adv-agent-dev" / "summary.json"),
                "traces": _enrich_traces(
                    _load_json_safe(RESULTS_DIR / "adv-agent-dev" / "traces.json"),
                    None,
                    st["adv_cases"],
                    adv_by_id,
                ),
            },
            "basic_selected_test": {
                "title": "Basic Selected Agent — Test (9 câu)",
                "summary": _load_json_safe(RESULTS_DIR / "agent-selected-test" / "summary.json"),
                "traces": _enrich_traces(
                    _load_json_safe(RESULTS_DIR / "agent-selected-test" / "traces.json"),
                    _load_json_safe(RESULTS_DIR / "agent-selected-test" / "reviews.json"),
                    st["basic_cases"],
                    basic_by_id,
                ),
            },
            "basic_baseline_test": {
                "title": "Basic Baseline Agent — Test (9 câu)",
                "summary": _load_json_safe(RESULTS_DIR / "agent-baseline-test" / "summary.json"),
                "traces": _enrich_traces(
                    _load_json_safe(RESULTS_DIR / "agent-baseline-test" / "traces.json"),
                    _load_json_safe(RESULTS_DIR / "agent-baseline-test" / "reviews.json"),
                    st["basic_cases"],
                    basic_by_id,
                ),
            },
        },
    }


def execute_live_query(payload: dict[str, Any]) -> dict[str, Any]:
    st = get_state()
    mode = str(payload.get("mode") or "advanced")
    llm_model = str(payload.get("llm_model") or "deepseek-chat").strip()
    question = str(payload.get("question") or "").strip()
    if not question:
        raise ValueError("Question must not be empty.")

    use_verifier = llm_model in ("deepseek-chat", "qwen2.5:7b")
    if mode == "advanced":
        ret = get_adv_retriever()
        res = run_adv_agent(
            question,
            ret,
            st["adv_chunks"],
            model=llm_model,
            use_two_pass_verifier=use_verifier,
        )
        chunks_map = st["adv_by_id"]
    elif mode == "basic_selected":
        ret = get_basic_retriever("selected")
        res = run_ollama_agent(
            question,
            ret,
            st["basic_chunks"],
            model=llm_model,
            top_k=3,
            budget=2800,
            packing_mode="adaptive",
            enable_adjacent_tool=True,
            expand_neighbors=True,
            allow_chapter_filter=True,
        )
        chunks_map = st["basic_by_id"]
    else:
        ret = get_basic_retriever("baseline")
        res = run_ollama_agent(
            question,
            ret,
            st["basic_chunks"],
            model=llm_model,
            top_k=3,
            budget=800,
            packing_mode="first_fit",
            enable_adjacent_tool=False,
            expand_neighbors=False,
            allow_chapter_filter=False,
        )
        chunks_map = st["basic_by_id"]

    ctx_details = []
    for cid in res.get("context_ids") or []:
        ch = chunks_map.get(cid)
        if ch:
            ctx_details.append(
                {
                    "id": cid,
                    "page": ch["page"],
                    "region_type": ch.get("metadata", {}).get("region_type", "text"),
                    "crop_path": ch.get("metadata", {}).get("crop_path"),
                    "caption": ch.get("metadata", {}).get("caption"),
                    "text": ch["text"],
                }
            )
    return {**res, "context_chunks": ctx_details}


HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Day 7 — Chạy Thử Agentic RAG</title>
<style>
  :root {
    --bg: #f8fafc;
    --panel: #ffffff;
    --text: #0f172a;
    --muted: #64748b;
    --accent: #2563eb;
    --green: #16a34a;
    --amber: #d97706;
    --red: #dc2626;
    --border: #e2e8f0;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.55;
  }
  header {
    background: #ffffff; border-bottom: 1px solid var(--border);
    padding: 14px 28px; display: flex; justify-content: space-between; align-items: center;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
  }
  header h1 { margin: 0; font-size: 18px; font-weight: 700; color: var(--accent); }
  header .meta { font-size: 13px; color: var(--muted); }
  .container { padding: 24px 28px; max-width: 1380px; margin: 0 auto; }
  .grid-2 { display: grid; grid-template-columns: 420px 1fr; gap: 22px; align-items: start; }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 18px 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
  }
  .card h2, .card h3 { margin-top: 0; font-size: 15px; color: var(--text); }
  label { display: block; font-size: 12px; font-weight: 600; color: var(--muted); margin-bottom: 4px; }
  select, textarea, input {
    width: 100%; background: #ffffff; color: var(--text); border: 1px solid #cbd5e1;
    border-radius: 6px; padding: 9px 11px; font-size: 13px; margin-bottom: 14px; font-family: inherit;
  }
  select:focus, textarea:focus, input:focus {
    outline: none; border-color: var(--accent); box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15);
  }
  button.primary {
    width: 100%; background: var(--accent); color: #ffffff; border: none; padding: 11px 18px;
    border-radius: 6px; font-weight: 700; cursor: pointer; font-size: 14px;
  }
  button.primary:hover { background: #1d4ed8; }
  button.primary:disabled { opacity: 0.65; cursor: wait; }
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 700; text-transform: uppercase;
  }
  .badge.answered { background: #dcfce7; color: var(--green); }
  .badge.abstain { background: #fef3c7; color: var(--amber); }
  .badge.error { background: #fee2e2; color: var(--red); }
  .chunk-box {
    background: #f8fafc; border: 1px solid var(--border); border-radius: 8px;
    padding: 12px 14px; margin-bottom: 10px; font-size: 13px; white-space: pre-wrap;
  }
  .answer-box {
    background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
    padding: 14px 16px; margin-bottom: 16px; font-size: 14px; white-space: pre-wrap;
  }
  .gold-hint {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px;
    padding: 10px 12px; font-size: 12px; color: #166534; margin-bottom: 14px;
  }
  .crop-img {
    max-width: 100%; max-height: 440px; border-radius: 6px; border: 1px solid var(--border);
    margin-top: 10px; display: block; background: #ffffff;
  }
</style>
</head>
<body>
<header>
  <h1>Day 7 — Chạy Thử Agentic RAG (Live Playground)</h1>
  <div class="meta">Bùi Ngọc Hùng · MSSV: 2A202603770 · K5-L3B</div>
</header>

<div class="container">
  <div class="grid-2">
    <div class="card">
      <h3>Cấu hình & Câu hỏi</h3>
      <label>Chế độ Agent:</label>
      <select id="live-mode" onchange="populatePresets()">
        <option value="advanced">Advanced Multimodal Agent (200 trang, 1077 chunks, Table + Figure + VLM)</option>
        <option value="basic_selected">Basic Selected Agent (30 trang Day 6, Hybrid RRF + budget=2800)</option>
        <option value="basic_baseline">Basic Baseline Agent (30 trang Day 6, TF-IDF + budget=800)</option>
      </select>

      <label>Tốc độ / Backend LLM:</label>
      <select id="live-llm">
        <option value="deepseek-chat">⚡ Siêu nhanh: DeepSeek V3 API (~3–5 giây)</option>
        <option value="qwen2.5:3b">⚡ Nhanh Local GPU: Ollama qwen2.5:3b (~5–8 giây)</option>
        <option value="qwen2.5:7b">🔒 Chuẩn Báo Cáo Local: Ollama qwen2.5:7b 2-pass (~45–60 giây)</option>
      </select>

      <label>Chọn nhanh câu hỏi mẫu (hoặc tự nhập bên dưới):</label>
      <select id="live-preset" onchange="applyPreset()"></select>

      <label>Câu hỏi gửi Agent:</label>
      <textarea id="live-question" rows="5" placeholder="Nhập câu hỏi tiếng Anh về nội dung giáo trình Sociology..."></textarea>

      <div id="live-gold-hint" class="gold-hint"></div>

      <button class="primary" id="run-btn" onclick="runLiveQuery()">Chạy Agent Ngay</button>
    </div>

    <div id="live-output">
      <div class="card">
        <p style="color:var(--muted);margin:0;">Chọn một câu hỏi mẫu hoặc nhập câu hỏi mới ở khung bên trái rồi bấm <b>Chạy Agent Ngay</b> để xem quá trình gọi tool, câu trả lời có trích dẫn và ảnh crop bảng/biểu đồ gốc.</p>
      </div>
    </div>
  </div>
</div>

<script>
let DATA = null;

async function init() {
  const res = await fetch('/api/overview');
  DATA = await res.json();
  populatePresets();
}

function populatePresets() {
  const mode = document.getElementById('live-mode').value;
  const sel = document.getElementById('live-preset');
  const list = mode === 'advanced' ? DATA.presets.advanced : DATA.presets.basic;
  sel.innerHTML = list.map((c, i) => `<option value="${i}">[${c.split}/${c.category || (c.answerable ? 'ans' : 'unans')}] ${c.id}: ${c.question.slice(0, 82)}...</option>`).join('');
  applyPreset();
}

function applyPreset() {
  const mode = document.getElementById('live-mode').value;
  const list = mode === 'advanced' ? DATA.presets.advanced : DATA.presets.basic;
  const item = list[Number(document.getElementById('live-preset').value)] || list[0];
  if (!item) return;
  document.getElementById('live-question').value = item.question;
  document.getElementById('live-gold-hint').textContent = 'Đáp án chuẩn (Gold): ' + (item.expected_answer || item.absence_reason || '');
}

async function runLiveQuery() {
  const btn = document.getElementById('run-btn');
  const out = document.getElementById('live-output');
  const mode = document.getElementById('live-mode').value;
  const llm_model = document.getElementById('live-llm').value;
  const question = document.getElementById('live-question').value;
  btn.disabled = true;
  const t0 = performance.now();
  const timerId = setInterval(() => {
    const sec = ((performance.now() - t0) / 1000).toFixed(1);
    btn.textContent = `Đang chạy (${sec}s)...`;
  }, 100);
  out.innerHTML = `<div class="card"><h3>Đang thực thi truy vấn & gọi tool (${escapeHtml(llm_model)})...</h3><p style="color:var(--muted);margin:0;">Đang tìm kiếm trên ChromaDB + BM25 và tổng hợp câu trả lời có trích dẫn...</p></div>`;
  try {
    const resp = await fetch('/api/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode, llm_model, question})
    });
    const res = await resp.json();
    if (res.error) throw new Error(res.error);
    const eventsHtml = (res.events || []).map(ev => {
      if (ev.event === 'search') {
        return `<div class="chunk-box"><b>Tool Call:</b> <code>${escapeHtml(ev.tool || 'search_chunks')}</code> · Query: <code>${escapeHtml(ev.query || '')}</code></div>`;
      }
      return '';
    }).join('');
    const chunksHtml = (res.context_chunks || []).map(ch => `
      <div class="chunk-box">
        <div style="color:var(--accent);font-weight:700;margin-bottom:4px;">Chunk [${ch.id.slice(0,12)}...] · Trang ${ch.page} · Loại: ${ch.region_type}</div>
        <div>${escapeHtml(ch.text)}</div>
        ${ch.crop_path ? `<img class="crop-img" src="/file/${encodeURI(ch.crop_path)}">` : ''}
      </div>
    `).join('');
    out.innerHTML = `
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
          <h2 style="margin:0;">Kết quả chạy Agent (${escapeHtml(llm_model)})</h2>
          <div><span class="badge ${res.status}">${res.status}</span> · <b>${(res.elapsed_ms/1000).toFixed(2)}s</b> · Searches: ${res.num_searches} · Citations: ${(res.citations||[]).length}</div>
        </div>
        <div class="answer-box"><b>Câu trả lời:</b>\n${escapeHtml(res.answer)}</div>
        <h3>Các bước gọi Tool</h3>
        ${eventsHtml || '<p style="color:var(--muted)">Không có tool call.</p>'}
        <h3>Context Chunks & Ảnh Crop (${(res.context_chunks||[]).length})</h3>
        ${chunksHtml || '<p style="color:var(--muted)">Không có context chunk.</p>'}
      </div>
    `;
  } catch (err) {
    out.innerHTML = `<div class="card"><h3 style="color:var(--red);margin:0;">Lỗi: ${escapeHtml(String(err))}</h3></div>`;
  } finally {
    clearInterval(timerId);
    btn.disabled = false;
    btn.textContent = 'Chạy Agent Ngay';
  }
}

function escapeHtml(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

init();
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/overview":
            self._send_json(build_overview_payload())
            return
        if parsed.path.startswith("/file/"):
            rel = urllib.parse.unquote(parsed.path[len("/file/") :])
            target = (ROOT / rel).resolve()
            if not str(target).startswith(str(ROOT.resolve())) or not target.is_file():
                self.send_error(404, "File not found")
                return
            ctype, _ = mimetypes.guess_type(str(target))
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/run":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(raw) if raw else {}
                result = execute_live_query(payload)
                self._send_json(result, status=200)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=400)
            return
        self.send_error(404, "Not found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Day 7 Agentic RAG interactive Web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    get_state()
    get_adv_retriever()
    get_basic_retriever("selected")
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Day 7 Web UI listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
