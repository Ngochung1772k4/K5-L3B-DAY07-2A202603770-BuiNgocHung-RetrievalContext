"""Run Advanced Challenge multimodal agent evaluation on dev or test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.adv_agent import make_adv_retriever, run_adv_agent
from src.adv_ingest import ADV_CHUNKS_PATH, VLM_MODEL
from src.config import AGENT_MODEL, RESULTS_DIR, ROOT

ADV_CASES_PATH = ROOT / "data" / "adv_cases.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Advanced multimodal agent.")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--cases", type=Path, default=ADV_CASES_PATH)
    parser.add_argument("--model", default=AGENT_MODEL)
    parser.add_argument("--vlm-model", default=VLM_MODEL)
    parser.add_argument("--budget", type=int, default=6500)
    parser.add_argument("--max-searches", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-verifier", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    chunks = json.loads(ADV_CHUNKS_PATH.read_text(encoding="utf-8"))
    all_cases = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = [c for c in all_cases if c["split"] == args.split]
    if not cases:
        raise ValueError(f"No advanced cases in split={args.split}")

    retriever = make_adv_retriever(chunks)
    out_dir = args.output_dir or (RESULTS_DIR / f"adv-agent-{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    traces: list[dict[str, Any]] = []
    for case in cases:
        try:
            res = run_adv_agent(
                case["question"],
                retriever,
                chunks,
                model=args.model,
                vlm_model=args.vlm_model,
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.budget,
                use_two_pass_verifier=not args.no_verifier,
            )
            trace = {
                "query_id": case["id"],
                "split": case["split"],
                "category": case["category"],
                "answerable": case["answerable"],
                "question": case["question"],
                "model": args.model,
                "vlm_model": args.vlm_model,
                "error": None,
                **res,
            }
        except Exception as exc:  # noqa: BLE001
            trace = {
                "query_id": case["id"],
                "split": case["split"],
                "category": case["category"],
                "answerable": case["answerable"],
                "question": case["question"],
                "model": args.model,
                "vlm_model": args.vlm_model,
                "status": "error",
                "answer": "",
                "citations": [],
                "events": [],
                "context": "",
                "context_ids": [],
                "num_searches": 0,
                "elapsed_ms": 0.0,
                "prompt_tokens": 0,
                "eval_tokens": 0,
                "service_cost_usd": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        traces.append(trace)
        (out_dir / "traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[adv/{args.split}] {case['id']} ({case['category']}) -> {trace['status']} "
            f"(searches={trace['num_searches']}, cites={len(trace['citations'])}, {trace['elapsed_ms']}ms)",
            flush=True,
        )

    print(f"Saved {len(traces)} traces to {out_dir / 'traces.json'}", flush=True)


if __name__ == "__main__":
    main()
