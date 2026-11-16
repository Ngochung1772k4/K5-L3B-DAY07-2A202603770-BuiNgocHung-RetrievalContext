"""Run the real Ollama agent on dev or held-out test cases and save traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import AGENT_CASES_PATH, AGENT_MODEL, CHUNKS_PATH, RESULTS_DIR
from src.ollama_agent import run_ollama_agent
from src.retrievers import (
    Bm25Retriever,
    ChromaDenseRetriever,
    HybridRrfRetriever,
    LlmRerankRetriever,
    TfidfRetriever,
)


def make_agent_config(config_name: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if config_name == "baseline":
        return {
            "retriever": TfidfRetriever(chunks, use_filters=False),
            "top_k": 3,
            "budget": 800,
            "max_searches": 4,
            "packing_mode": "first_fit",
            "enable_adjacent_tool": False,
            "expand_neighbors": False,
            "allow_chapter_filter": False,
        }
    if config_name == "dense":
        return {
            "retriever": ChromaDenseRetriever(chunks, use_filters=True),
            "top_k": 3,
            "budget": 1500,
            "max_searches": 4,
            "packing_mode": "adaptive",
            "enable_adjacent_tool": True,
            "expand_neighbors": False,
            "allow_chapter_filter": True,
        }
    if config_name == "selected":
        dense = ChromaDenseRetriever(chunks, use_filters=True)
        bm25 = Bm25Retriever(chunks, use_filters=True)
        hybrid = HybridRrfRetriever(dense, bm25, fetch_k=12)
        return {
            "retriever": hybrid,
            "top_k": 3,
            "budget": 2800,
            "max_searches": 4,
            "packing_mode": "adaptive",
            "enable_adjacent_tool": True,
            "expand_neighbors": True,
            "allow_chapter_filter": True,
        }
    if config_name == "rerank":
        dense = ChromaDenseRetriever(chunks, use_filters=True)
        bm25 = Bm25Retriever(chunks, use_filters=True)
        hybrid = HybridRrfRetriever(dense, bm25, fetch_k=12)
        rerank = LlmRerankRetriever(hybrid, fetch_k=8, name="hybrid_llm_rerank")
        return {
            "retriever": rerank,
            "top_k": 3,
            "budget": 1800,
            "max_searches": 4,
            "packing_mode": "adaptive",
            "enable_adjacent_tool": True,
            "expand_neighbors": True,
            "allow_chapter_filter": True,
        }
    raise ValueError(f"Unknown agent config: {config_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 7 agent on dev or test split.")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--config", choices=("baseline", "dense", "hybrid", "selected", "rerank"), required=True)
    parser.add_argument("--model", default=AGENT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    all_cases = json.loads(AGENT_CASES_PATH.read_text(encoding="utf-8"))
    cases = [c for c in all_cases if c["split"] == args.split]
    if not cases:
        raise ValueError(f"No cases found for split={args.split}")

    cfg = make_agent_config(args.config, chunks)
    out_dir = args.output_dir or (RESULTS_DIR / f"agent-{args.config}-{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    traces: list[dict[str, Any]] = []
    for case in cases:
        try:
            res = run_ollama_agent(
                case["question"],
                cfg["retriever"],
                chunks,
                model=args.model,
                max_searches=cfg["max_searches"],
                top_k=cfg["top_k"],
                budget=cfg["budget"],
                packing_mode=cfg["packing_mode"],
                enable_adjacent_tool=cfg["enable_adjacent_tool"],
                expand_neighbors=cfg["expand_neighbors"],
                allow_chapter_filter=cfg["allow_chapter_filter"],
            )
            trace = {
                "query_id": case["id"],
                "split": case["split"],
                "answerable": case["answerable"],
                "question": case["question"],
                "config": args.config,
                "model": args.model,
                "error": None,
                **res,
            }
        except Exception as exc:  # noqa: BLE001
            trace = {
                "query_id": case["id"],
                "split": case["split"],
                "answerable": case["answerable"],
                "question": case["question"],
                "config": args.config,
                "model": args.model,
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
        print(
            f"[{args.config}/{args.split}] {case['id']} -> {trace['status']} "
            f"(searches={trace['num_searches']}, cites={len(trace['citations'])}, {trace['elapsed_ms']}ms)",
            flush=True,
        )

    (out_dir / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(traces)} traces to {out_dir / 'traces.json'}")


if __name__ == "__main__":
    main()
