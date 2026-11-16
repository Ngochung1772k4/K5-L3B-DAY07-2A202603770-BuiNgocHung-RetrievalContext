"""Evaluate retrieval methods and context-packing configurations on dev or held-out test."""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import unicodedata
from pathlib import Path
from typing import Any

from src.config import CHUNKS_PATH, QUESTIONS_PATH, RESULTS_DIR
from src.context_packing import expand_with_neighbors, pack_first_fit
from src.retrievers import (
    Bm25Retriever,
    ChromaDenseRetriever,
    HybridRrfRetriever,
    LlmRerankRetriever,
    TfidfRetriever,
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def chunk_hits_gold(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    norm_text = _norm(chunk["text"])
    return any(
        chunk["source_id"] == g["source_id"]
        and int(chunk["page"]) == int(g["page"])
        and _norm(g["evidence"]) in norm_text
        for g in gold
    )


def context_covers_required(
    context: str, required_evidences: list[str]
) -> bool:
    norm_ctx = _norm(context)
    return all(_norm(ev) in norm_ctx for ev in required_evidences)


def evaluate_config(
    *,
    config_id: str,
    description: str,
    retriever: Any,
    chunks: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    top_k: int = 3,
    budget: int = 800,
    use_question_filter: bool = False,
    expand_neighbors: bool = False,
    expand_top_n: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    for q in questions:
        q_filter = q.get("filter") if use_question_filter else None
        t0 = time.perf_counter()
        ranked_top3 = retriever.search(q["question"], filters=q_filter, top_k=top_k)
        candidates = (
            expand_with_neighbors(ranked_top3, chunks, expand_top_n=expand_top_n)
            if expand_neighbors
            else ranked_top3
        )
        context, ctx_ids = pack_first_fit(candidates, budget=budget)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        marked = [{**c, "hit": chunk_hits_gold(c, q["gold"])} for c in ranked_top3]
        first_hit_rank = next(
            (idx for idx, c in enumerate(marked, start=1) if c["hit"]), None
        )
        rr = (1.0 / first_hit_rank) if first_hit_rank else 0.0
        required = q.get("required_evidences") or [q["gold"][0]["evidence"]]
        ctx_hit = context_covers_required(context, required)

        traces.append(
            {
                "query_id": q["id"],
                "split": q["split"],
                "type": q.get("type", "direct"),
                "question": q["question"],
                "filter_used": q_filter,
                "config_id": config_id,
                "strategy": retriever.name,
                "top_k": top_k,
                "budget_chars": budget,
                "expand_neighbors": expand_neighbors,
                "gold": q["gold"],
                "results": [
                    {
                        "id": c["id"],
                        "rank": c.get("rank"),
                        "score": c["score"],
                        "page": c["page"],
                        "chapter": c.get("metadata", {}).get("chapter"),
                        "hit": c["hit"],
                        "text": c["text"],
                    }
                    for c in marked
                ],
                "context": context,
                "context_ids": ctx_ids,
                "context_chars": len(context),
                "elapsed_ms": elapsed_ms,
                "hit1": bool(marked and marked[0]["hit"]),
                "hit3": any(c["hit"] for c in marked),
                "reciprocal_rank": rr,
                "context_hit": ctx_hit,
            }
        )

    n = len(traces)
    h1 = sum(t["hit1"] for t in traces)
    h3 = sum(t["hit3"] for t in traces)
    ch = sum(t["context_hit"] for t in traces)
    summary = {
        "config_id": config_id,
        "strategy": retriever.name,
        "description": description,
        "split": traces[0]["split"] if traces else "",
        "n": n,
        "top_k": top_k,
        "budget_chars": budget,
        "use_question_filter": use_question_filter,
        "expand_neighbors": expand_neighbors,
        "hit1": h1,
        "hit1_str": f"{h1}/{n}",
        "hit1_rate": round(h1 / n, 4) if n else 0.0,
        "hit3": h3,
        "hit3_str": f"{h3}/{n}",
        "hit3_rate": round(h3 / n, 4) if n else 0.0,
        "mrr": round(statistics.mean(t["reciprocal_rank"] for t in traces), 4) if n else 0.0,
        "context_hit": ch,
        "context_hit_str": f"{ch}/{n}",
        "context_hit_rate": round(ch / n, 4) if n else 0.0,
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1) if n else 0.0,
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2) if n else 0.0,
        "index_seconds": round(retriever.index_seconds, 4),
        "service_cost_usd": 0.0,
    }
    return summary, traces


def build_retrievers(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    tfidf_nofilter = TfidfRetriever(chunks, use_filters=False)
    bm25_nofilter = Bm25Retriever(chunks, use_filters=False)
    bm25_filtered = Bm25Retriever(chunks, use_filters=True)
    dense_nofilter = ChromaDenseRetriever(chunks, use_filters=False)
    dense_filtered = ChromaDenseRetriever(chunks, use_filters=True)
    hybrid_nofilter = HybridRrfRetriever(dense_nofilter, bm25_nofilter, fetch_k=12)
    hybrid_filtered = HybridRrfRetriever(dense_filtered, bm25_filtered, fetch_k=12)
    rerank_filtered = LlmRerankRetriever(
        hybrid_filtered, fetch_k=8, name="hybrid_llm_rerank"
    )
    return {
        "tfidf_nofilter": tfidf_nofilter,
        "bm25_nofilter": bm25_nofilter,
        "dense_nofilter": dense_nofilter,
        "dense_filtered": dense_filtered,
        "hybrid_nofilter": hybrid_nofilter,
        "hybrid_filtered": hybrid_filtered,
        "rerank_filtered": rerank_filtered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Day 7 retrieval configs.")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Optional subset of config IDs to run (e.g., baseline_tfidf selected_config).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    all_questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    selected_qs = [q for q in all_questions if q["split"] == args.split]
    if not selected_qs:
        raise ValueError(f"No questions found for split={args.split}")

    retrievers = build_retrievers(chunks)
    specs = [
        (
            "baseline_tfidf",
            "Baseline: TF-IDF lexical retrieval, top_k=3, no filter, budget=800",
            retrievers["tfidf_nofilter"],
            {"top_k": 3, "budget": 800, "use_question_filter": False, "expand_neighbors": False},
        ),
        (
            "method2_bm25",
            "Method 2 (Lexical BM25): Okapi BM25, top_k=3, no filter, budget=800",
            retrievers["bm25_nofilter"],
            {"top_k": 3, "budget": 800, "use_question_filter": False, "expand_neighbors": False},
        ),
        (
            "method3_dense",
            "Method 3 (Dense Day 6): Chroma day6_b + MiniLM-L6-v2 (384d), top_k=3, no filter, budget=800",
            retrievers["dense_nofilter"],
            {"top_k": 3, "budget": 800, "use_question_filter": False, "expand_neighbors": False},
        ),
        (
            "method4_hybrid_rrf",
            "Method 4 (Hybrid RRF): Dense + BM25 (fetch_k=12, k=60), top_k=3, no filter, budget=800",
            retrievers["hybrid_nofilter"],
            {"top_k": 3, "budget": 800, "use_question_filter": False, "expand_neighbors": False},
        ),
        (
            "imp1_hybrid_filter_ctx1800",
            "Improvement 1: Hybrid RRF + chapter filter + context budget=1800",
            retrievers["hybrid_filtered"],
            {"top_k": 3, "budget": 1800, "use_question_filter": True, "expand_neighbors": False},
        ),
        (
            "imp2_hybrid_llm_rerank_nb",
            "Improvement 2 (Method 5): Hybrid RRF (fetch_k=8) + qwen2.5:7b rerank + chapter filter + neighbor expansion + budget=1800",
            retrievers["rerank_filtered"],
            {"top_k": 3, "budget": 1800, "use_question_filter": True, "expand_neighbors": True, "expand_top_n": 1},
        ),
        (
            "imp3_hybrid_filter_ctx2800",
            "Improvement 3 (Selected candidate): Hybrid RRF + chapter filter + post-primary neighbor expansion + budget=2800",
            retrievers["hybrid_filtered"],
            {"top_k": 3, "budget": 2800, "use_question_filter": True, "expand_neighbors": True, "expand_top_n": 1},
        ),
    ]

    if args.configs:
        allowed = set(args.configs)
        specs = [s for s in specs if s[0] in allowed]
        if not specs:
            raise ValueError(f"No matching configs in {allowed}")

    out_dir = args.output_dir or (RESULTS_DIR / f"retrieval-{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for cfg_id, desc, ret, kwargs in specs:
        summary, traces = evaluate_config(
            config_id=cfg_id,
            description=desc,
            retriever=ret,
            chunks=chunks,
            questions=selected_qs,
            **kwargs,
        )
        summaries.append(summary)
        (out_dir / f"{cfg_id}_traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    (out_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
