"""Inspect retrieval errors and per-query differences on dev traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import RESULTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=RESULTS_DIR / "retrieval-dev")
    args = parser.parse_args()

    summary = json.loads((args.dir / "summary.json").read_text(encoding="utf-8"))
    print("=== SUMMARY ===")
    for row in summary:
        print(
            f"{row['config_id']:30s} | Hit@1={row['hit1_str']:5s} | Hit@3={row['hit3_str']:5s} "
            f"| MRR={row['mrr']:.3f} | CtxHit={row['context_hit_str']:5s} "
            f"| AvgCtx={row['avg_context_chars']:6.1f} | MedMs={row['median_ms']:6.2f}"
        )

    traces_by_cfg = {
        row["config_id"]: json.loads(
            (args.dir / f"{row['config_id']}_traces.json").read_text(encoding="utf-8")
        )
        for row in summary
    }

    print("\n=== PER-QUERY MISSES ===")
    for cfg_id, traces in traces_by_cfg.items():
        misses = [
            t for t in traces if not t["hit1"] or not t["hit3"] or not t["context_hit"]
        ]
        print(f"\n[{cfg_id}] queries with any miss (Hit@1 / Hit@3 / ContextHit): {len(misses)}/{len(traces)}")
        for t in misses:
            gold_pages = [g["page"] for g in t["gold"]]
            res_pages = [
                f"{r['page']}(ch{r['chapter']},hit={r['hit']},s={r['score']:.3f})"
                for r in t["results"]
            ]
            print(
                f"  - {t['query_id']} (gold_p={gold_pages}, filter={t['filter_used']}): "
                f"hit1={t['hit1']}, hit3={t['hit3']}, ctx_hit={t['context_hit']} | "
                f"top3={res_pages} | ctx_ids={len(t['context_ids'])} ({t['context_chars']} chars)"
            )


if __name__ == "__main__":
    main()
