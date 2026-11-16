"""Helper to prepare an empty manual audit sheet and print traces for human review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect agent traces for manual review.")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--template-out", type=Path, default=None)
    args = parser.parse_args()

    traces = json.loads(args.traces.read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in json.loads(args.cases.read_text(encoding="utf-8"))}

    template = []
    for t in traces:
        qid = t["query_id"]
        c = cases[qid]
        print("=" * 80)
        print(f"ID: {qid} | split={t['split']} | answerable={c['answerable']} | status={t['status']} | searches={t['num_searches']}")
        print(f"Q: {t['question']}")
        if c["answerable"]:
            print(f"Expected: {c.get('expected_answer')}")
            print(f"Gold: {c.get('gold')}")
        else:
            print(f"Absence reason: {c.get('absence_reason')}")
        print(f"Citations ({len(t['citations'])}): {t['citations']}")
        print(f"Answer: {t['answer']}")
        print(f"Context IDs: {t['context_ids']}")

        entry: dict = {
            "query_id": qid,
            "citation_support": [True for _ in t["citations"]],
            "notes": "",
        }
        if c["answerable"]:
            entry["context_grade"] = 0
            entry["answer_grade"] = 0
        else:
            entry["abstention_correct"] = t["status"] == "abstain"
        template.append(entry)

    if args.template_out:
        args.template_out.parent.mkdir(parents=True, exist_ok=True)
        args.template_out.write_text(
            json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
