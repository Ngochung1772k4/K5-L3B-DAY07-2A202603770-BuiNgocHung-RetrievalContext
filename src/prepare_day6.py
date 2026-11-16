"""Prepare Day 7 datasets and benchmark cases from the Day 6 repository."""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.config import (
    AGENT_CASES_PATH,
    CHUNKS_PATH,
    DAY6_REPO_DIR,
    QUESTIONS_PATH,
    STARTER_DIR,
)

sys.path.insert(0, str(STARTER_DIR))
from baseline import validate  # noqa: E402


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _find_exact_quote_in_chunks(
    chunks: list[dict[str, Any]], source_id: str, page: int, raw_quote: str
) -> list[str]:
    """Return verbatim quote substring(s) matching raw_quote in Day 6 chunks."""
    page_chunks = [
        c for c in chunks if c["source_id"] == source_id and c["page"] == page
    ]
    for c in page_chunks:
        if raw_quote in c["text"]:
            return [raw_quote]
    norm_q = _norm(raw_quote)
    for c in page_chunks:
        # Match across \n\n sentence joins produced by Day 6 by_paragraph.
        pattern = r"\s+".join(re.escape(tok) for tok in norm_q.split(" "))
        m = re.search(pattern, c["text"])
        if m:
            return [m.group(0)]
    return []


def build_chunks(day6_dir: Path) -> list[dict[str, Any]]:
    snap_path = day6_dir / "starter" / "results" / "chunk_snapshots.jsonl"
    raw_rows = [
        json.loads(line)
        for line in snap_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    b_rows = [row for row in raw_rows if row.get("strategy") == "B"]
    b_rows.sort(key=lambda r: (r["page"], r["char_start"]))

    per_page_counter: dict[int, int] = defaultdict(int)
    chunks: list[dict[str, Any]] = []
    for row in b_rows:
        page = int(row["page"])
        idx = per_page_counter[page]
        per_page_counter[page] += 1
        chunks.append(
            {
                "id": row["chunk_id"],
                "text": row["text"],
                "source_id": row["document_id"],
                "source": row["source"],
                "page": page,
                "metadata": {
                    "chapter": str(row["chapter"]),
                    "chunk_index": idx,
                    "char_start": int(row["char_start"]),
                    "char_end": int(row["char_end"]),
                    "char_count": len(row["text"]),
                    "language": "en",
                    "attribution": row["attribution"],
                },
            }
        )
    return chunks


def build_questions(
    day6_dir: Path, chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    raw_qs = json.loads(
        (day6_dir / "benchmark" / "questions.json").read_text(encoding="utf-8")
    )
    questions: list[dict[str, Any]] = []
    for q in raw_qs:
        quote = q["evidence_quote"]
        matched = _find_exact_quote_in_chunks(
            chunks, q["gold_source_id"], int(q["gold_page"]), quote
        )
        if matched:
            gold = [
                {
                    "source_id": q["gold_source_id"],
                    "page": int(q["gold_page"]),
                    "evidence": matched[0],
                }
            ]
            required = [matched[0]]
        elif q["id"] == "q03":
            # q03's evidence in Day 6 Strategy B spans two adjacent chunks on page 18.
            part1 = "Ashley found herself experiencing direct discrimination because of her sexual orientation."
            part2 = "Landlords would not rent to them."
            gold = [
                {"source_id": q["gold_source_id"], "page": 18, "evidence": part1},
                {"source_id": q["gold_source_id"], "page": 18, "evidence": part2},
            ]
            required = [part1, part2]
        else:
            raise ValueError(f"Cannot resolve Day 6 evidence for {q['id']}")

        item: dict[str, Any] = {
            "id": q["id"],
            "split": q["split"],
            "type": q.get("type", "direct"),
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold": gold,
            "required_evidences": required,
        }
        if q.get("filter"):
            item["filter"] = q["filter"]
        questions.append(item)

    # Add two Chapter 1 held-out test questions from unused pages (14 and 19)
    # so the test split balances Chapter 1 and Chapter 2.
    extra_test = [
        {
            "id": "q26",
            "split": "test",
            "type": "metadata_filter",
            "question": "In Chapter 1, how do sociologists define a society?",
            "expected_answer": "A group of people who live in a defined geographic area, interact with one another, and share a common culture.",
            "gold": [
                {
                    "source_id": "openstax-introduction-to-sociology-2e",
                    "page": 14,
                    "evidence": "A group of people who live in a defined geographic area, who interact with one another, and who share a common culture is what sociologists call a society.",
                }
            ],
            "required_evidences": [
                "A group of people who live in a defined geographic area, who interact with one another, and who share a common culture is what sociologists call a society."
            ],
            "filter": {"chapter": "1"},
        },
        {
            "id": "q27",
            "split": "test",
            "type": "direct",
            "question": "Who first coined the term 'sociology' in 1780, and who reinvented it in 1838?",
            "expected_answer": "French essayist Emmanuel-Joseph Sieyès first coined the term in 1780, and Auguste Comte reinvented it in 1838.",
            "gold": [
                {
                    "source_id": "openstax-introduction-to-sociology-2e",
                    "page": 19,
                    "evidence": "The term sociology was first coined in 1780 by the French essayist Emmanuel-Joseph Sieyès (1748–1836) in an unpublished manuscript (Fauré et al.\n\n1999).\n\nIn 1838, the term was reinvented by Auguste Comte (1798–1857).",
                }
            ],
            "required_evidences": [
                "The term sociology was first coined in 1780 by the French essayist Emmanuel-Joseph Sieyès (1748–1836) in an unpublished manuscript (Fauré et al.\n\n1999).\n\nIn 1838, the term was reinvented by Auguste Comte (1798–1857)."
            ],
        },
    ]
    questions.extend(extra_test)
    return questions


def build_agent_cases(
    day6_dir: Path, questions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    pages_path = day6_dir / "starter" / "data" / "pages.jsonl"
    pages_text = "\n".join(
        json.loads(line)["text"]
        for line in pages_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ).casefold()

    cases: list[dict[str, Any]] = []
    for q in questions:
        case: dict[str, Any] = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
            "expected_answer": q["expected_answer"],
            "gold": q["gold"],
        }
        if q.get("filter"):
            case["filter"] = q["filter"]
        cases.append(case)

    unanswerable = [
        {
            "id": "u_dev_01",
            "split": "dev",
            "question": "According to the textbook, what does the Sapir-Whorf hypothesis state about language and thought?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_keywords": ["sapir", "whorf", "sapir-whorf"],
            "absence_reason": "Checked all 30 pages of starter/data/pages.jsonl (Ch. 1-2): 0 occurrences of 'sapir' or 'whorf'; this belongs to Chapter 3 (Culture), outside the indexed corpus.",
        },
        {
            "id": "u_dev_02",
            "split": "dev",
            "question": "What did Charles Horton Cooley mean by the 'looking-glass self'?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_keywords": ["cooley", "looking-glass"],
            "absence_reason": "Checked all 30 pages of starter/data/pages.jsonl (Ch. 1-2): 0 occurrences of 'cooley' or 'looking-glass'; this belongs to Chapter 5 (Socialization).",
        },
        {
            "id": "u_dev_03",
            "split": "dev",
            "question": "What did Philip Zimbardo's Stanford Prison Experiment reveal about research ethics?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_keywords": ["zimbardo", "stanford prison"],
            "absence_reason": "Checked all 30 pages of starter/data/pages.jsonl (Ch. 1-2): 0 occurrences of 'zimbardo' or 'stanford prison'; p.51-52 only cover the ASA code of ethics and Max Weber's value neutrality.",
        },
        {
            "id": "u_test_01",
            "split": "test",
            "question": "How does the textbook define ethnocentrism and cultural relativism?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_keywords": ["ethnocentrism", "cultural relativism"],
            "absence_reason": "Checked all 30 pages of starter/data/pages.jsonl (Ch. 1-2): 0 occurrences of 'ethnocentrism' or 'cultural relativism'; belongs to Chapter 3 (Culture).",
        },
        {
            "id": "u_test_02",
            "split": "test",
            "question": "In Chapter 2, how does the textbook define snowball sampling compared with random sampling?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_keywords": ["snowball"],
            "absence_reason": "Checked all 30 pages of starter/data/pages.jsonl (Ch. 1-2): random sample is defined on p.43/53, but 'snowball' has 0 occurrences across all 30 pages.",
        },
    ]

    for item in unanswerable:
        for kw in item["absence_keywords"]:
            if kw in pages_text:
                raise ValueError(f"Keyword {kw!r} unexpectedly found in Day 6 pages for {item['id']}")
        cases.append({k: v for k, v in item.items() if k != "absence_keywords"})

    return cases


def main() -> None:
    chunks = build_chunks(DAY6_REPO_DIR)
    questions = build_questions(DAY6_REPO_DIR, chunks)
    validate(chunks, [q for q in questions if q["split"] == "dev"])
    validate(chunks, [q for q in questions if q["split"] == "test"])
    cases = build_agent_cases(DAY6_REPO_DIR, questions)

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    QUESTIONS_PATH.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    AGENT_CASES_PATH.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "chunks": len(chunks),
                "questions_dev": sum(q["split"] == "dev" for q in questions),
                "questions_test": sum(q["split"] == "test" for q in questions),
                "cases_dev": sum(c["split"] == "dev" for c in cases),
                "cases_test": sum(c["split"] == "test" for c in cases),
                "unanswerable_dev": sum(
                    c["split"] == "dev" and not c["answerable"] for c in cases
                ),
                "unanswerable_test": sum(
                    c["split"] == "test" and not c["answerable"] for c in cases
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
