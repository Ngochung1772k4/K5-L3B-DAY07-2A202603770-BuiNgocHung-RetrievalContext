"""Context selection and budget packing strategies for Day 7."""
from __future__ import annotations

from typing import Any


def format_chunk_block(chunk: dict[str, Any]) -> str:
    chapter = chunk.get("metadata", {}).get("chapter")
    chap_str = f" | ch {chapter}" if chapter else ""
    return f"[{chunk['id']} | {chunk['source']} | page {chunk['page']}{chap_str}]\n{chunk['text']}"


def pack_first_fit(
    results: list[dict[str, Any]], budget: int = 800
) -> tuple[str, list[str]]:
    """Baseline greedy packing in candidate order up to character budget."""
    blocks: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()
    for c in results:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        block = format_chunk_block(c)
        candidate = "\n\n".join(blocks + [block])
        if len(candidate) <= budget:
            blocks.append(block)
            ids.append(c["id"])
    return "\n\n".join(blocks), ids


def expand_with_neighbors(
    results: list[dict[str, Any]],
    all_chunks: list[dict[str, Any]],
    *,
    expand_top_n: int = 1,
    after_primary: bool = True,
) -> list[dict[str, Any]]:
    """Append immediate next/previous chunk on the same page for top-N hits."""
    by_page_idx: dict[tuple[str, int, int], dict[str, Any]] = {}
    for c in all_chunks:
        idx = c.get("metadata", {}).get("chunk_index")
        if idx is not None:
            by_page_idx[(c["source_id"], int(c["page"]), int(idx))] = c

    expanded: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(item: dict[str, Any]) -> None:
        if item["id"] not in seen:
            seen.add(item["id"])
            expanded.append(item)

    if after_primary:
        for c in results:
            _add(c)
        for c in results[:expand_top_n]:
            idx = c.get("metadata", {}).get("chunk_index")
            if idx is not None:
                for delta in (1, -1):
                    nb = by_page_idx.get(
                        (c["source_id"], int(c["page"]), int(idx) + delta)
                    )
                    if nb is not None:
                        _add({**nb, "score": c.get("score", 0.0) * 0.95, "neighbor_of": c["id"]})
        return expanded

    for rank_i, c in enumerate(results):
        _add(c)
        if rank_i < expand_top_n:
            idx = c.get("metadata", {}).get("chunk_index")
            if idx is not None:
                for delta in (1, -1):
                    nb = by_page_idx.get(
                        (c["source_id"], int(c["page"]), int(idx) + delta)
                    )
                    if nb is not None:
                        _add({**nb, "score": c.get("score", 0.0) * 0.95, "neighbor_of": c["id"]})
    return expanded


def pack_multi_turn(
    turn_results: list[list[dict[str, Any]]],
    *,
    budget: int = 1800,
    per_turn_cap: int | None = None,
) -> tuple[str, list[str]]:
    """Pack deduplicated chunks fairly across multiple search turns.

    Interleaves top hits from the latest turn and earlier turns so turn-1 chunks
    do not consume 100% of the budget before later searches can contribute.
    """
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    # Prioritize round-robin starting from the most recent search turn.
    reversed_turns = list(reversed(turn_results))
    max_len = max((len(t) for t in reversed_turns), default=0)
    for pos in range(max_len):
        for turn in reversed_turns:
            if per_turn_cap is not None and pos >= per_turn_cap:
                continue
            if pos < len(turn):
                c = turn[pos]
                if c["id"] not in seen:
                    seen.add(c["id"])
                    ordered.append(c)

    blocks: list[str] = []
    ids: list[str] = []
    for c in ordered:
        block = format_chunk_block(c)
        candidate = "\n\n".join(blocks + [block])
        if len(candidate) <= budget:
            blocks.append(block)
            ids.append(c["id"])
    return "\n\n".join(blocks), ids
