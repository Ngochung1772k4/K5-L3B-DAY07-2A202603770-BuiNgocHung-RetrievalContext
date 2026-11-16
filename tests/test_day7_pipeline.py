"""Unit tests for Day 7 retrievers, context packing, and Ollama agent citation/error rules."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adv_agent import _quote_supported_in_chunk, pack_adv_context
from src.adv_ingest import chunk_sentences, table_to_markdown
from src.context_packing import expand_with_neighbors, pack_first_fit, pack_multi_turn
from src.eval_retrieval import chunk_hits_gold, context_covers_required
from src.ollama_agent import extract_citations, run_ollama_agent
from src.retrievers import Bm25Retriever, TfidfRetriever
from src.starter_adapters import Day6StarterRetriever, OllamaStarterController, run_agent


SAMPLE_CHUNKS = [
    {
        "id": "c1",
        "text": "Structural functionalism views society as interrelated parts.",
        "source_id": "book-1",
        "source": "https://example.org/book.pdf",
        "page": 23,
        "metadata": {"chapter": "1", "chunk_index": 0},
    },
    {
        "id": "c2",
        "text": "Landlords would not rent to them after they moved.",
        "source_id": "book-1",
        "source": "https://example.org/book.pdf",
        "page": 23,
        "metadata": {"chapter": "1", "chunk_index": 1},
    },
    {
        "id": "c3",
        "text": "Surveys capture reported feelings and thoughts in Chapter 2.",
        "source_id": "book-1",
        "source": "https://example.org/book.pdf",
        "page": 43,
        "metadata": {"chapter": "2", "chunk_index": 0},
    },
]


class Day7PipelineTests(unittest.TestCase):
    def test_tfidf_and_bm25_with_chapter_filter(self):
        tfidf = TfidfRetriever(SAMPLE_CHUNKS, use_filters=True)
        bm25 = Bm25Retriever(SAMPLE_CHUNKS, use_filters=True)
        res_t = tfidf.search("society interrelated", filters={"chapter": "1"}, top_k=3)
        res_b = bm25.search("surveys thoughts", filters={"chapter": "2"}, top_k=3)
        self.assertEqual([r["id"] for r in res_t], ["c1"])
        self.assertEqual([r["id"] for r in res_b], ["c3"])
        self.assertEqual(
            tfidf.search("surveys thoughts", filters={"chapter": "1"}, top_k=3), []
        )

    def test_neighbor_expansion_and_multi_turn_packing(self):
        expanded = expand_with_neighbors([SAMPLE_CHUNKS[0]], SAMPLE_CHUNKS, expand_top_n=1)
        self.assertEqual([c["id"] for c in expanded], ["c1", "c2"])

        ctx, ids = pack_multi_turn(
            [[SAMPLE_CHUNKS[0]], [SAMPLE_CHUNKS[2]]], budget=1000
        )
        # Latest turn (c3) and earlier turn (c1) are interleaved fairly.
        self.assertEqual(ids, ["c3", "c1"])
        self.assertIn("[c3 |", ctx)

        ctx_small, ids_small = pack_first_fit(SAMPLE_CHUNKS, budget=50)
        self.assertEqual((ctx_small, ids_small), ("", []))

    def test_normalized_gold_and_multi_evidence_context(self):
        chunk = {**SAMPLE_CHUNKS[0], "text": "Part one.\n\nPart two."}
        gold = [{"source_id": "book-1", "page": 23, "evidence": "Part one. Part two."}]
        self.assertTrue(chunk_hits_gold(chunk, gold))
        self.assertTrue(
            context_covers_required("Part one. Part two.", ["Part one.", "Part two."])
        )
        self.assertFalse(
            context_covers_required("Part one only.", ["Part one.", "Part two."])
        )

    def test_ollama_agent_search_cite_and_abstain(self):
        bm25 = Bm25Retriever(SAMPLE_CHUNKS, use_filters=True)
        responses = iter(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_chunks",
                                    "arguments": {"query": "structural functionalism", "chapter": "1"},
                                }
                            }
                        ],
                    },
                    "prompt_eval_count": 50,
                    "eval_count": 12,
                },
                {
                    "message": {
                        "role": "assistant",
                        "content": "Society has interrelated parts [c1].",
                    },
                    "prompt_eval_count": 80,
                    "eval_count": 15,
                },
            ]
        )
        with patch("src.ollama_agent.ollama_chat", side_effect=lambda *a, **kw: next(responses)):
            res = run_ollama_agent(
                "In Chapter 1, what is structural functionalism?",
                bm25,
                SAMPLE_CHUNKS,
                budget=1500,
            )
        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["citations"], ["c1"])
        self.assertEqual(res["num_searches"], 1)

    def test_ollama_agent_rejects_ungrounded_citation_and_propagates_error(self):
        bm25 = Bm25Retriever(SAMPLE_CHUNKS, use_filters=True)
        bad_cite_responses = iter(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_chunks",
                                    "arguments": {"query": "society"},
                                }
                            }
                        ],
                    }
                },
                {
                    "message": {
                        "role": "assistant",
                        "content": "Answer citing unknown chunk [not-in-context].",
                    }
                },
            ]
        )
        with patch("src.ollama_agent.ollama_chat", side_effect=lambda *a, **kw: next(bad_cite_responses)):
            res = run_ollama_agent("Question?", bm25, SAMPLE_CHUNKS, budget=1500)
        self.assertEqual(res["status"], "abstain")
        self.assertEqual(res["citations"], [])

        with patch("src.ollama_agent.ollama_chat", side_effect=ConnectionError("offline")):
            with self.assertRaises(ConnectionError):
                run_ollama_agent("Question?", bm25, SAMPLE_CHUNKS)

    def test_extract_citations(self):
        cites = extract_citations(
            "Claim A [c1] and B [c2, fake, c1] and single [c1].", {"c1", "c2"}
        )
        self.assertEqual(cites, ["c1", "c2"])

    def test_starter_adapters_work_with_starter_run_agent(self):
        bm25 = Bm25Retriever(SAMPLE_CHUNKS, use_filters=True)
        replies = iter(
            [
                {"message": {"content": '{"action": "search", "query": "structural functionalism"}'}},
                {"message": {"content": '{"action": "answer", "answer": "Related parts [c1].", "citations": ["c1"]}'}},
            ]
        )
        with patch("src.starter_adapters.ollama_chat", side_effect=lambda *a, **kw: next(replies)):
            res = run_agent(
                "What is structural functionalism?",
                OllamaStarterController(),
                Day6StarterRetriever(bm25),
                budget=1200,
            )
        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["citations"], ["c1"])

    def test_adv_table_markdown_and_quote_verification(self):
        md = table_to_markdown([["Country", "Rate"], ["Sweden", "1.9%"]])
        self.assertIn("| Country | Rate |", md)
        self.assertIn("| Sweden | 1.9% |", md)

        chunks = chunk_sentences("Sentence one. Sentence two.", max_size=500)
        self.assertEqual(chunks, ["Sentence one.\n\nSentence two."])

        self.assertTrue(_quote_supported_in_chunk("Sweden fertility 1.9%", md))
        self.assertFalse(_quote_supported_in_chunk("Japan fertility 1.4%", md))

        ctx, ids = pack_adv_context([{**SAMPLE_CHUNKS[0], "live_vlm_note": "Red bar 17%"}], budget=1000)
        self.assertEqual(ids, ["c1"])
        self.assertIn("[LIVE_VLM_INSPECTION]: Red bar 17%", ctx)

    def test_ui_overview_payload(self):
        from src.ui_app import build_overview_payload

        overview = build_overview_payload()
        self.assertIn("runs", overview)
        self.assertIn("adv_test_v2", overview["runs"])
        self.assertEqual(overview["runs"]["adv_test_v2"]["summary"]["passed_cases"], 27)
        self.assertEqual(overview["runs"]["adv_test_v2"]["summary"]["total_cases"], 30)


if __name__ == "__main__":
    unittest.main()
