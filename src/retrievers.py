"""Retrieval methods for Day 7: TF-IDF, BM25, Day 6 Chroma Dense, Hybrid RRF, and Cross-Encoder Rerank."""
from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from src.config import (
    AGENT_MODEL,
    DAY6_CHROMA_DIR,
    DAY6_COLLECTION,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LOCAL_CHROMA_DIR,
    OLLAMA_BASE_URL,
    RERANK_MODEL,
    STARTER_DIR,
)

sys.path.insert(0, str(STARTER_DIR))
from baseline import tokenize  # noqa: E402


class Retriever(Protocol):
    name: str
    index_seconds: float

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]: ...


def matches_filters(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    meta = chunk.get("metadata", {})
    for key, expected in filters.items():
        actual = chunk.get(key) if key in chunk else meta.get(key)
        if actual is None or str(actual) != str(expected):
            return False
    return True


def _with_ranks(
    rows: list[dict[str, Any]], strategy: str, top_k: int
) -> list[dict[str, Any]]:
    out = []
    for idx, row in enumerate(rows[:top_k], start=1):
        out.append({**row, "rank": idx, "strategy": strategy})
    return out


class TfidfRetriever:
    """TF-IDF lexical retriever compatible with starter/baseline.py."""

    def __init__(
        self, chunks: list[dict[str, Any]], *, use_filters: bool = True
    ) -> None:
        self.name = "tfidf"
        self.use_filters = use_filters
        t0 = time.perf_counter()
        self.chunks = chunks
        counts = [Counter(tokenize(c["text"])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {
            term: math.log((1 + len(chunks)) / (1 + count)) + 1
            for term, count in df.items()
        }
        self.vectors = [self._vector(row) for row in counts]
        self.index_seconds = time.perf_counter() - t0

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        vec = {
            term: count * self.idf[term]
            for term, count in counts.items()
            if term in self.idf
        }
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {term: v / norm for term, v in vec.items()}

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        qvec = self._vector(Counter(tokenize(query)))
        active_filters = filters if self.use_filters else None
        scored = []
        for chunk, vec in zip(self.chunks, self.vectors):
            if not matches_filters(chunk, active_filters):
                continue
            score = sum(val * vec.get(term, 0.0) for term, val in qvec.items())
            if score > 0:
                scored.append({**chunk, "score": float(score)})
        scored.sort(key=lambda r: (-r["score"], r["id"]))
        return _with_ranks(scored, self.name, top_k)


class Bm25Retriever:
    """Okapi BM25 lexical retriever over normalized word tokens."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        use_filters: bool = True,
    ) -> None:
        self.name = "bm25"
        self.k1 = k1
        self.b = b
        self.use_filters = use_filters
        t0 = time.perf_counter()
        self.chunks = chunks
        self.doc_tfs = [Counter(tokenize(c["text"])) for c in chunks]
        self.doc_lens = [sum(tf.values()) for tf in self.doc_tfs]
        self.avg_dl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 1.0
        df = Counter(term for tf in self.doc_tfs for term in tf)
        n_docs = len(chunks)
        self.idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self.index_seconds = time.perf_counter() - t0

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        q_terms = tokenize(query)
        active_filters = filters if self.use_filters else None
        scored = []
        for chunk, tf, dl in zip(self.chunks, self.doc_tfs, self.doc_lens):
            if not matches_filters(chunk, active_filters):
                continue
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                f = tf[term]
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
                score += self.idf.get(term, 0.0) * (f * (self.k1 + 1.0)) / denom
            if score > 0:
                scored.append({**chunk, "score": float(score)})
        scored.sort(key=lambda r: (-r["score"], r["id"]))
        return _with_ranks(scored, self.name, top_k)


class ChromaDenseRetriever:
    """Dense retriever backed by Day 6's Persistent Chroma store and FastEmbed."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        chroma_dir: Path | None = None,
        collection_name: str = DAY6_COLLECTION,
        model_name: str = EMBEDDING_MODEL,
        use_filters: bool = True,
    ) -> None:
        import chromadb
        from fastembed import TextEmbedding

        self.name = "dense_chroma"
        self.use_filters = use_filters
        self.chunks_by_id = {c["id"]: c for c in chunks}
        t0 = time.perf_counter()

        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        self.embedder = TextEmbedding(model_name=model_name)

        target_dir = chroma_dir or (
            DAY6_CHROMA_DIR if DAY6_CHROMA_DIR.exists() else LOCAL_CHROMA_DIR
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir = target_dir
        self.client = chromadb.PersistentClient(path=str(target_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "dimensions": EMBEDDING_DIM},
        )
        # If running from a fresh clone without Day 6's chroma directory, populate cache.
        if self.collection.count() == 0:
            vectors = [v.tolist() for v in self.embedder.embed([c["text"] for c in chunks], batch_size=32)]
            for offset in range(0, len(chunks), 500):
                batch = chunks[offset : offset + 500]
                self.collection.upsert(
                    ids=[c["id"] for c in batch],
                    embeddings=vectors[offset : offset + 500],
                    documents=[c["text"] for c in batch],
                    metadatas=[
                        {
                            "document_id": c["source_id"],
                            "source": c["source"],
                            "page": c["page"],
                            **{
                                k: v
                                for k, v in c["metadata"].items()
                                if isinstance(v, (str, int, float, bool))
                            },
                        }
                        for c in batch
                    ],
                )
        self.index_seconds = time.perf_counter() - t0

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        qvec = next(self.embedder.query_embed(query)).tolist()
        where = None
        if self.use_filters and filters:
            clauses = [{str(k): str(v)} for k, v in filters.items()]
            where = clauses[0] if len(clauses) == 1 else {"$and": clauses}

        total_avail = self.collection.count()
        n_results = min(top_k, total_avail) if total_avail else top_k
        res = self.collection.query(
            query_embeddings=[qvec],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        rows = []
        ids = res["ids"][0] if res.get("ids") else []
        docs = res["documents"][0] if res.get("documents") else []
        metas = res["metadatas"][0] if res.get("metadatas") else []
        dists = res["distances"][0] if res.get("distances") else []

        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            base = self.chunks_by_id.get(cid)
            if base is not None:
                item = {**base, "score": float(1.0 - dist)}
            else:
                item = {
                    "id": cid,
                    "text": doc,
                    "source_id": meta.get("document_id", ""),
                    "source": meta.get("source", ""),
                    "page": meta.get("page", 0),
                    "metadata": meta,
                    "score": float(1.0 - dist),
                }
            rows.append(item)
        return _with_ranks(rows, self.name, top_k)


class HybridRrfRetriever:
    """Hybrid Dense + BM25 retriever fused via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense: ChromaDenseRetriever,
        lexical: Bm25Retriever,
        *,
        fetch_k: int = 12,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ) -> None:
        self.name = "hybrid_rrf"
        self.dense = dense
        self.lexical = lexical
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.index_seconds = dense.index_seconds + lexical.index_seconds

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        k_cand = max(self.fetch_k, top_k)
        dense_hits = self.dense.search(query, filters=filters, top_k=k_cand)
        lex_hits = self.lexical.search(query, filters=filters, top_k=k_cand)

        scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}
        for rank, item in enumerate(dense_hits, start=1):
            cid = item["id"]
            by_id[cid] = item
            scores[cid] = scores.get(cid, 0.0) + self.dense_weight / (self.rrf_k + rank)
        for rank, item in enumerate(lex_hits, start=1):
            cid = item["id"]
            by_id.setdefault(cid, item)
            scores[cid] = scores.get(cid, 0.0) + self.lexical_weight / (self.rrf_k + rank)

        fused = [{**by_id[cid], "score": float(sc)} for cid, sc in scores.items()]
        fused.sort(key=lambda r: (-r["score"], r["id"]))
        return _with_ranks(fused, self.name, top_k)


class CrossEncoderRerankRetriever:
    """Two-stage retrieval: candidate retriever followed by ONNX Cross-Encoder reranker."""

    def __init__(
        self,
        base_retriever: Retriever,
        *,
        fetch_k: int = 12,
        rerank_model: str = RERANK_MODEL,
        name: str = "hybrid_rerank",
    ) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = name
        self.base_retriever = base_retriever
        self.fetch_k = fetch_k
        t0 = time.perf_counter()
        self.reranker = TextCrossEncoder(model_name=rerank_model)
        self.index_seconds = base_retriever.index_seconds + (time.perf_counter() - t0)

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        k_cand = max(self.fetch_k, top_k)
        candidates = self.base_retriever.search(query, filters=filters, top_k=k_cand)
        if not candidates:
            return []
        rerank_scores = list(
            self.reranker.rerank(query, [c["text"] for c in candidates])
        )
        rescored = [
            {**cand, "pre_rerank_score": cand["score"], "score": float(rscore)}
            for cand, rscore in zip(candidates, rerank_scores)
        ]
        rescored.sort(key=lambda r: (-r["score"], r["id"]))
        return _with_ranks(rescored, self.name, top_k)


class LlmRerankRetriever:
    """Two-stage retrieval: candidate retriever reranked listwise via local Ollama LLM."""

    def __init__(
        self,
        base_retriever: Retriever,
        *,
        fetch_k: int = 8,
        model: str = AGENT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        name: str = "hybrid_llm_rerank",
    ) -> None:
        self.name = name
        self.base_retriever = base_retriever
        self.fetch_k = fetch_k
        self.model = model
        self.base_url = base_url
        self.index_seconds = base_retriever.index_seconds

    def _rerank_indices(self, query: str, candidates: list[dict[str, Any]]) -> list[int]:
        items_text = "\n".join(
            f"[{i}] (p.{c['page']}, ch.{c.get('metadata', {}).get('chapter')}) {c['text'][:650]}"
            for i, c in enumerate(candidates)
        )
        prompt = (
            "Rank the candidate textbook excerpts from most relevant to least relevant "
            "for answering the query directly with exact evidence.\n"
            'Return ONLY valid JSON: {"ranked_indices": [<int>, ...]} containing unique candidate indices.\n\n'
            f"Query: {query}\n\nCandidates:\n{items_text}"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 8192},
        }
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"Ollama rerank HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}: {exc.reason}") from exc

        content = data.get("message", {}).get("content", "{}")
        parsed = json.loads(content)
        raw_order = parsed.get("ranked_indices", [])
        order: list[int] = []
        seen: set[int] = set()
        for idx in raw_order:
            if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                seen.add(idx)
                order.append(idx)
        for idx in range(len(candidates)):
            if idx not in seen:
                order.append(idx)
        return order

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        k_cand = max(self.fetch_k, top_k)
        candidates = self.base_retriever.search(query, filters=filters, top_k=k_cand)
        if len(candidates) <= 1:
            return _with_ranks(candidates, self.name, top_k)
        order = self._rerank_indices(query, candidates)
        n = len(order)
        reranked = [
            {
                **candidates[orig_idx],
                "pre_rerank_score": candidates[orig_idx]["score"],
                "score": float(n - pos) / float(n),
            }
            for pos, orig_idx in enumerate(order)
        ]
        return _with_ranks(reranked, self.name, top_k)

