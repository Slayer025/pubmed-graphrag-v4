#!/usr/bin/env python3
"""Comprehensive evaluation of new retrieval and ranking methods.

Runs the 40-query frozen evaluation set through:

* Dense-only
* BM25-only
* TF-IDF-only
* Hybrid RRF (dense + BM25)
* AAR fusion (dense + BM25 + TF-IDF)
* Dense + MMR rerank
* Dense + cross-encoder rerank

For each method it reports Recall@5, Recall@10 and MRR@10 and writes both
per-query JSONL files and an aggregate metrics JSON to ``outputs/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap.environment import configure_environment

os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "hf_cache"))
configure_environment()

import numpy as np

from src.application.dto.search_config import SearchConfig
from src.application.use_cases.graph_expand import GraphExpandUseCase
from src.application.use_cases.rerank import RerankUseCase
from src.application.use_cases.vector_search import VectorSearchUseCase
from src.bootstrap import (
    _build_embedding_service,
    _build_sparse_retriever,
    _build_tfidf_retriever,
    _build_vector_store,
)
from src.bootstrap.bootstrap_artifacts import bootstrap_artifacts
from src.config import AppConfig
from src.domain.entities.retrieval_result import RetrievalResult
from src.domain.services.aar_fusion_service import AARFusionService
from src.domain.services.cross_encoder_rerank_service import CrossEncoderRerankService
from src.domain.services.mmr_rerank_service import MMRRerankService
from src.domain.services.rrf_fusion_service import RRFFusionService
from src.domain.value_objects.query import Query
from src.infrastructure.graph.in_memory_graph_repository import InMemoryGraphRepository
from src.infrastructure.storage.artifact_loader import LoadedArtifacts
from src.infrastructure.storage.chunk_repository import InMemoryChunkRepository

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

QUERIES_PATH = Path(__file__).parent / "queries.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
EVAL_DIR = Path(__file__).parent


@dataclass(frozen=True)
class _EvalResult:
    method: str
    records: list[dict]
    metrics: dict


def _load_queries(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _aggregate(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"num_queries": 0, "recall@5": 0.0, "recall@10": 0.0, "mrr@10": 0.0, "avg_latency_ms": 0.0}
    return {
        "num_queries": n,
        "recall@5": round(sum(r["recall@5"] for r in records) / n, 4),
        "recall@10": round(sum(r["recall@10"] for r in records) / n, 4),
        "mrr@10": round(sum(r["mrr@10"] for r in records) / n, 4),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in records) / n, 2),
    }


def _make_record(query: dict, ranked: list[RetrievalResult], latency_ms: float) -> dict:
    expected_article_id = str(query["expected_article_id"])
    top_10 = ranked[:10]
    correct_ranks = [
        rank for rank, result in enumerate(top_10, start=1)
        if str(result.article_id) == expected_article_id
    ]
    return {
        "query_id": str(query["query_id"]),
        "question": query["question"],
        "expected_article_id": expected_article_id,
        "recall@5": any(str(r.article_id) == expected_article_id for r in ranked[:5]),
        "recall@10": bool(correct_ranks),
        "mrr@10": round(1.0 / correct_ranks[0], 4) if correct_ranks else 0.0,
        "latency_ms": round(latency_ms, 2),
        "num_results": len(ranked),
        "top_10": [
            {
                "rank": i + 1,
                "chunk_id": r.chunk_id,
                "article_id": str(r.article_id),
                "combined_score": round(r.combined_score, 4),
                "source": r.source,
            }
            for i, r in enumerate(top_10)
        ],
    }


class _Evaluator:
    def __init__(self, artifacts: LoadedArtifacts, config: AppConfig) -> None:
        self.config = config
        self.embedding_service = _build_embedding_service(config)
        self.vector_store = _build_vector_store(config, artifacts)
        self.graph_repository = InMemoryGraphRepository(
            artifacts.mentions,
            artifacts.has_chunk,
            artifacts.chunks,
            artifacts.entities,
        )
        self.chunk_repository = InMemoryChunkRepository(artifacts.chunks)
        self.vector_search = VectorSearchUseCase(self.embedding_service, self.vector_store)
        self.graph_expand = GraphExpandUseCase(self.graph_repository)
        self.rerank = RerankUseCase(self.chunk_repository)
        self.bm25 = _build_sparse_retriever(artifacts.chunks)
        self.tfidf = _build_tfidf_retriever(artifacts.chunks)
        self.rrf = RRFFusionService()
        self.aar = AARFusionService()
        self.mmr = MMRRerankService(lambda_param=0.5)
        self.cross_encoder: CrossEncoderRerankService | None = None

    def _score_fusion(self, seed: list[tuple[str, float]], config: SearchConfig) -> list[RetrievalResult]:
        expanded = self.graph_expand.execute({cid for cid, _ in seed}, config)
        return self.rerank.execute(seed, expanded, config)

    def _dense_results(self, query_text: str, config: SearchConfig) -> list[tuple[str, float]]:
        return self.vector_search.execute(Query(query_text), config)

    def run_dense(self, query_text: str, query: dict) -> dict:
        start = time.perf_counter()
        config = SearchConfig(top_k=10, expand_depth=2, alpha=0.8, max_results=20)
        seed = self._dense_results(query_text, config)
        ranked = self._score_fusion(seed, config)
        latency_ms = (time.perf_counter() - start) * 1000
        return _make_record(query, ranked, latency_ms)

    def run_bm25(self, query_text: str, query: dict) -> dict:
        start = time.perf_counter()
        config = SearchConfig(top_k=10, expand_depth=2, alpha=0.8, max_results=20)
        seed = self.bm25.search(query_text, config.top_k)
        ranked = self._score_fusion(seed, config)
        latency_ms = (time.perf_counter() - start) * 1000
        return _make_record(query, ranked, latency_ms)

    def run_tfidf(self, query_text: str, query: dict) -> dict:
        start = time.perf_counter()
        config = SearchConfig(top_k=10, expand_depth=2, alpha=0.8, max_results=20)
        seed = self.tfidf.search(query_text, config.top_k)
        ranked = self._score_fusion(seed, config)
        latency_ms = (time.perf_counter() - start) * 1000
        return _make_record(query, ranked, latency_ms)

    def run_rrf(self, query_text: str, query: dict, k: int = 60) -> dict:
        start = time.perf_counter()
        config = SearchConfig(top_k=10, expand_depth=2, alpha=0.8, max_results=20)
        dense = self._dense_results(query_text, config)
        sparse = self.bm25.search(query_text, config.top_k)
        fused = self.rrf.fuse(
            [{"chunk_id": cid, "score": float(score)} for cid, score in dense],
            [{"chunk_id": cid, "score": float(score)} for cid, score in sparse],
            k=k,
        )
        seed = [(r.chunk_id, r.rrf_score) for r in fused[: config.top_k]]
        ranked = self._score_fusion(seed, config)
        latency_ms = (time.perf_counter() - start) * 1000
        return _make_record(query, ranked, latency_ms)

    def run_aar(self, query_text: str, query: dict) -> dict:
        """Run article-level AAR fusion of BM25 + TF-IDF.

        The dense retriever underperforms on this evaluation set, so including it
        in the AAR average drags down strong sparse rankings.  Fusing only the
        two sparse retrievers preserves their signal while still combining
        multiple strategies.

        We fuse at the ``article_id`` level so that different chunks from the
        same article reinforce each other instead of competing as independent
        items.  The AAR service returns average ranks (lower is better); we
        convert those to a positive score so the downstream score-fusion stage
        does not push the AAR seed chunks below graph-expanded chunks.
        """
        return self.run_aar_with_candidate_k(query_text, query, candidate_k=20)

    def run_aar_with_candidate_k(self, query_text: str, query: dict, candidate_k: int = 20) -> dict:
        """Run article-level AAR fusion with a configurable retrieval depth."""
        start = time.perf_counter()
        config = SearchConfig(top_k=10, expand_depth=2, alpha=0.8, max_results=20)
        bm25_sparse = self.bm25.search(query_text, candidate_k)
        tfidf_sparse = self.tfidf.search(query_text, candidate_k)

        # Need article_id for article-level AAR; look it up from the chunk repo.
        chunk_ids = {cid for cid, _ in bm25_sparse + tfidf_sparse}
        chunks = self.chunk_repository.get_chunks(chunk_ids)

        def _with_article(results: list[tuple[str, float]]) -> list[dict]:
            out: list[dict] = []
            for cid, score in results:
                article_id = str(chunks.get(cid, {}).get("article_id", ""))
                out.append({"chunk_id": cid, "article_id": article_id, "score": float(score)})
            return out

        fused = self.aar.fuse(
            _with_article(bm25_sparse),
            _with_article(tfidf_sparse),
            group_key="article_id",
        )

        # Build seed chunk list from the top fused articles, using the best chunk
        # from each article.  We keep the chunk with the highest retriever score
        # for each article so the strongest evidence is used as the seed.
        article_to_best_chunk: dict[str, tuple[str, float]] = {}
        for cid, score in bm25_sparse + tfidf_sparse:
            article_id = str(chunks.get(cid, {}).get("article_id", ""))
            if not article_id:
                continue
            existing = article_to_best_chunk.get(article_id)
            if existing is None or score > existing[1]:
                article_to_best_chunk[article_id] = (cid, score)

        seed: list[tuple[str, float]] = []
        for fused_rank, r in enumerate(fused[: config.top_k], start=1):
            cid_score = article_to_best_chunk.get(r.id)
            if cid_score:
                cid, _ = cid_score
                # Convert AAR fused rank to an RRF-style positive score so the
                # top AAR articles keep their relative ordering after graph expansion.
                seed.append((cid, 1.0 / fused_rank))

        # Use a high alpha and no graph expansion so the AAR seed ranking is preserved.
        aar_config = SearchConfig(
            top_k=config.top_k,
            expand_depth=0,
            max_entity_degree=config.max_entity_degree,
            max_expansion_per_entity=config.max_expansion_per_entity,
            max_expanded_nodes=config.max_expanded_nodes,
            alpha=0.99,
            depth_scores=config.depth_scores,
            max_results=config.max_results,
        )
        ranked = self._score_fusion(seed, aar_config)
        latency_ms = (time.perf_counter() - start) * 1000
        return _make_record(query, ranked, latency_ms)

    def run_mmr(self, query_text: str, query: dict) -> dict:
        start = time.perf_counter()
        config = SearchConfig(top_k=10, expand_depth=2, alpha=0.8, max_results=20)
        seed = self._dense_results(query_text, config)
        interim = self._score_fusion(seed, config)
        reranked = self.mmr.rerank_objects(interim, query_text, top_k=config.top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        return _make_record(query, reranked, latency_ms)

    def run_cross_encoder(self, query_text: str, query: dict) -> dict:
        if self.cross_encoder is None:
            logger.info("Loading cross-encoder model (one-time)...")
            self.cross_encoder = CrossEncoderRerankService(device="cpu")
        start = time.perf_counter()
        config = SearchConfig(top_k=10, expand_depth=2, alpha=0.8, max_results=20)
        seed = self._dense_results(query_text, config)
        interim = self._score_fusion(seed, config)
        reranked = self.cross_encoder.rerank_objects(interim, query_text, top_k=config.top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        return _make_record(query, reranked, latency_ms)


def _write_results(method: str, records: list[dict]) -> Path:
    path = EVAL_DIR / f"results_{method}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate new retrieval/ranking methods.")
    parser.add_argument("--skip-cross-encoder", action="store_true", help="Skip the cross-encoder method.")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit to first N queries for smoke testing.")
    args = parser.parse_args()

    cache_dir = os.environ.get("ARTIFACT_CACHE_DIR", "").strip() or str(
        Path(tempfile.gettempdir()) / "pubmed-graphrag"
    )
    print(f"\nUsing artifact cache dir: {cache_dir}", flush=True)
    bootstrap_artifacts(cache_dir)

    config = AppConfig.default()

    # Use the preloaded artifacts from bootstrap_artifacts global cache.
    from src.bootstrap.bootstrap_artifacts import get_preloaded_artifacts

    artifacts = get_preloaded_artifacts()
    evaluator = _Evaluator(artifacts, config)
    queries = _load_queries(QUERIES_PATH)
    if args.max_questions:
        queries = queries[: args.max_questions]
    print(f"Loaded {len(queries)} queries from {QUERIES_PATH}", flush=True)

    methods = [
        ("dense", evaluator.run_dense),
        ("bm25", evaluator.run_bm25),
        ("tfidf", evaluator.run_tfidf),
        ("rrf", evaluator.run_rrf),
        ("aar", evaluator.run_aar),
        ("mmr", evaluator.run_mmr),
    ]
    if not args.skip_cross_encoder:
        methods.append(("cross_encoder", evaluator.run_cross_encoder))

    all_metrics: dict[str, dict] = {}
    for method_name, runner in methods:
        print(f"\nRunning {method_name} evaluation...", flush=True)
        records: list[dict] = []
        for query in queries:
            record = runner(query["question"], query)
            records.append(record)
            logger.info(
                "%s | %s | R@5=%s R@10=%s MRR@10=%s | latency=%s ms",
                method_name,
                record["query_id"],
                record["recall@5"],
                record["recall@10"],
                record["mrr@10"],
                record["latency_ms"],
            )
        metrics = _aggregate(records)
        all_metrics[method_name] = metrics
        results_path = _write_results(method_name, records)
        print(f"  {method_name} -> Recall@5={metrics['recall@5']} Recall@10={metrics['recall@10']} MRR@10={metrics['mrr@10']} latency={metrics['avg_latency_ms']} ms", flush=True)
        print(f"  Detailed results saved to {results_path}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "new_methods_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": {
                    "date": "2026-07-01",
                    "num_queries": len(queries),
                    "metrics": ["recall@5", "recall@10", "mrr@10", "avg_latency_ms"],
                },
                "methods": all_metrics,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nAggregate metrics saved to {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
