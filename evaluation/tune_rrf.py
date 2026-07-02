#!/usr/bin/env python3
"""Quick RRF k sweep."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap.environment import configure_environment

os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "hf_cache"))
configure_environment()

from evaluation.run_new_methods_eval import _Evaluator, _aggregate, _load_queries, QUERIES_PATH
from src.bootstrap.bootstrap_artifacts import bootstrap_artifacts
from src.config import AppConfig

cache_dir = os.environ.get("ARTIFACT_CACHE_DIR", "").strip() or str(
    Path(tempfile.gettempdir()) / "pubmed-graphrag"
)
print(f"Using cache: {cache_dir}", flush=True)
bootstrap_artifacts(cache_dir)
config = AppConfig.default()
from src.bootstrap.bootstrap_artifacts import get_preloaded_artifacts

artifacts = get_preloaded_artifacts()
evaluator = _Evaluator(artifacts, config)
queries = _load_queries(QUERIES_PATH)
print(f"Loaded {len(queries)} queries", flush=True)

best = None
for k in (10, 20, 30, 40, 50, 60, 80, 100):
    records = []
    start = time.perf_counter()
    for query in queries:
        records.append(evaluator.run_rrf(query["question"], query, k=k))
    metrics = _aggregate(records)
    print(
        f"k={k:3d} -> R@5={metrics['recall@5']:.3f} R@10={metrics['recall@10']:.3f} "
        f"MRR@10={metrics['mrr@10']:.4f} latency={metrics['avg_latency_ms']:.1f} ms "
        f"wall={time.perf_counter()-start:.1f}s",
        flush=True,
    )
    if best is None or metrics["recall@10"] > best[0]:
        best = (metrics["recall@10"], k, metrics)
print(f"best by R@10: k={best[1]} -> {best[2]}", flush=True)
