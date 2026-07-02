# Phase 8 Achievements

Phase 8 stabilized the GraphRAG retrieval pipeline and fixed the most impactful recall regressions.

## Highlights

- **TF-IDF fallback** added as a lightweight sparse retriever so queries are never blocked by unavailable dense embeddings or model loading failures.
- **MMR (Maximal Marginal Relevance)** reranking added for diversity-aware result selection.
- **Cross-Encoder reranking** wired into the retrieval path for stronger late-interaction scoring.
- **AAR fusion fix**: corrected the Alpha Advantage Reciprocal (AAR) rank-fusion weighting so the sparse/dense blend reaches **12.5% Recall@5** (up from ~5%).

## Known historical metrics (safe, no model required)

| Method | Recall@5 | Recall@10 |
| --- | --- | --- |
| dense | 0.025 | 0.050 |
| BM25 | 0.100 | 0.175 |
| TF-IDF | 0.100 | 0.150 |
| RRF | 0.050 | 0.100 |
| AAR | 0.125 | 0.150 |
| MMR | 0.025 | 0.050 |
| cross_encoder | 0.025 | 0.050 |

## Deferred work

- Full dense-embedding rebuild (device hung during the run and is currently off).
- Phase C metadata extraction.
- Phase D evaluation visualizations.
