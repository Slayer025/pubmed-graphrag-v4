"""Unit tests for the Maximal Marginal Relevance reranker."""

from __future__ import annotations

import pytest

from src.domain.services.mmr_rerank_service import MMRCandidate, MMRRerankService


def _make_candidates() -> list[MMRCandidate]:
    return [
        MMRCandidate(id="c1", text="Metformin treats type 2 diabetes", score=0.9),
        MMRCandidate(id="c2", text="Insulin resistance in diabetes", score=0.7),
        MMRCandidate(id="c3", text="BRCA1 breast cancer mutation", score=0.6),
        MMRCandidate(id="c4", text="Obesity lifestyle intervention", score=0.5),
    ]


def test_relevance_prefers_high_score_and_matching_text() -> None:
    mmr = MMRRerankService(lambda_param=1.0)
    candidates = _make_candidates()
    results = mmr.rerank(candidates, "diabetes", top_k=3)

    assert len(results) == 3
    # c1 has the highest score and the most query-relevant text.
    assert results[0].id == "c1"


def test_diversity_spreads_topics() -> None:
    mmr = MMRRerankService(lambda_param=0.2)
    candidates = _make_candidates()
    results = mmr.rerank(candidates, "diabetes", top_k=3)

    ids = [r.id for r in results]
    # With strong diversity weighting the non-diabetes topic should appear.
    assert "c3" in ids or "c4" in ids


def test_top_k_capped() -> None:
    mmr = MMRRerankService(lambda_param=0.5)
    results = mmr.rerank(_make_candidates(), "diabetes", top_k=2)
    assert len(results) == 2


def test_empty_candidates() -> None:
    mmr = MMRRerankService()
    assert mmr.rerank([], "diabetes", top_k=5) == []


def test_non_positive_top_k() -> None:
    mmr = MMRRerankService()
    assert mmr.rerank(_make_candidates(), "diabetes", top_k=0) == []
    assert mmr.rerank(_make_candidates(), "diabetes", top_k=-1) == []


def test_invalid_lambda() -> None:
    with pytest.raises(ValueError):
        MMRRerankService(lambda_param=1.5)
    with pytest.raises(ValueError):
        MMRRerankService(lambda_param=-0.1)


def test_rerank_objects_with_dataclass() -> None:
    from dataclasses import dataclass

    @dataclass
    class Item:
        chunk_id: str
        text: str
        combined_score: float

    items = [
        Item("c1", "diabetes mellitus type 2 treatment with metformin", 0.99),
        Item("c2", "insulin resistance mechanism unrelated", 0.5),
        Item("c3", "BRCA1 breast cancer mutation study", 0.4),
    ]

    mmr = MMRRerankService(lambda_param=1.0)
    results = mmr.rerank_objects(items, "diabetes", top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "c1"  # highest score + most relevant text
