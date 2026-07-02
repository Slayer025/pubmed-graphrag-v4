"""Unit tests for the cross-encoder reranker."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.domain.services.cross_encoder_rerank_service import (
    CrossEncoderCandidate,
    CrossEncoderRerankService,
)


def _make_candidates() -> list[CrossEncoderCandidate]:
    return [
        CrossEncoderCandidate(id="c1", text="Diabetes treatment with metformin", score=0.8),
        CrossEncoderCandidate(id="c2", text="BRCA1 breast cancer mutation", score=0.7),
        CrossEncoderCandidate(id="c3", text="Obesity lifestyle changes", score=0.6),
    ]


def test_rerank_orders_by_model_scores() -> None:
    with patch(
        "src.domain.services.cross_encoder_rerank_service.CrossEncoder"
    ) as mock_cls:
        mock_model = MagicMock()
        # c2 gets highest score, c1 lowest.
        mock_model.predict.return_value = [0.2, 0.9, 0.5]
        mock_cls.return_value = mock_model

        reranker = CrossEncoderRerankService("dummy-model")
        results = reranker.rerank(_make_candidates(), "cancer", top_k=2)

        assert len(results) == 2
        assert results[0].id == "c2"
        assert results[1].id == "c3"

        # Verify the model received (query, text) pairs in candidate order.
        pairs = mock_model.predict.call_args[0][0]
        assert pairs == [
            ("cancer", "Diabetes treatment with metformin"),
            ("cancer", "BRCA1 breast cancer mutation"),
            ("cancer", "Obesity lifestyle changes"),
        ]


def test_top_k_capped() -> None:
    with patch(
        "src.domain.services.cross_encoder_rerank_service.CrossEncoder"
    ) as mock_cls:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.3, 0.6, 0.9]
        mock_cls.return_value = mock_model

        reranker = CrossEncoderRerankService("dummy-model")
        results = reranker.rerank(_make_candidates(), "query", top_k=2)
        assert len(results) == 2


def test_empty_candidates() -> None:
    with patch(
        "src.domain.services.cross_encoder_rerank_service.CrossEncoder"
    ) as mock_cls:
        mock_cls.return_value = MagicMock()
        reranker = CrossEncoderRerankService("dummy-model")
        assert reranker.rerank([], "query", top_k=5) == []


def test_non_positive_top_k() -> None:
    with patch(
        "src.domain.services.cross_encoder_rerank_service.CrossEncoder"
    ) as mock_cls:
        mock_cls.return_value = MagicMock()
        reranker = CrossEncoderRerankService("dummy-model")
        assert reranker.rerank(_make_candidates(), "query", top_k=0) == []
        assert reranker.rerank(_make_candidates(), "query", top_k=-1) == []


def test_rerank_objects_with_dataclass() -> None:
    @dataclass
    class Item:
        chunk_id: str
        text: str
        combined_score: float

    items = [
        Item("c1", "Diabetes treatment with metformin", 0.8),
        Item("c2", "BRCA1 breast cancer mutation", 0.7),
    ]

    with patch(
        "src.domain.services.cross_encoder_rerank_service.CrossEncoder"
    ) as mock_cls:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.8]
        mock_cls.return_value = mock_model

        reranker = CrossEncoderRerankService("dummy-model")
        results = reranker.rerank_objects(items, "cancer", top_k=2)

        assert len(results) == 2
        assert results[0].chunk_id == "c2"
