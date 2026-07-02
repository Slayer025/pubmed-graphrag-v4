"""Unit tests for Average Average Rank (AAR) fusion."""

from __future__ import annotations

import pytest

from src.domain.services.aar_fusion_service import AARFusionService


def _make_list_a() -> list[dict]:
    return [
        {"chunk_id": "c1", "article_id": "a1", "score": 0.9},
        {"chunk_id": "c2", "article_id": "a2", "score": 0.8},
        {"chunk_id": "c3", "article_id": "a3", "score": 0.7},
    ]


def _make_list_b() -> list[dict]:
    return [
        {"chunk_id": "c4", "article_id": "a2", "score": 0.95},
        {"chunk_id": "c5", "article_id": "a1", "score": 0.85},
        {"chunk_id": "c6", "article_id": "a4", "score": 0.75},
    ]


def test_fusion_prefers_consistent_low_ranks() -> None:
    service = AARFusionService()
    fused = service.fuse(_make_list_a(), _make_list_b())

    # list_a: c1(1), c2(2), c3(3)
    # list_b: c4(1), c5(2), c6(3)
    # AAR = average over lists where the chunk appears; no missing penalty.
    # Ties on score are broken by id, so rank-1 items c1 and c4 come first.
    ids = [r.id for r in fused]
    assert ids == ["c1", "c4", "c2", "c5", "c3", "c6"]


def test_no_missing_rank_penalty() -> None:
    """A chunk present in only one list should not be penalised by others."""
    service = AARFusionService()
    fused = service.fuse(_make_list_a(), _make_list_b())
    by_id = {r.id: r for r in fused}
    assert by_id["c3"].aar_score == 3.0
    assert by_id["c6"].aar_score == 3.0


def test_article_level_fusion_uses_best_chunk_rank() -> None:
    """Article-level fusion collapses multiple chunks from the same article."""
    service = AARFusionService()
    fused = service.fuse(_make_list_a(), _make_list_b(), group_key="article_id")

    # article a1: chunk c1 at rank 1 in list_a, chunk c5 at rank 2 in list_b.
    #            best ranks per list are 1 and 2 = AAR 1.5.
    # article a2: chunk c2 at rank 2 in list_a, chunk c4 at rank 1 in list_b.
    #            best ranks = 2 and 1 = AAR 1.5.
    # article a3: only list_a rank 3 = AAR 3.0.
    # article a4: only list_b rank 3 = AAR 3.0.
    by_id = {r.id: r for r in fused}
    assert by_id["a1"].aar_score == 1.5
    assert by_id["a2"].aar_score == 1.5
    assert by_id["a3"].aar_score == 3.0
    assert by_id["a4"].aar_score == 3.0


def test_article_level_ordering() -> None:
    fused = AARFusionService().fuse(_make_list_a(), _make_list_b(), group_key="article_id")
    ids = [r.id for r in fused]
    assert ids == ["a1", "a2", "a3", "a4"]


def test_single_list_preserves_order() -> None:
    service = AARFusionService()
    fused = service.fuse(_make_list_a())

    assert [r.id for r in fused] == ["c1", "c2", "c3"]
    assert fused[0].aar_score == 1.0


def test_empty_input_returns_empty() -> None:
    service = AARFusionService()
    assert service.fuse() == []


def test_all_empty_lists_return_empty() -> None:
    service = AARFusionService()
    assert service.fuse([], [], []) == []


def test_empty_list_is_skipped() -> None:
    service = AARFusionService()
    fused = service.fuse(_make_list_a(), [])
    # Empty list contributes no ranks; result is the same as a single list.
    assert [r.id for r in fused] == ["c1", "c2", "c3"]


def test_missing_chunk_id_ignored() -> None:
    service = AARFusionService()
    fused = service.fuse(
        [{"chunk_id": "c1", "article_id": "a1", "score": 0.9}, {"score": 0.8}],
        [{"chunk_id": "c2", "article_id": "a2", "score": 0.95}],
    )

    ids = [r.id for r in fused]
    assert "c1" in ids
    assert "c2" in ids
