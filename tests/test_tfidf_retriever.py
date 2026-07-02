"""Unit tests for the TF-IDF sparse retriever."""

from __future__ import annotations

import pytest

from src.infrastructure.retrievers.tfidf_retriever import TfidfRetriever


def _make_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "c1",
            "article_id": "a1",
            "text": "Metformin reduces cardiovascular risk in type 2 diabetes.",
        },
        {
            "chunk_id": "c2",
            "article_id": "a2",
            "text": "Insulin resistance is a hallmark of type 2 diabetes.",
        },
        {
            "chunk_id": "c3",
            "article_id": "a3",
            "text": "Obesity treatment often involves lifestyle changes.",
        },
        {
            "chunk_id": "c4",
            "article_id": "a4",
            "text": "BRCA1 mutations are linked to breast cancer risk.",
        },
    ]


def test_search_returns_top_k_limit() -> None:
    retriever = TfidfRetriever(_make_chunks())
    results = retriever.search("diabetes", top_k=2)

    assert len(results) == 2
    chunk_ids = [chunk_id for chunk_id, _ in results]
    assert all(isinstance(cid, str) for cid in chunk_ids)


def test_search_exact_term_boosts_relevant_chunk() -> None:
    retriever = TfidfRetriever(_make_chunks())
    results = retriever.search("metformin", top_k=3)

    assert len(results) == 3
    assert results[0][0] == "c1"
    assert results[0][1] > 0
    assert all(score >= 0 for _, score in results)


def _make_hyphenated_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "h1",
            "article_id": "a1",
            "text": "BRCA-1 mutations increase breast cancer risk.",
        },
        {
            "chunk_id": "h2",
            "article_id": "a2",
            "text": "IL-6 signaling is elevated in chronic inflammation.",
        },
        {
            "chunk_id": "h3",
            "article_id": "a3",
            "text": "T-cell activation requires co-stimulation.",
        },
    ]


def test_search_biomedical_symbol() -> None:
    retriever = TfidfRetriever(_make_chunks())
    results = retriever.search("BRCA1", top_k=2)

    assert len(results) > 0
    assert results[0][0] == "c4"


def test_tokenizer_keeps_hyphenated_terms() -> None:
    retriever = TfidfRetriever(_make_hyphenated_chunks())
    assert retriever.search("BRCA-1", top_k=1)[0][0] == "h1"
    assert retriever.search("IL-6", top_k=1)[0][0] == "h2"
    assert retriever.search("T-cell", top_k=1)[0][0] == "h3"


def test_empty_query_returns_empty_list() -> None:
    retriever = TfidfRetriever(_make_chunks())
    assert retriever.search("", top_k=5) == []


def test_non_positive_top_k_returns_empty_list() -> None:
    retriever = TfidfRetriever(_make_chunks())
    assert retriever.search("diabetes", top_k=0) == []
    assert retriever.search("diabetes", top_k=-1) == []


def test_top_k_larger_than_corpus_is_capped() -> None:
    chunks = _make_chunks()
    retriever = TfidfRetriever(chunks)
    results = retriever.search("diabetes", top_k=100)

    assert len(results) == len(chunks)


def test_empty_corpus_is_safe() -> None:
    retriever = TfidfRetriever([])
    assert retriever.search("diabetes", top_k=5) == []


def test_chunks_missing_required_fields_are_skipped() -> None:
    retriever = TfidfRetriever(
        [
            {"chunk_id": "c1", "text": "valid chunk"},
            {"text": "missing chunk_id"},
            {"chunk_id": "c3"},
        ]
    )
    results = retriever.search("valid", top_k=5)

    assert len(results) == 1
    assert results[0][0] == "c1"


def test_scores_are_sorted_descending() -> None:
    retriever = TfidfRetriever(_make_chunks())
    results = retriever.search("diabetes metformin", top_k=4)

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_tfidf_and_bm25_both_rank_diabetes_chunks_highest() -> None:
    """Both sparse retrievers should rank diabetes-related chunks above others."""
    from src.infrastructure.retrievers.bm25_retriever import BM25Retriever

    chunks = _make_chunks()
    tfidf = TfidfRetriever(chunks)
    bm25 = BM25Retriever(chunks)

    tfidf_results = tfidf.search("diabetes", top_k=2)
    bm25_results = bm25.search("diabetes", top_k=2)

    tfidf_top = {cid for cid, _ in tfidf_results}
    bm25_top = {cid for cid, _ in bm25_results}
    diabetes_chunks = {"c1", "c2"}

    assert tfidf_top == diabetes_chunks
    assert bm25_top == diabetes_chunks
