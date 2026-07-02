"""Lightweight TF-IDF sparse retriever backed by in-memory chunks.

This adapter uses ``sklearn.feature_extraction.text.TfidfVectorizer`` and mirrors
:py:class:`BM25Retriever` so it can be dropped into the same hybrid fusion
pipelines. It fits on the same chunk text as BM25 and returns ranked
``(chunk_id, tfidf_score)`` tuples.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfRetriever:
    """TF-IDF sparse retriever initialized from a list of chunk records."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        text_field: str = "text",
        chunk_id_field: str = "chunk_id",
    ) -> None:
        """Build a TF-IDF index from the provided chunks.

        Args:
            chunks: Chunk records (e.g. from ``InMemoryChunkRepository``). Each
                record must contain the configured ``text_field`` and
                ``chunk_id_field``.
            text_field: Key holding the chunk text to index.
            chunk_id_field: Key holding the chunk identifier.
        """
        self._text_field = text_field
        self._chunk_id_field = chunk_id_field

        self._chunks: list[dict[str, Any]] = []
        self._chunk_ids: list[str] = []
        corpus: list[str] = []

        for chunk in chunks:
            chunk_id = str(chunk.get(chunk_id_field, ""))
            text = str(chunk.get(text_field, ""))
            if not chunk_id or not text:
                continue
            self._chunks.append(chunk)
            self._chunk_ids.append(chunk_id)
            corpus.append(text)

        if not corpus:
            # Store an empty index safely; vectorizer cannot fit on empty data.
            self._vectorizer: TfidfVectorizer | None = None
            self._vectors: np.ndarray | None = None
            return

        # Sublinear tf (1 + log(tf)), l2-normalised rows, English stop words.
        # ``token_pattern`` keeps words/digits and hyphenated biomedical tokens
        # such as ``BRCA-1`` or ``non-small-cell``.
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            token_pattern=r"(?u)\b[\w-]+\b",
            sublinear_tf=True,
            norm="l2",
            max_df=1.0,
            min_df=1,
        )
        self._vectors = self._vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return the top-k matching ``(chunk_id, tfidf_score)`` pairs.

        Args:
            query: Raw query text.
            top_k: Maximum number of results to return. Values <= 0 return an
                empty list.

        Returns:
            Ranked list of ``(chunk_id, score)`` tuples, highest cosine
            similarity first.
        """
        if top_k <= 0 or not self._vectorizer or self._vectors is None or not query:
            return []

        query_vec = self._vectorizer.transform([query])
        if query_vec.nnz == 0:
            return []

        # Cosine similarity between l2-normalised vectors = dot product.
        scores = (self._vectors @ query_vec.T).toarray().ravel()
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )[:top_k]

        return [
            (self._chunk_ids[idx], float(scores[idx]))
            for idx in ranked_indices
        ]
