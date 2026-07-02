"""Maximal Marginal Relevance (MMR) re-ranking service.

MMR balances relevance against diversity by greedily selecting the candidate
that maximises::

    mmr_score = lambda * sim(query, candidate)
                - (1 - lambda) * max_{s in selected} sim(candidate, s)

The implementation uses a lightweight TF-IDF vectorizer from ``sklearn`` so it
has no heavy transformer dependencies and can be used as a drop-in second
stage reranker after dense, sparse, or fused retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class MMRCandidate:
    """Minimal wrapper for an item that can be re-ranked with MMR."""

    id: str
    text: str
    score: float
    payload: Any | None = None


T = TypeVar("T")


class MMRRerankService:
    """Re-rank a candidate list with Maximal Marginal Relevance."""

    def __init__(self, lambda_param: float = 0.5) -> None:
        """Initialise the MMR service.

        Args:
            lambda_param: Trade-off between relevance and diversity. ``0`` means
                pure diversity, ``1`` means pure relevance. Default ``0.5``.
        """
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError("lambda_param must be between 0.0 and 1.0")
        self.lambda_param = lambda_param

    @staticmethod
    def _vectorizer() -> TfidfVectorizer:
        """Return a TF-IDF vectorizer tuned for biomedical snippets."""
        return TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            token_pattern=r"(?u)\b[\w-]+\b",
            sublinear_tf=True,
            norm="l2",
            max_df=1.0,
            min_df=1,
        )

    def rerank(
        self,
        candidates: list[MMRCandidate],
        query_text: str,
        *,
        top_k: int = 10,
    ) -> list[MMRCandidate]:
        """Return ``top_k`` candidates re-ranked by MMR.

        Args:
            candidates: Ranked or unranked candidate items. Each must have a
                non-empty ``text`` field; the ``score`` field provides the
                relevance estimate used by MMR.
            query_text: The original query text.
            top_k: Number of items to return.

        Returns:
            Re-ranked list of up to ``top_k`` ``MMRCandidate`` objects.
        """
        if top_k <= 0:
            return []

        candidates = [c for c in candidates if c.text.strip()]
        if not candidates:
            return []

        if len(candidates) == 1:
            return candidates[:top_k]

        top_k = min(top_k, len(candidates))

        texts = [query_text] + [c.text for c in candidates]
        vectorizer = self._vectorizer()
        vectors = vectorizer.fit_transform(texts)

        query_vec = vectors[0]
        candidate_vecs = vectors[1:]

        # Pairwise similarities between candidates (used for the diversity penalty).
        pairwise = cosine_similarity(candidate_vecs)

        # Relevance combines the first-stage score with the query cosine similarity.
        # Both are normalised to [0, 1] over the candidate pool so they are on the
        # same scale and robust to different retriever score ranges.
        scores = np.array([c.score for c in candidates], dtype=float)
        min_score, max_score = scores.min(), scores.max()
        if max_score > min_score:
            norm_scores = (scores - min_score) / (max_score - min_score)
        else:
            norm_scores = np.ones_like(scores)

        query_sims = cosine_similarity(candidate_vecs, query_vec).ravel()
        min_sim, max_sim = query_sims.min(), query_sims.max()
        if max_sim > min_sim:
            norm_query_sims = (query_sims - min_sim) / (max_sim - min_sim)
        else:
            norm_query_sims = np.ones_like(query_sims)

        # Equal weighting for first-stage score and query similarity by default.
        relevance = 0.5 * norm_scores + 0.5 * norm_query_sims

        selected: list[int] = []
        remaining = set(range(len(candidates)))

        # First pick: highest relevance.
        first = int(np.argmax(relevance))
        selected.append(first)
        remaining.remove(first)

        while remaining and len(selected) < top_k:
            best_idx: int | None = None
            best_score = -float("inf")
            for idx in remaining:
                # Diversity penalty = max similarity to already selected items.
                max_sim = max(pairwise[idx, s] for s in selected)
                mmr_score = (
                    self.lambda_param * relevance[idx]
                    - (1.0 - self.lambda_param) * max_sim
                )
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is None:
                break

            selected.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[idx] for idx in selected]

    def rerank_objects(
        self,
        objects: list[T],
        query_text: str,
        *,
        text_attr: str = "text",
        score_attr: str = "combined_score",
        id_attr: str = "chunk_id",
        top_k: int = 10,
    ) -> list[T]:
        """Convenience wrapper that re-ranks arbitrary objects by attribute.

        Args:
            objects: Objects to re-rank. Must expose ``text_attr``,
                ``score_attr`` and ``id_attr``.
            query_text: Original query text.
            text_attr: Attribute holding the item text.
            score_attr: Attribute holding the relevance score.
            id_attr: Attribute holding the item identifier.
            top_k: Number of items to return.

        Returns:
            The re-ranked subset of ``objects``.
        """
        candidates = [
            MMRCandidate(
                id=str(getattr(obj, id_attr)),
                text=str(getattr(obj, text_attr)),
                score=float(getattr(obj, score_attr)),
                payload=obj,
            )
            for obj in objects
        ]
        reranked = self.rerank(candidates, query_text, top_k=top_k)
        return [c.payload for c in reranked if c.payload is not None]
