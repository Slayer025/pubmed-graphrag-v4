"""Cross-encoder second-stage reranking service.

Uses a lightweight ``sentence-transformers`` cross-encoder to score every
``(query, candidate)`` pair and re-rank the candidates by those scores. Cross
encoders are more accurate than dot-product similarity because they can attend
to both sentences jointly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TypeVar

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class CrossEncoderCandidate:
    """Minimal wrapper for an item that can be reranked by the cross-encoder."""

    id: str
    text: str
    score: float
    payload: Any | None = None


T = TypeVar("T")


class CrossEncoderRerankService:
    """Re-rank candidates with a lightweight cross-encoder model."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        *,
        device: str | None = None,
        max_seq_length: int | None = None,
    ) -> None:
        """Initialise the reranker and load the cross-encoder model.

        Args:
            model_name: Hugging Face model id for a cross-encoder.
            device: ``"cpu"``, ``"cuda"`` or ``None`` for auto.
            max_seq_length: Optional sequence length cap.
        """
        self.model_name = model_name
        self._model = CrossEncoder(model_name, device=device, max_length=max_seq_length)
        logger.info("Loaded cross-encoder model: %s", model_name)

    def rerank(
        self,
        candidates: list[CrossEncoderCandidate],
        query_text: str,
        *,
        top_k: int = 10,
        batch_size: int = 16,
    ) -> list[CrossEncoderCandidate]:
        """Return ``top_k`` candidates re-ranked by cross-encoder scores.

        Args:
            candidates: Candidate items. Each must have a non-empty ``text``
                field. The ``score`` field is preserved but not used by the
                cross-encoder; final ordering is determined by the model score.
            query_text: The query text.
            top_k: Number of items to return.
            batch_size: Batch size passed to the cross-encoder.

        Returns:
            Re-ranked list of up to ``top_k`` ``CrossEncoderCandidate`` objects.
        """
        if top_k <= 0:
            return []

        candidates = [c for c in candidates if c.text.strip()]
        if not candidates:
            return []

        pairs = [(query_text, c.text) for c in candidates]
        ce_scores = self._model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        scored = sorted(
            zip(candidates, ce_scores),
            key=lambda item: item[1],
            reverse=True,
        )
        return [candidate for candidate, _ in scored[:top_k]]

    def rerank_objects(
        self,
        objects: list[T],
        query_text: str,
        *,
        text_attr: str = "text",
        score_attr: str = "combined_score",
        id_attr: str = "chunk_id",
        top_k: int = 10,
        batch_size: int = 16,
    ) -> list[T]:
        """Convenience wrapper that reranks arbitrary objects by attribute.

        Args:
            objects: Objects to rerank. Must expose ``text_attr``,
                ``score_attr`` and ``id_attr``.
            query_text: Original query text.
            text_attr: Attribute holding the item text.
            score_attr: Attribute holding the relevance score (preserved only).
            id_attr: Attribute holding the item identifier.
            top_k: Number of items to return.
            batch_size: Batch size for the model.

        Returns:
            The re-ranked subset of ``objects``.
        """
        candidates = [
            CrossEncoderCandidate(
                id=str(getattr(obj, id_attr)),
                text=str(getattr(obj, text_attr)),
                score=float(getattr(obj, score_attr)),
                payload=obj,
            )
            for obj in objects
        ]
        reranked = self.rerank(
            candidates, query_text, top_k=top_k, batch_size=batch_size
        )
        return [c.payload for c in reranked if c.payload is not None]
