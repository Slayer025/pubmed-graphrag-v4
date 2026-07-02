"""Pure-domain Average Average Rank (AAR) fusion service.

AAR is a parameter-free rank aggregation method.  For each item it computes the
average of the item's ranks across the retrieval systems in which it appears.
Missing lists are **not** penalised; this prevents a single strong retriever
from being drowned out by missing-rank penalties from other lists.

The service can fuse at the chunk level (default) or at a coarser grain such as
``article_id``.  Coarser-grained fusion is useful when the evaluation metric is
defined at the article level: it prevents the same article from being split
across multiple chunks and missing the top ranks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AARResult:
    """One fused result produced by the AAR service."""

    id: str
    aar_score: float
    metadata: dict[str, Any]


class AARFusionService:
    """Fuse multiple ranked result lists using Average Average Rank."""

    def fuse(
        self,
        *result_lists: list[dict[str, Any]],
        group_key: str = "chunk_id",
    ) -> list[AARResult]:
        """Merge ranked lists and score by average rank.

        Args:
            result_lists: One or more ranked lists. Each inner list is assumed
                to be ordered from best (rank 1) to worst. Each item must be a
                mapping containing at least ``chunk_id`` and ``score`` keys.
                The ``score`` value is preserved in ``metadata`` but is not used
                for ranking.
            group_key: Key used to aggregate items.  Defaults to ``chunk_id``;
                use ``article_id`` to fuse at the article level.

        Returns:
            List of ``AARResult`` objects sorted by ascending ``aar_score``.
            The ``aar_score`` field stores the average rank; lower is better.
            Items that do not appear in any list are omitted.
        """
        if not result_lists:
            return []

        # Collect best rank per group per list, plus metadata.
        # ranks_by_group[list_index][group_id] = best_rank_in_this_list
        ranks_by_group: dict[str, list[int]] = {}
        metadata: dict[str, dict[str, Any]] = {}

        for ranked_list in result_lists:
            if not ranked_list:
                continue

            best_rank_in_list: dict[str, int] = {}
            for rank, item in enumerate(ranked_list, start=1):
                group_id = str(item.get(group_key, ""))
                if not group_id:
                    continue
                # Keep the best (lowest) rank for this group in this list.
                if group_id not in best_rank_in_list or rank < best_rank_in_list[group_id]:
                    best_rank_in_list[group_id] = rank
                if group_id not in metadata:
                    metadata[group_id] = dict(item)

            for group_id, rank in best_rank_in_list.items():
                ranks_by_group.setdefault(group_id, []).append(rank)

        if not ranks_by_group:
            return []

        fused = [
            AARResult(
                id=group_id,
                # Average rank over the lists in which the group actually appears.
                aar_score=round(sum(ranks) / len(ranks), 4),
                metadata=metadata[group_id],
            )
            for group_id, ranks in ranks_by_group.items()
        ]
        # Lower average rank is better; tie-break deterministically by id.
        return sorted(fused, key=lambda r: (r.aar_score, r.id))
