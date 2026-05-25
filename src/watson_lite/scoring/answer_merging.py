from __future__ import annotations

import logging

from watson_lite.core.models import AnswerCandidate
from watson_lite.scoring.type_coercion import resolve_span_to_qid

logger = logging.getLogger(__name__)


def merge_candidates_by_qid(
    candidates: list[AnswerCandidate],
) -> list[AnswerCandidate]:
    if len(candidates) <= 1:
        return candidates

    groups: dict[str, list[AnswerCandidate]] = {}
    ungrouped: list[AnswerCandidate] = []
    for c in candidates:
        qid = resolve_span_to_qid(c.span)
        if qid:
            groups.setdefault(qid, []).append(c)
        else:
            ungrouped.append(c)

    merged: list[AnswerCandidate] = list(ungrouped)
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
            continue
        canonical = min(group, key=lambda c: len(c.span))
        best_rank = min(c.rank for c in group)
        merged.append(
            AnswerCandidate(
                span=canonical.span,
                source=canonical.source,
                url=canonical.url,
                passage=canonical.passage,
                extraction_score=max(c.extraction_score for c in group),
                rank=best_rank,
                graph_corroborated=any(c.graph_corroborated for c in group),
                doc_frequency=sum(c.doc_frequency for c in group),
            )
        )

    merged.sort(key=lambda c: c.extraction_score, reverse=True)
    return merged
