from __future__ import annotations

from typing import Any


def validated_clarifications(
    question_metrics: dict[str, Any], qa_pairs: list[dict[str, Any]]
) -> list[dict[str, str]]:
    by_question = {str(pair.get("question", "")).strip(): pair for pair in qa_pairs}
    by_blocker = {
        str(pair["blocker_id"]): pair
        for pair in qa_pairs
        if pair.get("blocker_id") is not None
    }
    accepted = []
    for evaluation in question_metrics.get("evaluations", []):
        if not evaluation.get("correct"):
            continue
        pair = by_blocker.get(str(evaluation.get("matched_key_id")))
        if pair is None:
            pair = by_question.get(str(evaluation.get("matched_key_question", "")).strip())
        if pair is None:
            continue
        accepted.append(
            {
                "question": str(evaluation["question"]),
                "answer": str(pair.get("answer", "")),
            }
        )
    return accepted
