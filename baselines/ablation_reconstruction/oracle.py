from __future__ import annotations

from typing import Any

from baselines.common import evaluate_questions


def query_oracle_batch(
    client: Any,
    *,
    questions: list[str],
    qa_pairs: list[dict[str, Any]],
    judge_model: str,
    consistency: int,
    max_tokens: int,
    max_retries: int,
) -> list[dict[str, Any]]:
    if not questions:
        return []
    metrics = evaluate_questions(
        client,
        judge_model=judge_model,
        generated_questions=questions,
        qa_pairs=qa_pairs,
        consistency=consistency,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )
    answers = {
        str(pair.get("question", "")).strip(): str(pair.get("answer", "")).strip()
        for pair in qa_pairs
    }
    results: list[dict[str, Any]] = []
    for evaluation in metrics["evaluations"]:
        matched = evaluation.get("matched_key_question")
        answer = answers.get(str(matched), "") if matched else ""
        valid = bool(evaluation.get("correct") and answer)
        results.append(
            {
                "status": "valid" if valid else "invalid",
                "answer": answer if valid else None,
                "matched_key_question": matched,
                "evaluation": evaluation,
            }
        )
    return results

