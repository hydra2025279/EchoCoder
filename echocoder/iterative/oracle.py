from __future__ import annotations

from typing import Any

from openai import OpenAI

from baselines.common import AzureResponsesClient, evaluate_questions


def query_clarification_oracle_batch(
    client: OpenAI | AzureResponsesClient,
    *,
    questions: list[str],
    qa_pairs: list[dict[str, Any]],
    judge_model: str,
    consistency: int,
    max_tokens: int | None,
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
        is_valid = bool(evaluation.get("correct") and answer)
        results.append(
            {
                "status": "valid" if is_valid else "invalid",
                "answer": answer if is_valid else None,
                "matched_key_question": matched,
                "evaluation": evaluation,
            }
        )
    return results


def query_clarification_oracle(
    client: OpenAI | AzureResponsesClient,
    *,
    question: str,
    qa_pairs: list[dict[str, Any]],
    judge_model: str,
    consistency: int,
    max_tokens: int | None,
    max_retries: int,
) -> dict[str, Any]:
    """Compatibility wrapper for callers that submit a single question."""
    return query_clarification_oracle_batch(
        client,
        questions=[question],
        qa_pairs=qa_pairs,
        judge_model=judge_model,
        consistency=consistency,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )[0]
