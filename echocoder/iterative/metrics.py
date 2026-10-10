from __future__ import annotations

from typing import Any


def _matched_target_id(attempt: dict[str, Any]) -> str | None:
    matched_key_id = attempt.get("matched_key_id")
    if matched_key_id is not None:
        return f"id:{matched_key_id}"
    matched_question = attempt.get("matched_key_question")
    if matched_question:
        return f"question:{matched_question}"
    return None


def _question_and_level_metrics(
    question_attempts: list[dict[str, Any]],
    batch_attempts: list[dict[str, Any]],
    total_key_questions: int,
) -> dict[str, Any]:
    correct_questions = sum(item["correct"] for item in question_attempts)
    total_questions = len(question_attempts)
    covered_targets = {
        target_id
        for item in question_attempts
        if item["correct"]
        for target_id in [_matched_target_id(item)]
        if target_id is not None
    }
    nonempty_batches = [
        item for item in batch_attempts if int(item.get("question_count", 0)) > 0
    ]
    valid_nonempty_batches = sum(item["correct"] for item in nonempty_batches)
    recall_ques = (
        len(covered_targets) / total_key_questions if total_key_questions else 0.0
    )
    precision_ques = (
        correct_questions / total_questions if total_questions else 0.0
    )
    ask_f1_ques = (
        2 * precision_ques * recall_ques / (precision_ques + recall_ques)
        if precision_ques + recall_ques
        else 0.0
    )
    return {
        "recall_ques": recall_ques,
        "precision_ques": precision_ques,
        "precision_level": (
            valid_nonempty_batches / len(nonempty_batches)
            if nonempty_batches
            else 0.0
        ),
        "ask_f1_ques": ask_f1_ques,
        "covered_key_questions": len(covered_targets),
        "total_generated_questions": total_questions,
        "correct_generated_questions": correct_questions,
        "nonempty_batches": len(nonempty_batches),
        "valid_nonempty_batches": valid_nonempty_batches,
    }


def question_metrics_from_attempts(
    attempts: list[dict[str, Any]], total_key_questions: int
) -> dict[str, Any]:
    correct = sum(attempt.get("oracle_status") == "valid" for attempt in attempts)
    predicted = len(attempts)
    precision = correct / predicted if predicted else 0.0
    recall = correct / total_key_questions if total_key_questions else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total_key_questions": total_key_questions,
        "total_predicted_questions": predicted,
        "correct_questions": correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ask_f1": f1,
        "evaluations": [
            {
                "question": attempt["question"],
                "correct": attempt.get("oracle_status") == "valid",
                "duplicate": False,
                "matched_key_question": attempt.get("matched_key_question"),
                "matched_key_id": None,
                "attempt": attempt["attempt"],
            }
            for attempt in attempts
        ],
    }


def question_metrics_from_rounds(
    rounds: list[dict[str, Any]], total_key_questions: int
) -> dict[str, Any]:
    batch_attempts = []
    question_attempts = []
    question_call_number = 0
    for round_data in rounds:
        for reconstruction in round_data.get("reconstructions", []):
            for batch in reconstruction.get("question_batches", []):
                question_call_number += 1
                batch_questions = []
                for question in batch.get("questions", []):
                    evaluation = question.get("oracle_evaluation") or {}
                    question_attempt = (
                        {
                            "question": question.get("question"),
                            "correct": question.get("oracle_status") == "valid",
                            "duplicate": bool(evaluation.get("duplicate")),
                            "matched_key_question": question.get(
                                "matched_key_question"
                            ),
                            "matched_key_id": evaluation.get("matched_key_id"),
                            "specification_round": round_data.get(
                                "specification_round"
                            ),
                            "reconstruction_index": reconstruction.get(
                                "reconstruction_index"
                            ),
                            "question_call_index": batch.get(
                                "question_call_index"
                            ),
                            "question_call_number": question_call_number,
                        }
                    )
                    question_attempts.append(question_attempt)
                    batch_questions.append(question_attempt)

                batch_correct = any(item["correct"] for item in batch_questions)
                batch_attempts.append(
                    {
                        "batch_id": (
                            f"Q{reconstruction.get('reconstruction_index')}_"
                            f"{batch.get('question_call_index')}"
                        ),
                        "specification_round": round_data.get(
                            "specification_round"
                        ),
                        "reconstruction_index": reconstruction.get(
                            "reconstruction_index"
                        ),
                        "question_call_index": batch.get("question_call_index"),
                        "question_call_number": question_call_number,
                        "correct": batch_correct,
                        "batch_valid": batch_correct,
                        "question_count": len(batch_questions),
                        "valid_question_count": sum(
                            item["correct"] for item in batch_questions
                        ),
                    }
                )

    metrics = {
        "metric_unit": "question_and_nonempty_batch",
        "total_key_questions": total_key_questions,
        "total_question_calls": len(batch_attempts),
        "empty_batches": sum(
            int(item.get("question_count", 0)) == 0 for item in batch_attempts
        ),
        "evaluations": batch_attempts,
    }
    metrics.update(
        _question_and_level_metrics(
            question_attempts, batch_attempts, total_key_questions
        )
    )
    return metrics


def _correctness_category(record: dict[str, Any]) -> str | None:
    initial = record.get("initial_code_passed")
    final = record.get("final_code_passed")
    if not isinstance(initial, bool) or not isinstance(final, bool):
        return None
    return f"C0 {'correct' if initial else 'wrong'} -> C1 {'correct' if final else 'wrong'}"


def summarize_iterative_records(
    records: list[dict[str, Any]], *, max_attempts: int | None = None
) -> dict[str, Any]:
    if max_attempts is None:
        max_attempts = max(
            (int(record.get("total_question_calls", 0)) for record in records),
            default=0,
        )
    count = len(records)
    first_valid_calls = [
        int(record["clarification_call_number"])
        for record in records
        if record.get("valid_clarification_found")
        and record.get("clarification_call_number") is not None
    ]

    def success_at(k: int) -> float:
        if not count:
            return 0.0
        return sum(call <= k for call in first_valid_calls) / count

    correctness_records = [
        record
        for record in records
        if isinstance(record.get("initial_code_passed"), bool)
        and isinstance(record.get("final_code_passed"), bool)
    ]
    initial_correct = sum(record["initial_code_passed"] for record in correctness_records)
    final_correct = sum(record["final_code_passed"] for record in correctness_records)
    categories = {
        "C0 wrong -> C1 correct": 0,
        "C0 correct -> C1 correct": 0,
        "C0 wrong -> C1 wrong": 0,
        "C0 correct -> C1 wrong": 0,
    }
    for record in correctness_records:
        category = _correctness_category(record)
        if category is not None:
            categories[category] += 1

    total_attempts = sum(len(record.get("clarification_attempts", [])) for record in records)
    valid_count = len(first_valid_calls)
    stop_reasons: dict[str, int] = {}
    total_refinements = 0
    for record in records:
        reason = str(record.get("stop_reason", "unknown"))
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
        total_refinements += int(record.get("valid_clarification_count", 0))
    return {
        "sample_count": count,
        "max_question_calls": max_attempts,
        "valid_question": {
            "ValidQuestion@1": success_at(1),
            f"ValidQuestion@{max_attempts}": success_at(max_attempts),
            "average_calls_until_first_valid": (
                sum(first_valid_calls) / valid_count if valid_count else None
            ),
            "no_valid_clarification_fraction": (
                (count - valid_count) / count if count else 0.0
            ),
            "successful_tasks": valid_count,
            "total_questions": total_attempts,
            "total_question_calls": sum(
                int(record.get("total_question_calls", 0)) for record in records
            ),
        },
        "success_at": {f"Success@{k}": success_at(k) for k in (1, 3, 5)},
        "refinement": {
            "total_successful_refinements": total_refinements,
            "average_successful_refinements": total_refinements / count if count else 0.0,
            "stop_reasons": stop_reasons,
        },
        "code_correctness": {
            "evaluated_tasks": len(correctness_records),
            "initial_code_correctness": (
                initial_correct / len(correctness_records)
                if correctness_records
                else None
            ),
            "final_code_correctness": (
                final_correct / len(correctness_records)
                if correctness_records
                else None
            ),
            "delta_after_clarification": (
                (final_correct - initial_correct) / len(correctness_records)
                if correctness_records
                else None
            ),
            "transitions": categories,
        },
    }
