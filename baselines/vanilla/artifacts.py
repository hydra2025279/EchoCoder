from __future__ import annotations

from typing import Any

from benchmarks.ClarifyCodeBench.clarifycodebench.metrics import calculate_ora, calculate_tkqr


def _target_key(row: dict[str, Any]) -> str | None:
    matched_key_id = row.get("matched_key_id")
    if matched_key_id is not None:
        return f"id:{matched_key_id}"
    matched_question = row.get("matched_key_question")
    return f"question:{matched_question}" if matched_question else None


def _metrics(
    rows: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    total_key_questions: int,
) -> dict[str, Any]:
    correct = sum(bool(row.get("correct")) for row in rows)
    covered = {
        target
        for row in rows
        if row.get("correct")
        for target in [_target_key(row)]
        if target is not None
    }
    nonempty_batches = [batch for batch in batches if batch["question_count"] > 0]
    valid_nonempty_batches = sum(bool(batch["batch_valid"]) for batch in nonempty_batches)
    precision = correct / len(rows) if rows else 0.0
    recall = len(covered) / total_key_questions if total_key_questions else 0.0
    ask_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "metric_unit": "question_and_nonempty_batch",
        "total_key_questions": total_key_questions,
        "total_predicted_questions": len(rows),
        "correct_questions": correct,
        "precision": precision,
        "recall": recall,
        "f1": ask_f1,
        "ask_f1": ask_f1,
        "recall_ques": recall,
        "precision_ques": precision,
        "precision_level": (
            valid_nonempty_batches / len(nonempty_batches)
            if nonempty_batches
            else 0.0
        ),
        "ask_f1_ques": ask_f1,
        "covered_key_questions": len(covered),
        "total_generated_questions": len(rows),
        "correct_generated_questions": correct,
        "total_question_calls": len(batches),
        "nonempty_batches": len(nonempty_batches),
        "valid_nonempty_batches": valid_nonempty_batches,
        "empty_batches": len(batches) - len(nonempty_batches),
    }


def attach_iteration_metrics(
    iterations: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    total_key_questions: int,
    qa_pairs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    total_questions = sum(len(iteration["questions"]) for iteration in iterations)
    if total_questions != len(evaluations):
        raise ValueError("question evaluations do not align with iteration questions")

    answers_by_target: dict[str, str] = {}
    for ground_truth_index, qa_pair in enumerate(qa_pairs or [], start=1):
        question = str(qa_pair.get("question", "")).strip()
        answer = str(qa_pair.get("answer", ""))
        target_id = (
            qa_pair.get("blocker_id")
            or qa_pair.get("question_id")
            or f"GT_{ground_truth_index}"
        )
        answers_by_target[f"id:{target_id}"] = answer
        if question:
            answers_by_target[f"question:{question}"] = answer

    all_rows: list[dict[str, Any]] = []
    all_batches: list[dict[str, Any]] = []
    offset = 0
    for iteration in iterations:
        iteration_id = str(iteration["iteration_id"])
        batch_id = iteration.get("batch_id") or iteration_id
        rows: list[dict[str, Any]] = []
        for question_index, question in enumerate(iteration["questions"], start=1):
            evaluation = evaluations[offset]
            offset += 1
            target = _target_key(evaluation)
            is_correct = bool(evaluation.get("correct"))
            rows.append(
                {
                    "question_id": f"{iteration_id}_Q{question_index}",
                    "global_question_id": f"{batch_id}_Q{question_index}",
                    "iteration": iteration["iteration"],
                    "iteration_id": iteration_id,
                    "batch_id": batch_id,
                    "question": question,
                    "oracle_status": "valid" if is_correct else "invalid",
                    "answer": answers_by_target.get(target) if is_correct else None,
                    "correct": is_correct,
                    "duplicate": bool(evaluation.get("duplicate")),
                    "matched_key_question": evaluation.get("matched_key_question"),
                    "matched_key_id": evaluation.get("matched_key_id"),
                }
            )
        batch = {
            "iteration": iteration["iteration"],
            "iteration_id": iteration_id,
            "batch_id": batch_id,
            "question_count": len(rows),
            "valid_question_count": sum(bool(row["correct"]) for row in rows),
        }
        batch["invalid_question_count"] = (
            batch["question_count"] - batch["valid_question_count"]
        )
        batch["batch_valid"] = batch["valid_question_count"] > 0
        all_rows.extend(rows)
        all_batches.append(batch)
        local = _metrics(rows, [batch], total_key_questions)
        so_far = _metrics(all_rows, all_batches, total_key_questions)
        iteration["question_records"] = rows
        iteration["question_batch"] = batch
        iteration["metrics"] = {
            **local,
            "precision_ques_so_far": so_far["precision_ques"],
            "recall_ques_so_far": so_far["recall_ques"],
            "ask_f1_ques_so_far": so_far["ask_f1_ques"],
            "precision_level_so_far": so_far["precision_level"],
            "total_generated_questions_so_far": so_far["total_generated_questions"],
            "correct_generated_questions_so_far": so_far[
                "correct_generated_questions"
            ],
            "covered_key_questions_so_far": so_far["covered_key_questions"],
        }
        iteration["status"] = "evaluated"

    metrics = _metrics(all_rows, all_batches, total_key_questions)
    metrics["evaluations"] = all_rows
    hit_sequence = [int(bool(row["correct"])) for row in all_rows]
    metrics["TKQR"] = calculate_tkqr(hit_sequence + [0], total_key_questions)
    metrics["ORA"] = calculate_ora(len(all_rows), total_key_questions)
    return metrics
