from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from baselines.common import summarize_passrate_records, summarize_question_records, write_json
from benchmarks.ClarifyCodeBench.clarifycodebench.metrics import calculate_ora, calculate_tkqr


def _target_id(question: dict[str, Any]) -> str | None:
    evaluation = question.get("oracle_evaluation") or {}
    matched_id = evaluation.get("matched_key_id")
    if matched_id is not None:
        return str(matched_id)
    matched = question.get("matched_key_question")
    return str(matched) if matched else None


def _metrics(
    questions: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    total_key_questions: int,
) -> dict[str, Any]:
    correct = sum(item.get("oracle_status") == "valid" for item in questions)
    covered = {
        target
        for item in questions
        if item.get("oracle_status") == "valid"
        for target in [_target_id(item)]
        if target is not None
    }
    nonempty = [batch for batch in batches if batch.get("question_count", 0) > 0]
    valid_batches = sum(bool(batch.get("batch_valid")) for batch in nonempty)
    precision = correct / len(questions) if questions else 0.0
    recall = len(covered) / total_key_questions if total_key_questions else 0.0
    ask_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "metric_unit": "question_and_nonempty_batch",
        "total_key_questions": total_key_questions,
        "recall_ques": recall,
        "precision_ques": precision,
        "ask_f1_ques": ask_f1,
        "precision_level": valid_batches / len(nonempty) if nonempty else 0.0,
        "covered_key_questions": len(covered),
        "total_generated_questions": len(questions),
        "correct_generated_questions": correct,
        "total_question_calls": len(batches),
        "nonempty_batches": len(nonempty),
        "valid_nonempty_batches": valid_batches,
        "empty_batches": len(batches) - len(nonempty),
    }


def attach_metrics(
    iterations: list[dict[str, Any]], total_key_questions: int
) -> dict[str, Any]:
    all_questions: list[dict[str, Any]] = []
    all_batches: list[dict[str, Any]] = []
    for iteration in iterations:
        batch = iteration["question_batch"]
        questions = batch["questions"]
        batch_row = {
            "batch_id": batch["batch_id"],
            "global_batch_id": batch["global_batch_id"],
            "iteration": iteration["iteration"],
            "code_id": iteration["code_id"],
            "question_count": len(questions),
            "valid_question_count": sum(
                item.get("oracle_status") == "valid" for item in questions
            ),
        }
        batch_row["batch_valid"] = batch_row["valid_question_count"] > 0
        all_questions.extend(questions)
        all_batches.append(batch_row)
        local = _metrics(questions, [batch_row], total_key_questions)
        so_far = _metrics(all_questions, all_batches, total_key_questions)
        batch["question_count"] = batch_row["question_count"]
        batch["valid_question_count"] = batch_row["valid_question_count"]
        batch["invalid_question_count"] = (
            batch_row["question_count"] - batch_row["valid_question_count"]
        )
        batch["batch_valid"] = batch_row["batch_valid"]
        batch["metrics"] = {
            **local,
            "precision_ques_so_far": so_far["precision_ques"],
            "recall_ques_so_far": so_far["recall_ques"],
            "ask_f1_ques_so_far": so_far["ask_f1_ques"],
            "total_generated_questions_so_far": so_far["total_generated_questions"],
            "correct_generated_questions_so_far": so_far[
                "correct_generated_questions"
            ],
            "covered_key_questions_so_far": so_far["covered_key_questions"],
        }
        iteration["metrics"] = dict(batch["metrics"])

    task_metrics = _metrics(all_questions, all_batches, total_key_questions)
    task_metrics["evaluations"] = all_batches
    hit_sequence = [
        int(question.get("oracle_status") == "valid")
        for question in all_questions
    ]
    task_metrics["TKQR"] = calculate_tkqr(
        hit_sequence + [0], total_key_questions
    )
    task_metrics["ORA"] = calculate_ora(len(all_questions), total_key_questions)
    return task_metrics


def task_result_path(output_dir: str | Path, task_id: Any) -> Path:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(task_id))
    return Path(output_dir) / "result" / f"{safe or 'unknown_task'}.json"


def write_task_result(record: dict[str, Any], output_dir: str | Path) -> Path:
    path = task_result_path(output_dir, record.get("task_id"))
    write_json(record, path)
    return path


def clear_task_results(output_dir: str | Path) -> None:
    result_dir = Path(output_dir) / "result"
    if result_dir.exists():
        for path in result_dir.glob("*.json"):
            path.unlink()


def sync_task_results(records: Iterable[dict[str, Any]], output_dir: str | Path) -> None:
    result_dir = Path(output_dir) / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    expected = {write_task_result(record, output_dir) for record in records}
    for path in result_dir.glob("*.json"):
        if path not in expected:
            path.unlink()


def summarize_questions(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_question_records(records)
    scored = [record for record in records if isinstance(record.get("question_metrics"), dict)]
    if scored:
        summary["TKQR"] = sum(record["question_metrics"]["TKQR"] for record in scored) / len(scored)
        summary["ORA"] = sum(record["question_metrics"]["ORA"] for record in scored) / len(scored)
    return summary


def summarize_passrates(records: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_passrate_records(records, benchmark="clarifycodebench")


def summarize_iterations(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    question_calls = [int(record.get("total_question_calls", 0)) for record in records]
    valid_counts = [int(record.get("valid_clarification_count", 0)) for record in records]
    return {
        "sample_count": count,
        "average_question_calls": sum(question_calls) / count if count else 0.0,
        "average_valid_clarifications": sum(valid_counts) / count if count else 0.0,
        "tasks_with_valid_clarification": sum(value > 0 for value in valid_counts),
        "stop_reasons": dict(Counter(record.get("stop_reason") for record in records)),
        "reconstruction_enabled": False,
    }


def write_text_log(
    output_dir: str | Path,
    task_id: Any,
    *,
    problem: str,
    iterations: list[dict[str, Any]],
    llm_calls: list[dict[str, Any]],
    stop_reason: str | None,
    question_metrics: dict[str, Any] | None,
    final_specification: str | None = None,
    final_code: str | None = None,
) -> None:
    lines = ["PROBLEM DESCRIPTION", "=" * 80, problem, ""]
    for iteration in iterations:
        lines.extend(
            [
                f"ITERATION {iteration['iteration']} | {iteration['state_id']} | {iteration['code_id']}",
                "=" * 80,
                "CURRENT SPECIFICATION",
                iteration["specification"],
                "",
                f"COMPLETE CODE {iteration['code_id']}",
                iteration["complete_code"],
                "",
                f"QUESTION BATCH {iteration['question_batch']['global_batch_id']}",
            ]
        )
        for question in iteration["question_batch"].get("questions", []):
            lines.extend(
                [
                    f"[{question['global_question_id']}] {question['question']}",
                    f"status: {question.get('oracle_status')}",
                    f"matched: {question.get('matched_key_question')}",
                    f"answer: {question.get('answer')}",
                ]
            )
        if not iteration["question_batch"].get("questions"):
            lines.append("(empty question batch)")
        lines.append("")

    if final_code is not None:
        lines.extend(
            [
                "FINAL SPECIFICATION",
                "=" * 80,
                final_specification or problem,
                "",
                "FINAL CODE C1",
                "=" * 80,
                final_code,
                "",
            ]
        )

    lines.extend(["LLM PROMPTS AND REASONING", "=" * 80])
    for call in llm_calls:
        call_id = call.get("call_id", "unknown")
        lines.extend(
            [
                f"--- SYSTEM PROMPT {call_id} ---",
                str(call.get("system_prompt") or ""),
                f"--- USER PROMPT {call_id} ---",
                str(call.get("user_prompt") or ""),
                f"--- REASONING {call_id} ---",
                str(call.get("reasoning") or ""),
                "",
            ]
        )
    lines.extend(
        [
            "STOP REASON",
            "=" * 80,
            stop_reason or "in_progress",
            "",
            "QUESTION METRICS",
            "=" * 80,
            json.dumps(question_metrics, ensure_ascii=False, indent=2)
            if question_metrics is not None
            else "in_progress",
            "",
        ]
    )
    safe_task_id = "".join(
        c if c.isalnum() or c in "._-" else "_" for c in str(task_id)
    )
    path = Path(output_dir) / "logs" / f"{safe_task_id or 'unknown_task'}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary.write("\n".join(lines))
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def code_sha256(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()
