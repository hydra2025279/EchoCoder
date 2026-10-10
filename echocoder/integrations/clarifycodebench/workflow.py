from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from baselines.common import (
    evaluate_pass_rate,
    log_step,
    set_llm_call_log,
    set_live_text_refresh,
    skipped_pass_evaluation,
    summarize_llm_call_usage,
    write_task_error_log,
    write_task_log,
)
from echocoder.clarification import generate_questions
from echocoder.core import run_specification_completion
from echocoder.generation import generate_complete_candidate
from echocoder.iterative.metrics import question_metrics_from_rounds
from echocoder.iterative.oracle import query_clarification_oracle_batch
from echocoder.reconstruction import reconstruct_requirements
from echocoder.schemas import OracleResult, WorkflowConfig


METHOD = "echocoder_iterative"


def augment_problem(problem: str, question: str, answer: str) -> str:
    """Append only an oracle-confirmed clarification to the specification."""
    return (
        f"{problem.rstrip()}\n\nClarification:\n"
        f"Q: {question.strip()}\nA: {answer.strip()}\n"
    )


def process_task(
    task: dict[str, Any],
    *,
    client: Any,
    args: argparse.Namespace,
    output_dir: Path,
    judge_model: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    task_id = task.get("task_id")
    problem = task["modified_content"]
    qa_pairs = task.get("qa_pairs", []) or []
    task_type = task.get("task_type", "programming")
    llm_call_logs: list[dict[str, Any]] = []
    set_llm_call_log(
        llm_call_logs,
        live_output_dir=output_dir,
        task_id=str(task_id) if task_id is not None else None,
    )

    def progress(
        round_index: int,
        reconstruction_index: int | None,
        question_call_index: int | None,
        step: str,
        status: str,
    ) -> None:
        location = f"state_{round_index}"
        if reconstruction_index is not None:
            location += f".r_{reconstruction_index}"
        if question_call_index is not None:
            location += f".q_{question_call_index}"
        log_step(METHOD, task_id, f"{location}.{step}", status)

    latest_round_records: list[Any] = []
    latest_stop_reason: str | None = None

    def annotate_rounds(rounds_data: list[dict[str, Any]]) -> None:
        for state_index, round_data in enumerate(rounds_data):
            round_data["generation_engine"] = "echocoder_integrated_generation"
            round_data["generation_operation"] = (
                "initial_question_detection_and_complete_code_generation"
                if state_index == 0
                else "complete_code_repair_or_regeneration_after_clarification"
            )

    def persist_snapshot(round_records: list[Any], stop_reason: str | None) -> None:
        nonlocal latest_round_records, latest_stop_reason
        latest_round_records = round_records
        latest_stop_reason = stop_reason
        rounds_data = [asdict(round_record) for round_record in round_records]
        annotate_rounds(rounds_data)
        generated_questions = [
            question.get("question")
            for round_data in rounds_data
            for reconstruction in round_data.get("reconstructions", [])
            for batch in reconstruction.get("question_batches", [])
            for question in batch.get("questions", [])
        ]
        live_record = {
            "trajectory_id": f"{task_id}_trajectory",
            "rounds": rounds_data,
            "stop_reason": stop_reason,
        }
        write_task_log(
            output_dir,
            task_id,
            {
                "method": METHOD,
                "workflow_version": "kr_kq_v1",
                "benchmark": "clarifycodebench",
                "task_type": task_type,
                "question_id": task.get("question_id"),
                "input_problem": problem,
                "qa_pairs": qa_pairs,
                "steps": {
                    "budgets": {
                        "K_r": args.reconstruction_budget,
                        "K_q": args.question_call_budget,
                    },
                    "rounds": rounds_data,
                    "stop_reason": stop_reason or "in_progress",
                },
                "generated_questions": generated_questions,
                "llm_call_logs": llm_call_logs,
                "question_metrics": None,
                "record": live_record,
            },
        )

    set_live_text_refresh(
        lambda: persist_snapshot(latest_round_records, latest_stop_reason)
    )

    try:
        result = run_specification_completion(
            problem=problem,
            generate=lambda specification, previous, round_index, valid, invalid: generate_complete_candidate(
                client,
                model=args.model,
                specification=specification,
                task_type=task_type,
                previous_valid_clarifications=valid,
                previous_invalid_questions=invalid,
                include_history=not getattr(args, "no_history", False),
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
            ),
            reconstruct=lambda specification, solution: reconstruct_requirements(
                client,
                model=args.model,
                specification=specification,
                solution=solution,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
            ),
            detect=lambda specification, solution, reconstruction, initial, valid, invalid: generate_questions(
                client,
                model=args.model,
                specification=specification,
                solution=solution,
                reconstruction=reconstruction,
                initial_questions=initial,
                previous_valid_clarifications=valid,
                previous_invalid_questions=invalid,
                include_history=not getattr(args, "no_history", False),
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
            ),
            ask_oracle_batch=lambda questions: [
                OracleResult(**oracle_result)
                for oracle_result in query_clarification_oracle_batch(
                    client,
                    questions=questions,
                    qa_pairs=qa_pairs,
                    judge_model=judge_model,
                    consistency=args.question_consistency,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                )
            ],
            update_specification=augment_problem,
            config=WorkflowConfig(
                reconstruction_budget=args.reconstruction_budget,
                question_call_budget=args.question_call_budget,
                semantic_stop_on_empty_questions=True,
            ),
            progress=progress,
            snapshot=persist_snapshot,
        )

        result_data = result.to_dict()
        rounds = result_data["rounds"]
        annotate_rounds(rounds)
        attempts = []
        question_call_number = 0
        for round_data in rounds:
            state_index = int(round_data["specification_round"])
            state_id = round_data.get("state_id") or f"S{state_index}"
            code_id = round_data.get("code_id") or f"C{state_index}"
            for reconstruction in round_data["reconstructions"]:
                r = int(reconstruction["reconstruction_index"])
                reconstruction_id = (
                    reconstruction.get("reconstruction_id") or f"{state_id}_{code_id}_R{r}"
                )
                for batch in reconstruction["question_batches"]:
                    question_call_number += 1
                    q = int(batch["question_call_index"])
                    batch_id = batch.get("batch_id") or f"Q{r}_{q}"
                    global_batch_id = (
                        batch.get("global_batch_id")
                        or f"{state_id}_{code_id}_{batch_id}"
                    )
                    for question_index, question in enumerate(
                        batch["questions"], start=1
                    ):
                        attempts.append(
                            {
                                "attempt": len(attempts) + 1,
                                "question_call_number": question_call_number,
                                "specification_round": state_index,
                                "state_id": state_id,
                                "generation_id": round_data.get("generation_id"),
                                "code_id": code_id,
                                "reconstruction_index": r,
                                "reconstruction_id": reconstruction_id,
                                "question_call_index": q,
                                "batch_id": batch_id,
                                "global_batch_id": global_batch_id,
                                "question_id": question.get("question_id")
                                or f"{batch_id}_{question_index}",
                                "global_question_id": question.get(
                                    "global_question_id"
                                )
                                or f"{global_batch_id}_{question_index}",
                                **question,
                            }
                        )
        question_metrics = question_metrics_from_rounds(rounds, len(qa_pairs))
        initial_code = rounds[0]["complete_code"]
        final_code = result.final_solution

        if args.question_only:
            final_evaluation = skipped_pass_evaluation("--question-only was set")
        else:
            log_step(METHOD, task_id, "evaluate_final_code", "start")
            final_evaluation = evaluate_pass_rate(
                pass_command=args.pass_command,
                task_id=task_id,
                question_id=task.get("question_id"),
                codes=[final_code],
            )
            log_step(METHOD, task_id, "evaluate_final_code", "done")
        first_valid_attempt = next(
            (a["attempt"] for a in attempts if a["oracle_status"] == "valid"), None
        )
        first_valid_call = next(
            (
                a["question_call_number"]
                for a in attempts
                if a["oracle_status"] == "valid"
            ),
            None,
        )
        generated_questions = [a["question"] for a in attempts]
        ground_truth = [
            {
                "ground_truth_id": str(
                    qa.get("blocker_id") or qa.get("question_id") or f"GT_{index}"
                ),
                "question": qa.get("question"),
                "answer": qa.get("answer"),
            }
            for index, qa in enumerate(qa_pairs, start=1)
        ]
        record = {
            "task_id": task_id,
            "question_id": task.get("question_id"),
            "benchmark": "clarifycodebench",
            "task_type": task_type,
            "method": METHOD,
            "workflow_version": "kr_kq_v1",
            "trajectory_id": f"{task_id}_trajectory",
            "problem": problem,
            "original_problem": problem,
            "rounds": rounds,
            "clarification_history": result_data["clarification_history"],
            "rejected_questions": result.rejected_questions,
            "stop_reason": result.stop_reason,
            "final_specification": result.final_specification,
            "final_solution": final_code,
            "final_state_id": rounds[-1].get("state_id") or f"S{len(rounds) - 1}",
            "final_code_id": rounds[-1].get("code_id") or f"C{len(rounds) - 1}",
            "final_candidates": [final_code],
            "selected_submission": final_code,
            "initial_code": initial_code,
            "final_code": final_code,
            "clarification_attempts": attempts,
            "generated_questions": generated_questions,
            "valid_clarification_found": bool(result.clarification_history),
            "valid_clarification_count": len(result.clarification_history),
            "clarification_attempt_number": first_valid_attempt,
            "clarification_call_number": first_valid_call,
            "reconstruction_budget": args.reconstruction_budget,
            "question_call_budget": args.question_call_budget,
            "total_oracle_questions": result.total_oracle_questions,
            "total_question_calls": result.total_question_calls,
            "pass_evaluation": final_evaluation,
            "question_metrics": question_metrics,
            "ground_truth": ground_truth,
            "llm_calls": llm_call_logs,
            "token_usage": summarize_llm_call_usage(llm_call_logs),
            "run_config": {
                "K_r": args.reconstruction_budget,
                "K_q": args.question_call_budget,
                "max_tokens": args.max_tokens,
                "max_retries": args.max_retries,
                "question_consistency": args.question_consistency,
                "no_history": bool(getattr(args, "no_history", False)),
                "evaluation_timing": "post_hoc_after_clarification",
                "evaluation_used_as_signal": False,
                "model": args.model,
                "judge_model": judge_model,
            },
            "generation_metadata": {"model": args.model, "prompt_version": "method_v3"},
        }
        log_payload = {
            "method": METHOD,
            "workflow_version": "kr_kq_v1",
            "benchmark": "clarifycodebench",
            "task_type": task_type,
            "question_id": task.get("question_id"),
            "input_problem": problem,
            "qa_pairs": qa_pairs,
            "steps": {
                "budgets": {
                    "K_r": args.reconstruction_budget,
                    "K_q": args.question_call_budget,
                },
                "rounds": rounds,
                "final_specification": result.final_specification,
                "stop_reason": result.stop_reason,
            },
            "generated_questions": generated_questions,
            "llm_call_logs": llm_call_logs,
            "question_metrics": question_metrics,
            "record": record,
        }
        return record, log_payload
    except Exception as exc:  # noqa: BLE001
        log_step(METHOD, task_id, "task", f"failed ({type(exc).__name__}: {exc})")
        write_task_error_log(
            output_dir,
            task_id,
            method=METHOD,
            question_id=task.get("question_id"),
            benchmark="clarifycodebench",
            task_type=task_type,
            step="task",
            error=exc,
            input_problem=problem,
            qa_pairs=qa_pairs,
            llm_call_logs=llm_call_logs,
        )
        return None
    finally:
        set_llm_call_log(None)
