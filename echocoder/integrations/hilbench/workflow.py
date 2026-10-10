
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from baselines.common import (
    log_step,
    set_llm_call_log,
    set_live_text_refresh,
    summarize_llm_call_usage,
    write_task_error_log,
    write_task_log,
)
from echocoder.integrations.base import IntegrationConfig, Mode, run_integrated_echocoder
from echocoder.iterative.metrics import question_metrics_from_rounds

from .backend import create_backend
from .evaluation import evaluate_final
from .models import HilBenchTask
from .sweagent.assumptions import analyze_accepted, append_analysis


def method_for_mode(mode: str | Mode) -> str:
    if Mode(mode) != Mode.ECHOCODER_SWEAGENT:
        raise ValueError(f"Unsupported HiL-Bench mode: {mode}")
    return "echocoder_sweagent"


def evaluate_code_states(
    task: HilBenchTask,
    rounds: list[dict[str, Any]],
    selected_submission: str,
    selected_evaluation: dict[str, Any],
    *,
    timeout: int,
    method: str,
) -> list[dict[str, Any]]:
    """Evaluate generated C_i after the selected submission, reusing its result when identical."""
    evaluations = []
    for index, round_data in enumerate(rounds):
        code_id = round_data.get("code_id") or f"C{index}"
        candidate = round_data["complete_code"]
        if candidate == selected_submission:
            evaluation = selected_evaluation
        else:
            log_step(method, task.task_id, f"evaluate_{code_id}", "start")
            try:
                evaluation = evaluate_final(task, candidate, timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                evaluation = {
                    "enabled": False,
                    "passed": None,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            log_step(method, task.task_id, f"evaluate_{code_id}", "done")
        evaluations.append({
            "code_id": code_id,
            "state_id": round_data.get("state_id") or f"S{index}",
            "generation_id": round_data.get("generation_id"),
            "generation_operation": round_data.get("generation_operation"),
            "specification_state": index,
            "evaluation": evaluation,
        })
    return evaluations


def _create_task_backend(
    task: HilBenchTask,
    *,
    client: Any,
    args: argparse.Namespace,
    output_dir: Path,
) -> Any:
    return create_backend(
        task,
        mode=Mode(args.mode),
        client=client,
        model=args.model,
        judge_model=args.question_judge_model or args.model,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        question_consistency=args.question_consistency,
        sweagent_artifact_root=output_dir / "sweagent" / task.task_id,
        sweagent_config_path=Path(args.sweagent_config).resolve(),
        sweagent_model=args.sweagent_model,
        sweagent_max_steps=args.sweagent_max_steps,
        sweagent_timeout=args.sweagent_timeout,
        sweagent_cost_limit=args.sweagent_cost_limit,
        sweagent_call_limit=args.sweagent_call_limit,
        progressive_edit=args.progressive_edit,
    )


def process_task(task: HilBenchTask, *, client: Any, args: argparse.Namespace, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    method = method_for_mode(args.mode)
    llm_call_logs: list[dict[str, Any]] = []
    set_llm_call_log(
        llm_call_logs,
        live_output_dir=output_dir,
        task_id=task.task_id,
    )
    backend = None
    latest_round_records: list[Any] = []
    latest_stop_reason: str | None = None

    def annotate_rounds(rounds_data: list[dict[str, Any]]) -> None:
        for state_index, round_data in enumerate(rounds_data):
            if task.task_type == "swe" and Mode(args.mode) == Mode.ECHOCODER_SWEAGENT:
                round_data["generation_engine"] = "upstream_sweagent"
                round_data["generation_operation"] = (
                    "sweagent_progressive_patch_edit"
                    if args.progressive_edit and state_index > 0
                    else "sweagent_clean_environment_complete_patch_generation"
                )
            else:
                round_data["generation_engine"] = "echocoder_sql_generation"
                round_data["generation_operation"] = (
                    "initial_complete_sql_generation"
                    if state_index == 0
                    else "complete_sql_repair_or_regeneration_after_clarification"
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
            "trajectory_id": f"{task.task_id}_trajectory",
            "rounds": rounds_data,
            "stop_reason": stop_reason,
        }
        write_task_log(
            output_dir,
            task.task_id,
            {
                "method": method,
                "workflow_version": "kr_kq_v1",
                "benchmark": "hil-bench",
                "task_type": task.task_type,
                "question_id": task.metadata.get("hf_task_id")
                or task.metadata.get("instance_id")
                or task.task_id,
                "input_problem": task.problem,
                "qa_pairs": task.qa_pairs,
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
        backend = _create_task_backend(
            task,
            client=client,
            args=args,
            output_dir=output_dir,
        )
        result = run_integrated_echocoder(
            problem=task.problem,
            backend=backend,
            config=IntegrationConfig(
                mode=Mode(args.mode),
                reconstruction_budget=args.reconstruction_budget,
                question_call_budget=args.question_call_budget,
                semantic_stop_on_empty_questions=task.task_type == "swe",
            ),
            analyze_accepted=(
                lambda specification, reconstruction, accepted: analyze_accepted(
                    client,
                    model=args.model,
                    specification=specification,
                    reconstruction=reconstruction,
                    accepted=accepted,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                )
                if args.progressive_edit else None
            ),
            append_analysis=append_analysis if args.progressive_edit else None,
            progress=lambda round_index, reconstruction_index, question_call_index, step, status: log_step(
                method,
                task.task_id,
                ".".join(
                    part
                    for part in (
                        f"state_{round_index}",
                        f"r_{reconstruction_index}"
                        if reconstruction_index is not None else "",
                        f"q_{question_call_index}"
                        if question_call_index is not None else "",
                        step,
                    )
                    if part
                ),
                status,
            ),
            snapshot=persist_snapshot,
        )
        result_data = result.to_dict()
        rounds = result_data["rounds"]
        annotate_rounds(rounds)
        selected_submission = result.final_solution
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
                        attempts.append({
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
                        })
        metrics = question_metrics_from_rounds(rounds, len(task.qa_pairs))
        if args.question_only:
            evaluation = {"enabled": False, "passed": None, "reason": "--question-only was set"}
        else:
            log_step(method, task.task_id, "hidden_test_evaluation", "start")
            evaluation = evaluate_final(task, selected_submission, timeout=args.eval_timeout)
            log_step(method, task.task_id, "hidden_test_evaluation", "done")
        if getattr(args, "evaluate_each_code_state", False) and not args.question_only:
            code_state_evaluations = evaluate_code_states(
                task,
                rounds,
                selected_submission,
                evaluation,
                timeout=args.eval_timeout,
                method=method,
            )
            initial_evaluation = code_state_evaluations[0]["evaluation"]
            initial_passed = (
                bool(initial_evaluation.get("passed"))
                if initial_evaluation.get("enabled") else None
            )
        else:
            code_state_evaluations = {
                "status": "not_run",
                "reason": (
                    "--question-only was set" if args.question_only
                    else "--evaluate-each-code-state was not set"
                ),
            }
            initial_evaluation = None
            initial_passed = None
        first_valid = next((item["attempt"] for item in attempts if item["oracle_status"] == "valid"), None)
        first_valid_call = next(
            (
                item["question_call_number"]
                for item in attempts
                if item["oracle_status"] == "valid"
            ),
            None,
        )
        ground_truth = [
            {
                "ground_truth_id": str(
                    qa.get("blocker_id") or qa.get("question_id") or f"GT_{index}"
                ),
                "question": qa.get("question"),
                "answer": qa.get("answer"),
            }
            for index, qa in enumerate(task.qa_pairs, start=1)
        ]
        record = {
            "task_id": task.task_id,
            "question_id": task.metadata.get("hf_task_id") or task.metadata.get("instance_id") or task.task_id,
            "benchmark": "hil-bench",
            "task_type": task.task_type,
            "method": method,
            "workflow_version": "kr_kq_v1",
            "trajectory_id": f"{task.task_id}_trajectory",
            "problem": task.problem,
            "rounds": rounds,
            "clarification_history": result_data["clarification_history"],
            "rejected_questions": result.rejected_questions,
            "stop_reason": result.stop_reason,
            "final_specification": result.final_specification,
            "final_solution": selected_submission,
            "final_state_id": rounds[-1].get("state_id") or f"S{len(rounds) - 1}",
            "final_code_id": rounds[-1].get("code_id")
            or f"C{len(rounds) - 1}",
            "clarification_probe": result.final_solution,
            "selected_submission": selected_submission,
            "sweagent_runs": backend.final_artifacts(),
            "generated_questions": [item["question"] for item in attempts],
            "clarification_attempts": attempts,
            "valid_clarification_found": bool(result.clarification_history),
            "valid_clarification_count": len(result.clarification_history),
            "clarification_attempt_number": first_valid,
            "clarification_call_number": first_valid_call,
            "reconstruction_budget": args.reconstruction_budget,
            "question_call_budget": args.question_call_budget,
            "total_oracle_questions": result.total_oracle_questions,
            "total_question_calls": result.total_question_calls,
            "question_metrics": metrics,
            "pass_evaluation": evaluation,
            "initial_code_passed": initial_passed,
            "initial_code_evaluation": initial_evaluation,
            "final_code_evaluation": evaluation,
            "final_code_passed": (
                bool(evaluation.get("passed")) if evaluation.get("enabled") else None
            ),
            "final_correct": (
                bool(evaluation.get("passed")) if evaluation.get("enabled") else None
            ),
            "ground_truth": ground_truth,
            "code_state_evaluations": code_state_evaluations,
            "llm_calls": llm_call_logs,
            "token_usage": summarize_llm_call_usage(llm_call_logs),
            "run_config": {
                "K_r": args.reconstruction_budget,
                "K_q": args.question_call_budget,
                "max_tokens": args.max_tokens,
                "max_retries": args.max_retries,
                "question_consistency": args.question_consistency,
                "model": args.model,
                "judge_model": args.question_judge_model or args.model,
                "mode": args.mode,
                "progressive_edit": args.progressive_edit,
                "evaluate_each_code_state": bool(getattr(args, "evaluate_each_code_state", False)),
                "evaluation_timing": "post_hoc_after_clarification",
                "evaluation_used_as_signal": False,
                "sweagent_max_steps": args.sweagent_max_steps,
                "sweagent_timeout": args.sweagent_timeout,
            },
            "generation_metadata": {
                "model": args.model,
                "prompt_version": "method_v3",
                "mode": args.mode,
                "progressive_edit": args.progressive_edit,
            },
        }
        payload = {
            "method": method,
            "workflow_version": "kr_kq_v1",
            "benchmark": "hil-bench",
            "task_type": task.task_type,
            "question_id": record["question_id"],
            "input_problem": task.problem,
            "qa_pairs": task.qa_pairs,
            "steps": {
                "budgets": {
                    "K_r": args.reconstruction_budget,
                    "K_q": args.question_call_budget,
                },
                "rounds": rounds,
                "final_specification": result.final_specification,
                "stop_reason": result.stop_reason,
                    "sweagent_runs": record["sweagent_runs"],
            },
            "generated_questions": record["generated_questions"],
            "llm_call_logs": llm_call_logs,
            "question_metrics": metrics,
            "pass_evaluation": evaluation,
            "record": record,
        }
        return record, payload
    except Exception as exc:  # noqa: BLE001
        log_step(method, task.task_id, "task", f"failed ({type(exc).__name__}: {exc})")
        write_task_error_log(
            output_dir,
            task.task_id,
            method=method,
            question_id=str(task.metadata.get("hf_task_id") or task.task_id),
            benchmark="hil-bench",
            task_type=task.task_type,
            step="task",
            error=exc,
            input_problem=task.problem,
            qa_pairs=task.qa_pairs,
            llm_call_logs=llm_call_logs,
        )
        return None
    finally:
        if backend is not None:
            backend.close()
        set_llm_call_log(None)
