from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from baselines.ablation_reconstruction.artifacts import (
    attach_metrics,
    code_sha256,
    write_text_log,
)
from baselines.ablation_reconstruction.oracle import query_oracle_batch
from baselines.ablation_reconstruction.step_generate_code import generate_code
from baselines.ablation_reconstruction.step_generate_questions import generate_questions
from baselines.common import (
    evaluate_pass_rate,
    llm_call_context,
    log_step,
    set_live_text_refresh,
    set_llm_call_log,
    skipped_pass_evaluation,
    summarize_llm_call_usage,
    write_task_error_log,
)


METHOD = "ablation_reconstruction"


def augment_specification(
    problem: str, clarifications: list[dict[str, str]]
) -> str:
    specification = problem.rstrip()
    for item in clarifications:
        specification += (
            "\n\nClarification:\n"
            f"Q: {item['question'].strip()}\n"
            f"A: {item['answer'].strip()}"
        )
    return specification


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
    llm_calls: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    valid_history: list[dict[str, str]] = []
    invalid_history: list[str] = []
    accepted_target_ids: set[str] = set()
    stop_reason: str | None = None
    question_metrics: dict[str, Any] | None = None
    final_specification = problem
    final_code: str | None = None

    set_llm_call_log(
        llm_calls,
        live_output_dir=output_dir,
        task_id=str(task_id) if task_id is not None else None,
    )

    def persist() -> None:
        write_text_log(
            output_dir,
            task_id,
            problem=problem,
            iterations=iterations,
            llm_calls=llm_calls,
            stop_reason=stop_reason,
            question_metrics=question_metrics,
            final_specification=final_specification,
            final_code=final_code,
        )

    set_live_text_refresh(persist)

    def fail(step: str, error: Exception) -> None:
        log_step(METHOD, task_id, step, f"failed ({type(error).__name__}: {error})")
        write_task_error_log(
            output_dir,
            task_id,
            method=METHOD,
            question_id=task.get("question_id"),
            benchmark="clarifycodebench",
            task_type=task.get("task_type", "programming"),
            step=step,
            error=error,
            input_problem=problem,
            qa_pairs=qa_pairs,
            llm_call_logs=llm_calls,
        )

    try:
        state_id = "S0"
        code_id = "C0"
        generation_id = "S0_GEN"
        log_step(METHOD, task_id, f"{state_id}.generate_code", "start")
        with llm_call_context(
            phase="generation",
            specification_state=0,
            state_id=state_id,
            generation_id=generation_id,
            generation_operation="complete_code_generation",
            code_id=code_id,
        ):
            code, generation_response = generate_code(
                client,
                model=args.model,
                specification=problem,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
            )
        log_step(METHOD, task_id, f"{state_id}.generate_code", "done")

        for iteration_index in range(args.iteration_budget):
            batch_id = f"Q{iteration_index + 1}"
            global_batch_id = f"{state_id}_{code_id}_{batch_id}"

            iteration = {
                "iteration": iteration_index,
                "state_id": state_id,
                "generation_id": generation_id,
                "code_id": code_id,
                "code_sha256": code_sha256(code),
                "specification": problem,
                "complete_code": code,
                "generation_response": generation_response if iteration_index == 0 else None,
                "question_batch": {
                    "question_call_index": iteration_index + 1,
                    "batch_id": batch_id,
                    "global_batch_id": global_batch_id,
                    "detector_response": None,
                    "questions": [],
                },
                "accepted_clarifications": [],
            }
            iterations.append(iteration)
            persist()

            log_step(METHOD, task_id, f"{state_id}.{batch_id}.generate_questions", "start")
            with llm_call_context(
                phase="question_generation",
                specification_state=0,
                state_id=state_id,
                generation_id=generation_id,
                code_id=code_id,
                q=iteration_index + 1,
                batch_id=batch_id,
                global_batch_id=global_batch_id,
            ):
                detector_response, questions = generate_questions(
                    client,
                    model=args.model,
                    specification=problem,
                    code=code,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                )
            iteration["question_batch"]["detector_response"] = detector_response
            log_step(
                METHOD,
                task_id,
                f"{state_id}.{batch_id}.generate_questions",
                f"done ({len(questions)} question(s))",
            )

            if not questions:
                stop_reason = "no_questions_from_detector"
                persist()
                break

            log_step(METHOD, task_id, f"{state_id}.{batch_id}.question_oracle", "start")
            with llm_call_context(
                phase="question_oracle",
                specification_state=0,
                state_id=state_id,
                code_id=code_id,
                q=iteration_index + 1,
                batch_id=batch_id,
                global_batch_id=global_batch_id,
            ):
                oracle_results = query_oracle_batch(
                    client,
                    questions=questions,
                    qa_pairs=qa_pairs,
                    judge_model=judge_model,
                    consistency=args.question_consistency,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                )
            accepted: list[dict[str, str]] = []
            for question_index, (question, oracle) in enumerate(
                zip(questions, oracle_results, strict=True), start=1
            ):
                evaluation = dict(oracle.get("evaluation") or {})
                target_id = evaluation.get("matched_key_id") or oracle.get(
                    "matched_key_question"
                )
                already_resolved = (
                    oracle.get("status") == "valid"
                    and target_id is not None
                    and str(target_id) in accepted_target_ids
                )
                status = "invalid" if already_resolved else oracle.get("status")
                answer = None if already_resolved else oracle.get("answer")
                if already_resolved:
                    evaluation["already_resolved"] = True
                question_record = {
                    "question_id": f"{batch_id}_{question_index}",
                    "global_question_id": f"{global_batch_id}_{question_index}",
                    "question": question,
                    "oracle_status": status,
                    "answer": answer,
                    "matched_key_question": oracle.get("matched_key_question"),
                    "oracle_evaluation": evaluation,
                }
                iteration["question_batch"]["questions"].append(question_record)
                if status == "valid" and answer:
                    clarification = {"question": question, "answer": answer}
                    accepted.append(clarification)
                    valid_history.append(clarification)
                    if target_id is not None:
                        accepted_target_ids.add(str(target_id))
                else:
                    invalid_history.append(question)

            iteration["accepted_clarifications"] = accepted
            log_step(
                METHOD,
                task_id,
                f"{state_id}.{batch_id}.question_oracle",
                f"done ({len(accepted)} valid)",
            )
            persist()
        else:
            stop_reason = "iteration_budget_exhausted"

        question_metrics = attach_metrics(iterations, len(qa_pairs))
        for iteration in iterations:
            code_id = iteration["code_id"]
            generation_calls = [
                call
                for call in llm_calls
                if call.get("code_id") == code_id
                and call.get("phase") == "generation"
            ] if iteration["iteration"] == 0 else []
            question_calls = [
                call
                for call in llm_calls
                if call.get("global_batch_id") == iteration["question_batch"]["global_batch_id"]
                and call.get("phase") in {"question_generation", "question_oracle"}
            ]
            iteration["generation_llm_call_ids"] = [
                call.get("call_id") for call in generation_calls
            ]
            iteration["generation_token_usage"] = summarize_llm_call_usage(
                generation_calls
            )
            iteration["question_batch"]["llm_call_ids"] = [
                call.get("call_id") for call in question_calls
            ]
            iteration["question_batch"]["token_usage"] = summarize_llm_call_usage(
                question_calls
            )
            iteration["question_batch"]["evidence"] = {
                "specification_state": 0,
                "state_id": iteration["state_id"],
                "code_id": code_id,
                "code_sha256": iteration["code_sha256"],
            }
        final_specification = augment_specification(problem, valid_history)
        final_generation_response: str | None = None
        if args.question_only:
            log_step(METHOD, task_id, "S1.generate_final_code", "skipped (--question-only)")
        else:
            log_step(METHOD, task_id, "S1.generate_final_code", "start")
            with llm_call_context(
                phase="final_generation",
                specification_state=1,
                state_id="S1",
                generation_id="S1_GEN",
                generation_operation="complete_code_after_clarification",
                code_id="C1",
                parent_code_id="C0",
            ):
                final_code, final_generation_response = generate_code(
                    client,
                    model=args.model,
                    specification=final_specification,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                )
            log_step(METHOD, task_id, "S1.generate_final_code", "done")
            persist()

        if args.question_only:
            final_evaluation = skipped_pass_evaluation("--question-only was set")
        else:
            log_step(METHOD, task_id, "evaluate_C1", "start")
            final_evaluation = evaluate_pass_rate(
                pass_command=args.pass_command,
                task_id=task_id,
                question_id=task.get("question_id"),
                codes=[final_code],
            )
            log_step(METHOD, task_id, "evaluate_C1", "done")
        all_questions = [
            question
            for iteration in iterations
            for question in iteration["question_batch"]["questions"]
        ]
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
            "task_type": task.get("task_type", "programming"),
            "method": METHOD,
            "workflow_version": "fixed_code_independent_questions_final_code_v3",
            "original_problem": problem,
            "iterations": iterations,
            "rounds": [],
            "clarification_history": valid_history,
            "rejected_questions": invalid_history,
            "stop_reason": stop_reason,
            "final_specification": final_specification,
            "initial_code": code,
            "final_code": final_code,
            "final_code_id": "C1" if final_code is not None else None,
            "final_generation_response": final_generation_response,
            "selected_submission": final_code,
            "generated_questions": [item["question"] for item in all_questions],
            "question_metrics": question_metrics,
            "ground_truth": ground_truth,
            "valid_clarification_found": bool(valid_history),
            "valid_clarification_count": len(valid_history),
            "total_question_calls": len(iterations),
            "total_oracle_questions": len(all_questions),
            "iteration_budget": args.iteration_budget,
            "pass_evaluation": final_evaluation,
            "llm_calls": llm_calls,
            "token_usage": summarize_llm_call_usage(llm_calls),
            "run_config": {
                "iteration_budget": args.iteration_budget,
                "reconstruction_enabled": False,
                "code_generation_count": 1 if final_code is None else 2,
                "independent_question_calls": True,
                "question_history_used_for_question_generation": False,
                "accepted_clarifications_used_for_final_code": True,
                "evaluation_timing": "post_hoc_after_clarification",
                "evaluation_used_as_signal": False,
                "model": args.model,
                "judge_model": judge_model,
            },
        }
        record["ask_result"] = {
            "generated_questions": record["generated_questions"],
            "metrics": question_metrics,
            "iterations": iterations,
        }
        persist()
        return record, {
            "method": METHOD,
            "benchmark": "clarifycodebench",
            "input_problem": problem,
            "iterations": iterations,
            "llm_call_logs": llm_calls,
            "question_metrics": question_metrics,
            "record": record,
        }
    except Exception as error:  # noqa: BLE001
        fail("task", error)
        return None
    finally:
        set_live_text_refresh(None)
        set_llm_call_log(None)
