#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
from typing import Any, Callable

from baselines.common import (
    add_common_args,
    append_jsonl,
    client_from_args,
    clear_task_error_records,
    completed_task_ids,
    cost_jsonl_path,
    cost_record_from_result,
    cost_summary_path,
    evaluate_pass_rate,
    evaluate_questions,
    load_tasks,
    load_jsonl_records,
    load_task_error_records,
    log_step,
    llm_call_context,
    method_output_dir,
    parse_json_object,
    passrate_summary_path,
    question_summary_path,
    reset_jsonl,
    remove_task_error_record,
    resolve_results_path,
    set_llm_call_log,
    set_live_text_refresh,
    skipped_pass_evaluation,
    summarize_cost_records,
    summarize_llm_call_usage,
    summarize_passrate_records,
    summarize_question_records,
    write_json,
    write_jsonl,
    write_task_error_log,
    write_task_log,
)
from baselines.vanilla.artifacts import attach_iteration_metrics
from baselines.vanilla.step_generate_questions import generate_questions
from baselines.vanilla.oracle import validated_clarifications
from echocoder.generation import generate_code


def _generate_question_iterations(
    *,
    client,
    model: str,
    problem: str,
    max_tokens: int,
    max_retries: int,
    question_call_budget: int,
    method: str,
    task_id,
    on_iteration: Callable[[list[dict[str, Any]], str], None] | None = None,
) -> tuple[list[dict], list[str], str]:
    iterations = []
    generated_questions = []
    stop_reason = "question_call_budget_exhausted"

    for iteration in range(1, question_call_budget + 1):
        step = f"generate_questions_iteration_{iteration}"
        iteration_id = f"I{iteration}"
        log_step(method, task_id, step, "start")
        with llm_call_context(
            phase="question_generation",
            iteration=iteration,
            iteration_id=iteration_id,
            batch_id=iteration_id,
            global_batch_id=iteration_id,
        ):
            response = generate_questions(
                client,
                model=model,
                problem=problem,
                max_tokens=max_tokens,
                max_retries=max_retries,
                step_name=step,
            )
        question_data = parse_json_object(
            response, required_keys=("clarifying_questions",)
        )
        if question_data is None or not isinstance(
            question_data.get("clarifying_questions"), list
        ):
            raise RuntimeError(
                f"Iteration {iteration} did not return a clarifying_questions list"
            )
        batch = []
        seen_questions: set[str] = set()
        for value in question_data["clarifying_questions"]:
            if not isinstance(value, str):
                continue
            question = " ".join(value.split()).strip()
            key = question.casefold()
            if question and key not in seen_questions:
                seen_questions.add(key)
                batch.append(question)
        iterations.append(
            {
                "iteration": iteration,
                "iteration_id": iteration_id,
                "batch_id": iteration_id,
                "raw_response": response,
                "questions": batch,
                "question_count": len(batch),
                "empty": not batch,
                "status": "empty" if not batch else "generated",
            }
        )
        if on_iteration is not None:
            on_iteration(iterations, stop_reason if not batch else "in_progress")
        generated_questions.extend(batch)
        log_step(method, task_id, step, f"done ({len(batch)} question(s))")
        if not batch:
            stop_reason = "empty_question_batch"
            break

    return iterations, generated_questions, stop_reason


def _task_result_path(output_dir: Path, task_id: Any) -> Path:
    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id or "unknown_task"))
    return output_dir / "result" / f"{safe_task_id}.json"


def _write_task_result(output_dir: Path, record: dict[str, Any]) -> None:
    write_json(record, _task_result_path(output_dir, record.get("task_id")))


def _clear_task_results(output_dir: Path) -> None:
    result_dir = output_dir / "result"
    if result_dir.exists():
        for path in result_dir.glob("*.json"):
            path.unlink()


def _refresh_costs(records: list[dict[str, Any]], output_dir: Path) -> None:
    sources = [*records, *load_task_error_records(output_dir)]
    costs = [cost_record_from_result(record) for record in sources]
    write_jsonl(costs, cost_jsonl_path(output_dir))
    write_json(summarize_cost_records(costs), cost_summary_path(output_dir))


def _revise_specification(
    problem: str, clarifications: list[dict[str, str]]
) -> str:
    specification = problem.rstrip()
    for clarification in clarifications:
        specification += (
            "\n\nClarification:\n"
            f"Q: {clarification['question'].strip()}\n"
            f"A: {clarification['answer'].strip()}"
        )
    return specification


def _process_task(
    task: dict,
    *,
    client,
    args: argparse.Namespace,
    method: str,
    output_dir,
    question_judge_model: str,
):
    task_id = task.get("task_id")
    problem = task["modified_content"]
    qa_pairs = task.get("qa_pairs", []) or []
    llm_call_logs: list[dict[str, Any]] = []
    latest_iterations: list[dict[str, Any]] = []
    latest_generated_questions: list[str] = []
    latest_question_metrics: dict[str, Any] | None = None
    latest_stop_reason = "in_progress"
    latest_clarifications: list[dict[str, str]] = []
    latest_specification = problem
    latest_code: str | None = None
    latest_pass_evaluation: dict[str, Any] | None = None
    set_llm_call_log(
        llm_call_logs,
        live_output_dir=output_dir,
        task_id=str(task_id) if task_id is not None else None,
    )

    def persist_live() -> None:
        write_task_log(
            output_dir,
            task_id,
            {
                "method": method,
                "benchmark": task.get("benchmark", args.benchmark),
                "task_type": task.get("task_type"),
                "question_id": task.get("question_id"),
                "input_problem": problem,
                "qa_pairs": qa_pairs,
                "steps": {
                    "question_iterations": latest_iterations,
                    "stop_reason": latest_stop_reason,
                    "revised_specification": latest_specification,
                    "validated_clarifications": latest_clarifications,
                    "final_code": latest_code,
                },
                "generated_questions": latest_generated_questions,
                "llm_call_logs": llm_call_logs,
                "question_metrics": latest_question_metrics,
                "pass_evaluation": latest_pass_evaluation,
                "record": {
                    "task_id": task_id,
                    "status": "in_progress",
                    "question_iterations": latest_iterations,
                    "stop_reason": latest_stop_reason,
                    "final_specification": latest_specification,
                    "validated_clarifications": latest_clarifications,
                    "final_code": latest_code,
                },
            },
        )

    set_live_text_refresh(persist_live)

    def fail(step: str, error: Exception):
        log_step(method, task_id, step, f"failed ({type(error).__name__}: {error})")
        write_task_error_log(
            output_dir,
            task_id,
            method=method,
            question_id=task.get("question_id"),
            benchmark=task.get("benchmark", args.benchmark),
            task_type=task.get("task_type"),
            step=step,
            error=error,
            input_problem=problem,
            qa_pairs=qa_pairs,
            llm_call_logs=llm_call_logs,
        )
        return None

    try:
        try:
            def on_iteration(iterations: list[dict[str, Any]], stop_reason: str) -> None:
                nonlocal latest_iterations, latest_generated_questions, latest_stop_reason
                latest_iterations = iterations
                latest_generated_questions = [
                    question
                    for iteration in iterations
                    for question in iteration["questions"]
                ]
                latest_stop_reason = stop_reason
                persist_live()

            question_iterations, generated_questions, stop_reason = (
                _generate_question_iterations(
                    client=client,
                    model=args.model,
                    problem=problem,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                    question_call_budget=args.question_budget,
                    method=method,
                    task_id=task_id,
                    on_iteration=on_iteration,
                )
            )
        except Exception as exc:  # noqa: BLE001
            return fail("generate_question_iterations", exc)
        questions = (
            question_iterations[-1]["raw_response"]
            if question_iterations
            else '{"clarifying_questions": []}'
        )
        latest_iterations = question_iterations
        latest_generated_questions = generated_questions
        latest_stop_reason = stop_reason
        persist_live()
        log_step(
            method,
            task_id,
            "generate_question_iterations",
            (
                f"done ({len(question_iterations)} call(s), "
                f"{len(generated_questions)} total question(s), {stop_reason})"
            ),
        )

        if not question_iterations:
            return fail(
                "generate_question_iterations",
                RuntimeError("Question iteration loop produced no calls"),
            )

        log_step(method, task_id, "extract_questions", "start")
        try:
            generated_questions = [
                question
                for iteration in question_iterations
                for question in iteration["questions"]
            ]
        except Exception as exc:  # noqa: BLE001
            return fail("extract_questions", exc)
        log_step(
            method,
            task_id,
            "extract_questions",
            f"done ({len(generated_questions)} question(s) across all iterations)",
        )

        question_metrics = None
        clarifications: list[dict[str, str]] = []
        if args.skip_question_metrics:
            log_step(method, task_id, "question_metrics", "skipped")
        else:
            log_step(method, task_id, "question_metrics", "start")
            try:
                question_metadata = [
                    {
                        "phase": "question_judge",
                        "iteration": iteration["iteration"],
                        "iteration_id": iteration["iteration_id"],
                        "batch_id": iteration["batch_id"],
                        "global_batch_id": iteration["batch_id"],
                        "question_id": f"{iteration['iteration_id']}_Q{question_index}",
                        "global_question_id": (
                            f"{iteration['batch_id']}_Q{question_index}"
                        ),
                    }
                    for iteration in question_iterations
                    for question_index, _ in enumerate(iteration["questions"], start=1)
                ]
                judged_question_metrics = evaluate_questions(
                    client,
                    judge_model=question_judge_model,
                    generated_questions=generated_questions,
                    qa_pairs=task.get("qa_pairs", []) or [],
                    consistency=args.question_consistency,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                    question_metadata=question_metadata,
                )
                clarifications = validated_clarifications(
                    judged_question_metrics, qa_pairs
                )
                question_metrics = attach_iteration_metrics(
                    question_iterations,
                    judged_question_metrics["evaluations"],
                    len(qa_pairs),
                    qa_pairs,
                )
                latest_question_metrics = question_metrics
                latest_clarifications = clarifications
                latest_specification = _revise_specification(problem, clarifications)
                persist_live()
            except Exception as exc:  # noqa: BLE001
                return fail("question_metrics", exc)
            log_step(
                method,
                task_id,
                "question_metrics",
                f"done ({question_metrics['correct_questions']}/"
                f"{question_metrics['total_key_questions']} correct)",
            )

        revised_specification = _revise_specification(problem, clarifications)
        latest_clarifications = clarifications
        latest_specification = revised_specification
        final_code = None
        if args.question_only:
            pass_evaluation = skipped_pass_evaluation("--question-only was set")
            log_step(method, task_id, "generate_code", "skipped (--question-only)")
            log_step(method, task_id, "pass_evaluation", "skipped (--question-only)")
        else:
            log_step(method, task_id, "generate_code", "start")
            try:
                with llm_call_context(
                    phase="code_generation",
                    code_id="C0",
                    specification_state=0,
                ):
                    final_code = generate_code(
                        client,
                        model=args.model,
                        problem=revised_specification,
                        task_type=task.get("task_type") or "programming",
                        max_tokens=args.max_tokens,
                        max_retries=args.max_retries,
                    )
            except Exception as exc:  # noqa: BLE001
                return fail("generate_code", exc)
            latest_code = final_code
            persist_live()
            log_step(method, task_id, "generate_code", "done")
            log_step(method, task_id, "pass_evaluation", "start")
            try:
                pass_evaluation = evaluate_pass_rate(
                    pass_command=args.pass_command,
                    task_id=task_id,
                    question_id=task.get("question_id"),
                    codes=[final_code],
                )
            except Exception as exc:  # noqa: BLE001
                return fail("pass_evaluation", exc)
            log_step(method, task_id, "pass_evaluation", "done")
        latest_pass_evaluation = pass_evaluation
        persist_live()

        record = {
            "task_id": task_id,
            "question_id": task.get("question_id"),
            "benchmark": task.get("benchmark", args.benchmark),
            "task_type": task.get("task_type"),
            "method": method,
            "questions": questions,
            "question_responses": [
                iteration["raw_response"] for iteration in question_iterations
            ],
            "question_iterations": question_iterations,
            "iteration_count": len(question_iterations),
            "question_call_budget": args.question_budget,
            "stop_reason": stop_reason,
            "generated_questions": generated_questions,
            "question_metrics": question_metrics,
            "validated_clarifications": clarifications,
            "original_problem": problem,
            "final_specification": revised_specification,
            "final_code": final_code,
            "final_solution": final_code,
            "selected_submission": final_code,
            "code_states": (
                []
                if final_code is None
                else [
                    {
                        "code_id": "C0",
                        "specification": revised_specification,
                        "code": final_code,
                        "generation_reason": "post_question_loop_generation",
                    }
                ]
            ),
            "llm_calls": llm_call_logs,
            "token_usage": summarize_llm_call_usage(llm_call_logs),
            "status": "completed",
            "pass_evaluation": pass_evaluation,
        }
        log_payload = {
            "method": method,
            "benchmark": task.get("benchmark", args.benchmark),
            "task_type": task.get("task_type"),
            "question_id": task.get("question_id"),
            "input_problem": problem,
            "qa_pairs": qa_pairs,
            "steps": {
                "question_iterations": question_iterations,
                "stop_reason": stop_reason,
                "revised_specification": revised_specification,
                "validated_clarifications": clarifications,
                "final_code": final_code,
            },
            "generated_questions": generated_questions,
            "llm_call_logs": llm_call_logs,
            "question_metrics": question_metrics,
            "pass_evaluation": pass_evaluation,
            "record": record,
        }
        return record, log_payload
    except Exception as exc:  # noqa: BLE001
        return fail("task", exc)
    finally:
        set_live_text_refresh(None)
        set_llm_call_log(None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--question-budget",
        "--iteration-budget",
        type=int,
        default=6,
        help=(
            "maximum direct question-generation calls per task; each call sees "
            "only the original problem (default: 6)"
        ),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.question_budget < 1:
        parser.error("--question-budget must be at least 1")

    method = "vanilla"
    print(f"[{method}][setup] benchmark={args.benchmark} data={args.data or '(default)'}", flush=True)
    client = client_from_args(args)
    tasks = load_tasks(
        args.data,
        args.limit,
        benchmark=args.benchmark,
        task_ids=args.task_ids,
    )
    output_dir = method_output_dir(__file__, args.output_dir, benchmark=args.benchmark)
    results_path = resolve_results_path(args.out, output_dir)
    task_cost_path = cost_jsonl_path(output_dir)
    total_cost_path = cost_summary_path(output_dir)
    question_judge_model = args.question_judge_model or args.model
    print(f"[{method}][setup] loaded {len(tasks)} task(s)", flush=True)
    print(f"[{method}][setup] results -> {results_path}", flush=True)
    print(f"[{method}][setup] task_costs -> {task_cost_path}", flush=True)
    print(f"[{method}][setup] cost_summary -> {total_cost_path}", flush=True)
    print(f"[{method}][setup] logs -> {output_dir / 'logs'}", flush=True)
    print(f"[{method}][setup] question_summary -> {question_summary_path(output_dir)}", flush=True)
    print(f"[{method}][setup] passrate_summary -> {passrate_summary_path(output_dir)}", flush=True)
    existing_records = [] if args.no_resume else load_jsonl_records(results_path)
    done_task_ids = completed_task_ids(existing_records)
    if args.no_resume:
        reset_jsonl(results_path)
        reset_jsonl(task_cost_path)
        _clear_task_results(output_dir)
        clear_task_error_records(output_dir)
        write_json(summarize_cost_records([]), total_cost_path)
        write_json(summarize_question_records([]), question_summary_path(output_dir))
        if args.question_only:
            passrate_summary_path(output_dir).unlink(missing_ok=True)
        else:
            write_json(summarize_passrate_records([], benchmark=args.benchmark), passrate_summary_path(output_dir))
    else:
        print(f"[{method}][setup] resume -> {len(done_task_ids)} completed task(s)", flush=True)
        for record in existing_records:
            _write_task_result(output_dir, record)
            remove_task_error_record(output_dir, record.get("task_id"))
        _refresh_costs(existing_records, output_dir)
        write_json(summarize_question_records(existing_records), question_summary_path(output_dir))
        if args.question_only:
            passrate_summary_path(output_dir).unlink(missing_ok=True)
        else:
            write_json(summarize_passrate_records(existing_records, benchmark=args.benchmark), passrate_summary_path(output_dir))

    summary_records = list(existing_records)
    passrate_records = [
        record for record in existing_records
        if "pass_evaluation" in record or "pass_at_1" in record
    ]
    pending_tasks = []
    for task in tasks:
        task_id = task.get("task_id")
        if str(task_id) in done_task_ids:
            log_step(method, task_id, "task", "skipped (already in results.jsonl)")
        else:
            pending_tasks.append(task)

    print(f"[{method}][setup] batch_size={args.batch_size}", flush=True)
    print(
        f"[{method}][setup] question_budget={args.question_budget} "
        "(each call receives original P only)",
        flush=True,
    )
    batch_count = (len(pending_tasks) + args.batch_size - 1) // args.batch_size
    for batch_start in range(0, len(pending_tasks), args.batch_size):
        batch = pending_tasks[batch_start : batch_start + args.batch_size]
        batch_number = batch_start // args.batch_size + 1
        task_ids = ", ".join(str(task.get("task_id")) for task in batch)
        print(
            f"[{method}][batch {batch_number}/{batch_count}] start -> {task_ids}",
            flush=True,
        )

        with ThreadPoolExecutor(
            max_workers=len(batch),
            thread_name_prefix=f"{method}-batch-{batch_number}",
        ) as executor:
            future_to_task = {
                executor.submit(
                    _process_task,
                    task,
                    client=client,
                    args=args,
                    method=method,
                    output_dir=output_dir,
                    question_judge_model=question_judge_model,
                ): task
                for task in batch
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                task_id = task.get("task_id")
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    log_step(
                        method,
                        task_id,
                        "task",
                        f"failed unexpectedly ({type(exc).__name__}: {exc})",
                    )
                    write_task_error_log(
                        output_dir,
                        task_id,
                        method=method,
                        question_id=task.get("question_id"),
                        benchmark=task.get("benchmark", args.benchmark),
                        task_type=task.get("task_type"),
                        step="task",
                        error=exc,
                        input_problem=task["modified_content"],
                        qa_pairs=task.get("qa_pairs", []) or [],
                        llm_call_logs=[],
                    )
                    _refresh_costs(summary_records, output_dir)
                    continue
                if result is None:
                    _refresh_costs(summary_records, output_dir)
                    continue

                record, log_payload = result
                log_step(method, task_id, "write_task_log", "start")
                write_task_log(output_dir, task_id, log_payload)
                log_step(method, task_id, "write_task_log", "done")

                log_step(method, task_id, "append_result", "start")
                append_jsonl(record, results_path)
                log_step(method, task_id, "append_result", "done")
                _write_task_result(output_dir, record)
                done_task_ids.add(str(task_id))
                summary_records.append(record)

                log_step(method, task_id, "update_cost", "start")
                _refresh_costs(summary_records, output_dir)
                log_step(method, task_id, "update_cost", "done")

                log_step(method, task_id, "update_question_summary", "start")
                write_json(
                    summarize_question_records(summary_records),
                    question_summary_path(output_dir),
                )
                log_step(method, task_id, "update_question_summary", "done")

                if not args.question_only:
                    passrate_records.append(record)
                    log_step(method, task_id, "update_passrate_summary", "start")
                    write_json(
                        summarize_passrate_records(passrate_records, benchmark=args.benchmark),
                        passrate_summary_path(output_dir),
                    )
                    log_step(method, task_id, "update_passrate_summary", "done")
                log_step(method, task_id, "task", "done")

        print(f"[{method}][batch {batch_number}/{batch_count}] done", flush=True)
    summary_path = question_summary_path(output_dir)
    print(f"[{method}][write_question_summary] start", flush=True)
    write_json(summarize_question_records(summary_records), summary_path)
    print(f"[{method}][write_question_summary] done -> {summary_path}", flush=True)
    _refresh_costs(summary_records, output_dir)
    print(f"[{method}][write_cost_summary] done -> {total_cost_path}", flush=True)
    print(f"[{method}][done] results -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
