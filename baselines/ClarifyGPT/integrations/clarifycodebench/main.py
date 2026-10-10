#!/usr/bin/env python3
"""Run the ClarifyGPT paper workflow on ClarifyCodeBench."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
from typing import Any

from baselines.common import (
    AzureResponsesClient,
    add_common_args,
    append_jsonl,
    clear_task_error_records,
    client_from_args,
    completed_task_ids,
    cost_jsonl_path,
    cost_record_from_result,
    cost_summary_path,
    evaluate_pass_rate,
    load_jsonl_records,
    load_task_error_records,
    load_tasks,
    log_step,
    method_output_dir,
    passrate_summary_path,
    question_summary_path,
    remove_task_error_record,
    reset_jsonl,
    resolve_results_path,
    set_live_text_refresh,
    set_llm_call_log,
    skipped_pass_evaluation,
    summarize_cost_records,
    summarize_llm_call_usage,
    summarize_passrate_records,
    write_json,
    write_jsonl,
    write_task_error_log,
    write_task_log,
)
from baselines.ClarifyGPT.integrations.clarifycodebench.workflow import run_clarifygpt


METHOD = "clarifygpt"


def _result_path(output_dir: Path, task_id: Any) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id or "unknown_task"))
    return output_dir / "result" / f"{safe}.json"


def _write_result(output_dir: Path, record: dict[str, Any]) -> None:
    write_json(record, _result_path(output_dir, record.get("task_id")))


def _clear_results(output_dir: Path) -> None:
    directory = output_dir / "result"
    if directory.exists():
        for path in directory.glob("*.json"):
            path.unlink()


def _refresh_costs(records: list[dict[str, Any]], output_dir: Path) -> None:
    costs = [cost_record_from_result(item) for item in [*records, *load_task_error_records(output_dir)]]
    write_jsonl(costs, cost_jsonl_path(output_dir))
    write_json(summarize_cost_records(costs), cost_summary_path(output_dir))


def _question_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    from baselines.common import summarize_question_records

    summary = summarize_question_records(records)
    scored = [item["question_metrics"] for item in records if isinstance(item.get("question_metrics"), dict)]
    if scored:
        summary["TKQR"] = sum(float(item.get("TKQR", 0.0)) for item in scored) / len(scored)
        summary["ORA"] = sum(float(item.get("ORA", 0.0)) for item in scored) / len(scored)
    return summary


def _process_task(
    task: dict[str, Any], *, client: Any, args: argparse.Namespace, output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    task_id = task.get("task_id")
    problem = str(task["modified_content"])
    qa_pairs = task.get("qa_pairs", []) or []
    calls: list[dict[str, Any]] = []
    state: dict[str, Any] = {"stages": {}, "pass_evaluation": None}
    set_llm_call_log(calls, live_output_dir=output_dir, task_id=str(task_id))

    def persist() -> None:
        write_task_log(
            output_dir,
            task_id,
            {
                "method": METHOD,
                "benchmark": task.get("benchmark", args.benchmark),
                "task_type": task.get("task_type"),
                "question_id": task.get("question_id"),
                "input_problem": problem,
                "qa_pairs": qa_pairs,
                "steps": state["stages"],
                "generated_questions": state["stages"].get("question_generation", {}).get("generated_questions", []),
                "llm_call_logs": calls,
                "question_metrics": state["stages"].get("oracle", {}).get("question_metrics"),
                "pass_evaluation": state["pass_evaluation"],
            },
        )

    set_live_text_refresh(persist)
    try:
        def on_stage(name: str, payload: dict[str, Any]) -> None:
            state["stages"][name] = payload.copy()
            persist()

        log_step(METHOD, task_id, "clarifygpt", "start")
        workflow = run_clarifygpt(
            task,
            client=client,
            model=args.model,
            candidate_count=args.candidate_count,
            seed_input_count=args.seed_input_count,
            test_input_count=args.test_input_count,
            timeout_seconds=args.candidate_timeout,
            demonstrations=args.demonstrations,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            question_consistency=args.question_consistency,
            question_judge_model=args.question_judge_model or args.model,
            skip_question_metrics=args.skip_question_metrics,
            question_only=args.question_only,
            representative_seed=args.representative_seed,
            input_mutation_seed=args.input_mutation_seed,
            on_stage=on_stage,
        )
        log_step(METHOD, task_id, "clarifygpt", f"done ({workflow['stop_reason']})")
        if args.question_only:
            evaluation = skipped_pass_evaluation("--question-only was set")
        else:
            log_step(METHOD, task_id, "pass_evaluation", "start")
            evaluation = evaluate_pass_rate(
                pass_command=args.pass_command,
                task_id=str(task_id) if task_id is not None else None,
                question_id=task.get("question_id"),
                codes=[workflow["final_code"]],
            )
            log_step(METHOD, task_id, "pass_evaluation", "done")
        state["pass_evaluation"] = evaluation
        persist()
        record = {
            "task_id": task_id,
            "question_id": task.get("question_id"),
            "benchmark": task.get("benchmark", args.benchmark),
            "task_type": task.get("task_type") or "programming",
            "method": METHOD,
            "workflow_version": "clarifygpt_paper_clarifycodebench_v1",
            "settings": {
                "candidate_count": args.candidate_count,
                "candidate_temperature_requested": 0.8,
                "candidate_temperature_effective": (
                    None if isinstance(client, AzureResponsesClient) else 0.8
                ),
                "seed_input_count": args.seed_input_count,
                "test_input_count": args.test_input_count,
                "candidate_timeout_seconds": args.candidate_timeout,
                "demonstrations": args.demonstrations,
                "representative_seed": args.representative_seed,
                "input_mutation_seed": args.input_mutation_seed,
            },
            "original_problem": problem,
            **workflow,
            "final_solution": workflow["final_code"],
            "selected_submission": workflow["final_code"],
            "code_states": ([] if workflow["final_code"] is None else [{
                "code_id": "C0",
                "code": workflow["final_code"],
                "generation_reason": workflow["stop_reason"],
            }]),
            "llm_calls": calls,
            "token_usage": summarize_llm_call_usage(calls),
            "status": "completed",
            "pass_evaluation": evaluation,
        }
        payload = {
            "method": METHOD,
            "benchmark": record["benchmark"],
            "task_type": record["task_type"],
            "question_id": record["question_id"],
            "input_problem": problem,
            "qa_pairs": qa_pairs,
            "steps": state["stages"],
            "generated_questions": workflow["generated_questions"],
            "llm_call_logs": calls,
            "question_metrics": workflow["question_metrics"],
            "pass_evaluation": evaluation,
            "record": record,
        }
        return record, payload
    except Exception as error:  # noqa: BLE001
        log_step(METHOD, task_id, "task", f"failed ({type(error).__name__}: {error})")
        write_task_error_log(
            output_dir,
            task_id,
            method=METHOD,
            question_id=task.get("question_id"),
            benchmark=task.get("benchmark", args.benchmark),
            task_type=task.get("task_type"),
            step="task",
            error=error,
            input_problem=problem,
            qa_pairs=qa_pairs,
            llm_call_logs=calls,
        )
        return None
    finally:
        set_live_text_refresh(None)
        set_llm_call_log(None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, benchmark_choices=("clarifycodebench",))
    parser.add_argument("--candidate-count", type=int, default=25)
    parser.add_argument("--seed-input-count", type=int, default=5)
    parser.add_argument("--test-input-count", type=int, default=20)
    parser.add_argument("--candidate-timeout", type=float, default=2.0)
    parser.add_argument("--demonstrations", type=int, choices=range(4), default=3)
    parser.add_argument("--representative-seed", type=int, default=0)
    parser.add_argument("--input-mutation-seed", type=int, default=0)
    args = parser.parse_args()
    for name in ("batch_size", "candidate_count", "seed_input_count", "test_input_count"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")

    client = client_from_args(args)
    tasks = load_tasks(args.data, args.limit, benchmark=args.benchmark, task_ids=args.task_ids)
    output_dir = method_output_dir(__file__, args.output_dir, benchmark=args.benchmark)
    results_path = resolve_results_path(args.out, output_dir)
    existing = [] if args.no_resume else load_jsonl_records(results_path)
    if args.no_resume:
        reset_jsonl(results_path)
        reset_jsonl(cost_jsonl_path(output_dir))
        _clear_results(output_dir)
        clear_task_error_records(output_dir)
    for record in existing:
        _write_result(output_dir, record)
        remove_task_error_record(output_dir, record.get("task_id"))
    records = list(existing)
    done = completed_task_ids(records)
    _refresh_costs(records, output_dir)
    write_json(_question_summary(records), question_summary_path(output_dir))
    write_json(summarize_passrate_records(records, benchmark="clarifycodebench"), passrate_summary_path(output_dir))
    pending = [task for task in tasks if str(task.get("task_id")) not in done]
    print(f"[{METHOD}][setup] loaded={len(tasks)} pending={len(pending)} candidates={args.candidate_count}", flush=True)

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {executor.submit(_process_task, task, client=client, args=args, output_dir=output_dir): task for task in batch}
            for future in as_completed(futures):
                task = futures[future]
                outcome = future.result()
                if outcome is None:
                    _refresh_costs(records, output_dir)
                    continue
                record, payload = outcome
                _write_result(output_dir, record)
                write_task_log(output_dir, record.get("task_id"), payload)
                append_jsonl(record, results_path)
                records.append(record)
                _refresh_costs(records, output_dir)
                write_json(_question_summary(records), question_summary_path(output_dir))
                write_json(summarize_passrate_records(records, benchmark="clarifycodebench"), passrate_summary_path(output_dir))
                log_step(METHOD, task.get("task_id"), "task", "done")
    _refresh_costs(records, output_dir)
    write_json(_question_summary(records), question_summary_path(output_dir))
    write_json(summarize_passrate_records(records, benchmark="clarifycodebench"), passrate_summary_path(output_dir))
    print(f"[{METHOD}][done] results -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
