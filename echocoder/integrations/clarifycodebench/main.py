#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from baselines.common import (
    add_common_args, append_jsonl, client_from_args, completed_task_ids,
    clear_task_error_records, cost_jsonl_path, cost_record_from_result,
    cost_summary_path, initialize_run_manifest, load_jsonl_records,
    load_task_error_records, load_tasks, log_step, passrate_summary_path,
    question_summary_path, remove_task_error_record, reset_jsonl,
    resolve_results_path, summarize_cost_records, update_run_manifest,
    write_json, write_jsonl, write_task_log, write_task_log_from_record,
)
from echocoder.integrations.clarifycodebench.workflow import (
    METHOD,
    process_task,
)
from echocoder.iterative.metrics import summarize_iterative_records
from echocoder.iterative.question_results import (
    clear_task_question_results,
    summarize_task_pass_rate_results,
    summarize_task_question_results,
    sync_task_question_results,
    write_task_question_result,
)


def _refresh_summaries(records: list[dict[str, Any]], output_dir: Path) -> None:
    failures = load_task_error_records(output_dir)
    execution = {
        "completed_task_count": len(records),
        "failed_task_count": len(failures),
        "failed_tasks": [
            {
                "task_id": item.get("task_id"),
                "error_type": item.get("error_type"),
                "error": item.get("error"),
            }
            for item in failures
        ],
    }
    question = summarize_task_question_results(output_dir)
    question["execution"] = execution
    write_json(question, question_summary_path(output_dir))
    passrate = summarize_task_pass_rate_results(output_dir, benchmark="clarifycodebench")
    passrate["execution"] = execution
    write_json(passrate, passrate_summary_path(output_dir))
    iterative = summarize_iterative_records(records)
    iterative.pop("code_correctness", None)
    iterative["execution"] = execution
    write_json(iterative, output_dir / "summary.iterative.json")


def _refresh_costs(records: list[dict[str, Any]], output_dir: Path) -> None:
    sources = [*records, *load_task_error_records(output_dir)]
    costs = [cost_record_from_result(record) for record in sources]
    write_jsonl(costs, cost_jsonl_path(output_dir))
    write_json(summarize_cost_records(costs), cost_summary_path(output_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--reconstruction-budget", "--kr",
        dest="reconstruction_budget", type=int, default=3,
        help="requirement reconstructions from the same code per specification state (K_r)",
    )
    parser.add_argument(
        "--question-budget", "--question-call-budget", "--kq",
        dest="question_call_budget", type=int, default=2,
        help="question-generation LLM calls per reconstruction (K_q); batches are not truncated",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help=(
            "omit separate accepted/rejected question history from LLM prompts; "
            "accepted Q/A still augments the next specification"
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.benchmark != "clarifycodebench":
        parser.error("this adapter supports ClarifyCodeBench only")
    for name in ("batch_size", "reconstruction_budget", "question_call_budget"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.skip_question_metrics:
        parser.error("the clarification oracle cannot be disabled")

    tasks = load_tasks(args.data, args.limit, benchmark="clarifycodebench", task_ids=args.task_ids)
    default_method_dir = "iterative_no_history" if args.no_history else "iterative"
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parents[2] / "output" / default_method_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    results_path = resolve_results_path(args.out, output_dir)
    task_cost_path = cost_jsonl_path(output_dir)
    total_cost_path = cost_summary_path(output_dir)
    judge_model = args.question_judge_model or args.model

    print(f"[{METHOD}][setup] loaded {len(tasks)} task(s)", flush=True)
    print(
        f"[{METHOD}][setup] Kr={args.reconstruction_budget} "
        f"Kq={args.question_call_budget}",
        flush=True,
    )
    existing = [] if args.no_resume else load_jsonl_records(results_path)
    done = completed_task_ids(existing)
    if args.no_resume:
        reset_jsonl(results_path)
        reset_jsonl(task_cost_path)
        clear_task_question_results(output_dir)
        clear_task_error_records(output_dir)
    else:
        for task_id in done:
            remove_task_error_record(output_dir, task_id)

    records = list(existing)
    for record in records:
        write_task_log_from_record(output_dir, record)
    initialize_run_manifest(
        output_dir,
        args=args,
        tasks=tasks,
        method=METHOD,
        benchmark="clarifycodebench",
        prompt_file=Path(__file__).resolve().parents[2] / "prompts.py",
    )
    _refresh_costs(records, output_dir)
    sync_task_question_results(records, output_dir)
    _refresh_summaries(records, output_dir)
    pending = [task for task in tasks if str(task.get("task_id")) not in done]
    client = client_from_args(args) if pending else None
    for task in tasks:
        if str(task.get("task_id")) in done:
            log_step(METHOD, task.get("task_id"), "task", "skipped (already completed)")

    batch_count = (len(pending) + args.batch_size - 1) // args.batch_size
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        number = start // args.batch_size + 1
        print(f"[{METHOD}][batch {number}/{batch_count}] start", flush=True)
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {executor.submit(process_task, task, client=client, args=args, output_dir=output_dir, judge_model=judge_model): task for task in batch}
            for future in as_completed(futures):
                task = futures[future]
                outcome = future.result()
                if outcome is None:
                    _refresh_costs(records, output_dir)
                    _refresh_summaries(records, output_dir)
                    update_run_manifest(output_dir, completed_records=records)
                    continue
                record, log_payload = outcome
                write_task_log(output_dir, task.get("task_id"), log_payload)
                append_jsonl(record, results_path)
                records.append(record)
                remove_task_error_record(output_dir, task.get("task_id"))
                write_task_question_result(record, output_dir)
                _refresh_costs(records, output_dir)
                _refresh_summaries(records, output_dir)
                update_run_manifest(output_dir, completed_records=records)
                log_step(METHOD, task.get("task_id"), "task", "done")
        print(f"[{METHOD}][batch {number}/{batch_count}] done", flush=True)
    update_run_manifest(output_dir, completed_records=records, status="completed")


if __name__ == "__main__":
    main()
