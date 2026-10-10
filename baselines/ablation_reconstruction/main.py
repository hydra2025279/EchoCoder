#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from baselines.ablation_reconstruction.artifacts import (
    clear_task_results,
    summarize_passrates,
    summarize_iterations,
    summarize_questions,
    sync_task_results,
    write_task_result,
)
from baselines.ablation_reconstruction.workflow import METHOD, process_task
from baselines.common import (
    add_common_args,
    append_jsonl,
    clear_task_error_records,
    client_from_args,
    completed_task_ids,
    cost_jsonl_path,
    cost_record_from_result,
    cost_summary_path,
    initialize_run_manifest,
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
    summarize_cost_records,
    update_run_manifest,
    write_json,
    write_jsonl,
)


def _refresh(records: list[dict], output_dir: Path) -> None:
    failures = load_task_error_records(output_dir)
    execution = {
        "completed_task_count": len(records),
        "failed_task_count": len(failures),
        "failed_tasks": [
            {"task_id": item.get("task_id"), "error": item.get("error")}
            for item in failures
        ],
    }
    question = summarize_questions(records)
    question["execution"] = execution
    write_json(question, question_summary_path(output_dir))
    passrate = summarize_passrates(records)
    passrate["execution"] = execution
    write_json(passrate, passrate_summary_path(output_dir))
    iterative = summarize_iterations(records)
    iterative["execution"] = execution
    write_json(iterative, output_dir / "summary.iterative.json")
    sources = [*records, *failures]
    costs = [cost_record_from_result(record) for record in sources]
    write_jsonl(costs, cost_jsonl_path(output_dir))
    write_json(summarize_cost_records(costs), cost_summary_path(output_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, benchmark_choices=("clarifycodebench",))
    parser.set_defaults(max_tokens=8192)
    parser.add_argument(
        "--iteration-budget", "--budget", type=int, default=6,
        help="maximum independent question calls from one generated code per task",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.iteration_budget < 1:
        raise SystemExit("--iteration-budget must be at least 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.skip_question_metrics:
        raise SystemExit("the ClarifyCodeBench oracle cannot be disabled")

    client = client_from_args(args)
    tasks = load_tasks(
        args.data,
        args.limit,
        benchmark="clarifycodebench",
        task_ids=args.task_ids,
    )
    output_dir = method_output_dir(__file__, args.output_dir)
    results_path = resolve_results_path(args.out, output_dir)
    judge_model = args.question_judge_model or args.model
    existing = [] if args.no_resume else load_jsonl_records(results_path)
    done = completed_task_ids(existing)
    if args.no_resume:
        reset_jsonl(results_path)
        clear_task_results(output_dir)
        clear_task_error_records(output_dir)
    else:
        for task_id in done:
            remove_task_error_record(output_dir, task_id)

    records = list(existing)
    manifest = initialize_run_manifest(
        output_dir,
        args=args,
        tasks=tasks,
        method=METHOD,
        benchmark="clarifycodebench",
        prompt_file=Path(__file__).with_name("prompts.py"),
    )
    manifest["prompt"]["version"] = "ablation_reconstruction_v3"
    manifest["workflow"] = {
        "reconstruction_enabled": False,
        "iteration_budget": args.iteration_budget,
        "code_generation_count": 2,
        "independent_question_calls": True,
        "question_history_used_for_question_generation": False,
        "accepted_clarifications_used_for_final_code": True,
    }
    write_json(manifest, output_dir / "run.json")
    sync_task_results(records, output_dir)
    _refresh(records, output_dir)
    update_run_manifest(output_dir, completed_records=records)
    pending = [task for task in tasks if str(task.get("task_id")) not in done]
    print(
        f"[{METHOD}][setup] loaded={len(tasks)} pending={len(pending)} "
        f"budget={args.iteration_budget} output={output_dir}",
        flush=True,
    )

    batch_count = (len(pending) + args.batch_size - 1) // args.batch_size
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        batch_number = start // args.batch_size + 1
        print(f"[{METHOD}][batch {batch_number}/{batch_count}] start", flush=True)
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(
                    process_task,
                    task,
                    client=client,
                    args=args,
                    output_dir=output_dir,
                    judge_model=judge_model,
                ): task
                for task in batch
            }
            for future in as_completed(futures):
                task = futures[future]
                outcome = future.result()
                if outcome is None:
                    _refresh(records, output_dir)
                    update_run_manifest(output_dir, completed_records=records)
                    continue
                record, _ = outcome
                append_jsonl(record, results_path)
                records.append(record)
                done.add(str(task.get("task_id")))
                remove_task_error_record(output_dir, task.get("task_id"))
                write_task_result(record, output_dir)
                _refresh(records, output_dir)
                update_run_manifest(output_dir, completed_records=records)
                log_step(METHOD, task.get("task_id"), "task", "done")
        print(f"[{METHOD}][batch {batch_number}/{batch_count}] done", flush=True)

    update_run_manifest(output_dir, completed_records=records, status="completed")
    print(f"[{METHOD}][done] results -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
