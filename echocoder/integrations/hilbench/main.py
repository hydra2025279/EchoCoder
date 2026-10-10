#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from baselines.common import (
    add_common_args, append_jsonl, client_from_args, completed_task_ids,
    clear_task_error_records, cost_jsonl_path, cost_record_from_result,
    cost_summary_path, initialize_run_manifest, load_jsonl_records,
    load_task_error_records, log_step, remove_task_error_record, reset_jsonl,
    resolve_results_path, summarize_cost_records, update_run_manifest,
    write_json, write_jsonl,
    write_task_log, write_task_log_from_record,
)
from echocoder.integrations.base import Mode
from echocoder.iterative.metrics import summarize_iterative_records
from echocoder.iterative.question_results import (
    clear_task_question_results,
    summarize_task_pass_rate_results,
    summarize_task_question_results,
    sync_task_question_results,
    write_task_question_result,
)

from .task_loader import (
    ASSET_ROOT_ENV,
    download_sql_assets,
    load_hilbench_tasks,
    resolve_asset_root,
)
from .image_lifecycle import (
    build_cache_records, cleanup_batch_images, cleanup_build_cache,
    collect_swerex_runtime_images, load_batch_images, load_cache_cleanup_ledger,
    load_cleanup_ledger, snapshot_swerex_artifacts, write_cleanup_ledger,
)
from .sweagent.adapter import DEFAULT_CONFIG as DEFAULT_SWEAGENT_CONFIG
from .workflow import method_for_mode, process_task


DEFAULT_DATA = Path(__file__).resolve().parents[3] / "benchmarks" / "hil-bench"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(max_tokens=None, benchmark="hil-bench")
    parser.add_argument("--task-type", required=True, choices=("sql", "swe"))
    parser.add_argument("--task-parallel", type=int, default=1, help="maximum HiL-Bench tasks processed concurrently (default: 1)")
    parser.add_argument(
        "--mode",
        choices=(Mode.ECHOCODER_SWEAGENT.value,),
        default=Mode.ECHOCODER_SWEAGENT.value,
    )
    parser.add_argument(
        "--reconstruction-budget", "--kr",
        dest="reconstruction_budget", type=int, default=3,
    )
    parser.add_argument(
        "--question-budget", "--question-call-budget", "--kq",
        dest="question_call_budget", type=int, default=2,
    )
    parser.add_argument("--repository-root", help="SWE checkout, or a directory containing one clean checkout per task")
    parser.add_argument(
        "--asset-root",
        help=f"persistent SQL/SWE asset directory; defaults to ${ASSET_ROOT_ENV}",
    )
    parser.add_argument("--download-assets", action="store_true", help="download missing SQL artifacts; SWE images are loaded automatically")
    parser.add_argument("--eval-timeout", type=int, default=1800)
    parser.add_argument(
        "--evaluate-each-code-state", action="store_true",
        help="after evaluating the selected submission, evaluate C0, C1, ... without evaluating the original code",
    )
    parser.add_argument(
        "--sweagent-config",
        default=str(DEFAULT_SWEAGENT_CONFIG),
        help="SWE-agent YAML whose upstream agent prompts and tools are preserved",
    )
    parser.add_argument("--sweagent-model", help="LiteLLM model name for SWE-agent; defaults to --model")
    parser.add_argument("--sweagent-max-steps", type=int, default=200)
    parser.add_argument("--sweagent-timeout", type=int, default=21600)
    parser.add_argument("--sweagent-cost-limit", type=float, default=100.0)
    parser.add_argument("--sweagent-call-limit", type=int, default=0)
    parser.add_argument("--progressive-edit", action="store_true", help="continue each SWE-agent patch from the previous candidate after accepted clarification")
    return parser


def _refresh(records: list[dict[str, Any]], output_dir: Path) -> None:
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
    write_json(question, output_dir / "summary.question.json")
    passrate = summarize_task_pass_rate_results(output_dir)
    passrate["execution"] = execution
    write_json(passrate, output_dir / "summary.passrate.json")
    iterative = summarize_iterative_records(records)
    iterative["execution"] = execution
    write_json(iterative, output_dir / "summary.iterative.json")


def _refresh_costs(records: list[dict[str, Any]], output_dir: Path) -> None:
    sources = [*records, *load_task_error_records(output_dir)]
    costs = [cost_record_from_result(record) for record in sources]
    write_jsonl(costs, cost_jsonl_path(output_dir))
    write_json(summarize_cost_records(costs), cost_summary_path(output_dir))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.benchmark != "hil-bench":
        parser.error("this adapter supports HiL-Bench only")
    if args.max_tokens is not None:
        parser.error("HiL-Bench runs do not set a max output token limit")
    if args.skip_question_metrics:
        parser.error("the HiL-Bench clarification oracle cannot be disabled")
    if args.progressive_edit and (args.task_type != "swe" or args.mode != Mode.ECHOCODER_SWEAGENT.value):
        parser.error("--progressive-edit requires --task-type swe --mode echocoder_sweagent")
    data = Path(args.data).resolve() if args.data else DEFAULT_DATA
    asset_root = resolve_asset_root(args.asset_root)
    for name in (
        "batch_size", "task_parallel", "reconstruction_budget", "question_call_budget", "eval_timeout",
        "sweagent_max_steps", "sweagent_timeout",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    tasks = load_hilbench_tasks(
        data,
        task_type=args.task_type,
        task_ids=args.task_ids,
        limit=args.limit,
        repository_root=args.repository_root,
        asset_root=asset_root,
    )
    if args.download_assets and args.task_type == "sql":
        for task in tasks:
            if task.task_type == "sql":
                download_sql_assets(task, asset_root=asset_root)
        tasks = load_hilbench_tasks(
            data,
            task_type=args.task_type,
            task_ids=args.task_ids,
            limit=args.limit,
            repository_root=args.repository_root,
            asset_root=asset_root,
        )
    client = client_from_args(args)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parents[2] / "output" / (f"hilbench_{args.task_type}_progressive" if args.progressive_edit else f"hilbench_{args.task_type}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    cleanup_ledger = output_dir / "image_cleanup.json"
    stale_images = load_cleanup_ledger(cleanup_ledger)
    stale_cache = load_cache_cleanup_ledger(cleanup_ledger)
    if stale_images or stale_cache:
        image_failures = cleanup_batch_images(
            stale_images,
            runtime_ids={name for name, owned_id in stale_images.items() if name == owned_id},
        )
        cache_failures = cleanup_build_cache(stale_cache)
        write_cleanup_ledger(cleanup_ledger, image_failures, cache_failures)
        if image_failures or cache_failures:
            raise RuntimeError(
                "Could not clean Docker artifacts from the previous run: "
                f"images={image_failures}, build_cache={sorted(cache_failures)}"
            )
    results_path = resolve_results_path(args.out, output_dir)
    costs_path = cost_jsonl_path(output_dir)
    existing = [] if args.no_resume else load_jsonl_records(results_path)
    if args.no_resume:
        reset_jsonl(results_path)
        reset_jsonl(costs_path)
        clear_task_question_results(output_dir)
        clear_task_error_records(output_dir)
    done = completed_task_ids(existing)
    if not args.no_resume:
        for task_id in done:
            remove_task_error_record(output_dir, task_id)
    records = list(existing)
    for record in records:
        write_task_log_from_record(output_dir, record)
    method = method_for_mode(args.mode)
    initialize_run_manifest(
        output_dir,
        args=args,
        tasks=tasks,
        method=method,
        benchmark="hil-bench",
        prompt_file=Path(__file__).resolve().parents[2] / "prompts.py",
    )
    _refresh_costs(records, output_dir)
    sync_task_question_results(records, output_dir)
    _refresh(records, output_dir)
    pending = [task for task in tasks if task.task_id not in done]
    print(f"[{method}][setup] loaded={len(tasks)} pending={len(pending)} type={args.task_type}", flush=True)
    for task in tasks:
        if task.task_id in done:
            log_step(method, task.task_id, "task", "skipped (already completed)")
    batch_width = (
        args.task_parallel if args.task_type == "swe"
        else max(args.batch_size, args.task_parallel)
    )
    for start in range(0, len(pending), batch_width):
        batch = pending[start:start + batch_width]
        owned_images: dict[str, str] = {}
        track_swerex = args.task_type == "swe" and args.mode == Mode.ECHOCODER_SWEAGENT.value
        if track_swerex:
            print("[echocoder_hilbench][images] checking Docker image/cache state", flush=True)
        snapshot_started = time.perf_counter()
        before_image_ids, before_cache_ids = (
            snapshot_swerex_artifacts() if track_swerex else (set(), set())
        )
        if track_swerex:
            print(
                f"[echocoder_hilbench][images] Docker state ready in "
                f"{time.perf_counter() - snapshot_started:.1f}s",
                flush=True,
            )
        try:
            if args.task_type == "swe":
                load_batch_images(
                    batch, asset_root, owned_images,
                    on_owned=lambda: write_cleanup_ledger(cleanup_ledger, owned_images),
                    max_workers=args.task_parallel,
                )
            with ThreadPoolExecutor(max_workers=min(args.task_parallel, len(batch))) as executor:
                futures = {executor.submit(process_task, task, client=client, args=args, output_dir=output_dir): task for task in batch}
                for future in as_completed(futures):
                    task = futures[future]
                    outcome = future.result()
                    if outcome is None:
                        _refresh_costs(records, output_dir)
                        _refresh(records, output_dir)
                        update_run_manifest(output_dir, completed_records=records)
                        continue
                    record, payload = outcome
                    write_task_log(output_dir, task.task_id, payload)
                    append_jsonl(record, results_path)
                    records.append(record)
                    remove_task_error_record(output_dir, task.task_id)
                    write_task_question_result(record, output_dir)
                    _refresh_costs(records, output_dir)
                    _refresh(records, output_dir)
                    update_run_manifest(output_dir, completed_records=records)
                    log_step(method, task.task_id, "task", "done")
        finally:
            runtime_error = None
            cache_error = None
            cache_ids: set[str] = set()
            runtime_ids: set[str] = set()
            if track_swerex:
                try:
                    runtime_images = collect_swerex_runtime_images(batch, before_image_ids)
                    runtime_ids = set(runtime_images)
                    owned_images.update(runtime_images)
                except Exception as exc:
                    runtime_error = exc
                try:
                    cache_ids = set(build_cache_records()) - before_cache_ids
                except Exception as exc:
                    cache_error = exc
            write_cleanup_ledger(cleanup_ledger, owned_images, cache_ids)
            failures = cleanup_batch_images(owned_images, runtime_ids=runtime_ids)
            cache_failures = cleanup_build_cache(cache_ids) if cache_ids else set()
            write_cleanup_ledger(cleanup_ledger, failures, cache_failures)
            if failures or cache_failures:
                raise RuntimeError(
                    f"Could not clean batch Docker artifacts: images={failures}, "
                    f"build_cache={sorted(cache_failures)}"
                )
            if runtime_error is not None:
                raise RuntimeError("Could not inspect SWE-agent runtime images") from runtime_error
            if cache_error is not None:
                raise RuntimeError("Could not inspect SWE-agent build cache") from cache_error
    update_run_manifest(output_dir, completed_records=records, status="completed")


if __name__ == "__main__":
    main()
