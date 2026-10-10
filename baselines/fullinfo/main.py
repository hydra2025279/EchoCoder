#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
from typing import Any

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
    extract_tag,
    load_jsonl_records,
    load_task_error_records,
    load_tasks,
    log_step,
    llm_call_context,
    method_output_dir,
    passrate_summary_path,
    remove_task_error_record,
    reset_jsonl,
    resolve_results_path,
    set_llm_call_log,
    set_live_text_refresh,
    summarize_cost_records,
    summarize_llm_call_usage,
    summarize_passrate_records,
    write_json,
    write_jsonl,
    write_task_error_log,
    write_task_log,
)
from baselines.fullinfo.prompts import (
    FULL_INFO_CODE_SYSTEM_PROMPT,
    build_full_info_code_user_prompt,
)
from baselines.fullinfo.livecodebench_loader import DEFAULT_LCB_ROOT, _load_livecodebench


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


def _generate_code(
    client: Any,
    *,
    model: str,
    problem: str,
    max_tokens: int,
    max_retries: int,
) -> str:
    from baselines.common import chat_text

    def call(token_budget: int) -> str:
        return chat_text(
            client,
            model=model,
            system_prompt=FULL_INFO_CODE_SYSTEM_PROMPT,
            user_prompt=build_full_info_code_user_prompt(problem=problem),
            temperature=0.0,
            max_tokens=token_budget,
            max_retries=max_retries,
            step_name="generate_code",
        )

    response = call(max_tokens)
    code = extract_tag(response, "CODE") if "[CODE]" in response.upper() else ""
    if not code:
        response = call(max_tokens * 2)
        code = extract_tag(response, "CODE") if "[CODE]" in response.upper() else ""
    if not code:
        raise RuntimeError("Code-generation response did not contain a [CODE] block")
    return code


def _process_task(
    task: dict[str, Any],
    *,
    complete_problem: str,
    client: Any,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    method = "fullinfo"
    task_id = task.get("task_id")
    llm_call_logs: list[dict[str, Any]] = []
    code: str | None = None
    pass_evaluation: dict[str, Any] | None = None
    set_llm_call_log(llm_call_logs, live_output_dir=output_dir, task_id=str(task_id))

    def persist_live() -> None:
        write_task_log(
            output_dir,
            task_id,
            {
                "method": method,
                "benchmark": "clarifycodebench_fullinfo",
                "question_id": task.get("question_id"),
                "input_problem": complete_problem,
                "steps": {"code_id": "C0", "final_code": code},
                "llm_call_logs": llm_call_logs,
                "pass_evaluation": pass_evaluation,
            },
        )

    set_live_text_refresh(persist_live)
    try:
        log_step(method, task_id, "generate_code", "start (complete LiveCodeBench statement)")
        with llm_call_context(phase="code_generation", code_id="C0", specification_state=0):
            code = _generate_code(
                client,
                model=args.model,
                problem=complete_problem,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
            )
        persist_live()
        log_step(method, task_id, "generate_code", "done")

        log_step(method, task_id, "pass_evaluation", "start")
        pass_evaluation = evaluate_pass_rate(
            pass_command=args.pass_command,
            task_id=str(task_id) if task_id is not None else None,
            question_id=task.get("question_id"),
            codes=[code],
        )
        persist_live()
        log_step(method, task_id, "pass_evaluation", "done")
        record = {
            "task_id": task_id,
            "question_id": task.get("question_id"),
            "benchmark": "clarifycodebench_fullinfo",
            "task_type": "programming",
            "method": method,
            "source_problem": "livecodebench.question_content",
            "complete_problem": complete_problem,
            "final_specification": complete_problem,
            "final_code": code,
            "final_solution": code,
            "selected_submission": code,
            "code_states": [
                {
                    "code_id": "C0",
                    "specification": complete_problem,
                    "code": code,
                    "generation_reason": "full_information_direct_generation",
                }
            ],
            "llm_calls": llm_call_logs,
            "token_usage": summarize_llm_call_usage(llm_call_logs),
            "status": "completed",
            "pass_evaluation": pass_evaluation,
        }
        return record, {
            "method": method,
            "benchmark": "clarifycodebench_fullinfo",
            "question_id": task.get("question_id"),
            "input_problem": complete_problem,
            "steps": {"code_id": "C0", "final_code": code},
            "llm_call_logs": llm_call_logs,
            "pass_evaluation": pass_evaluation,
            "record": record,
        }
    except Exception as exc:  # noqa: BLE001
        log_step(method, task_id, "task", f"failed ({type(exc).__name__}: {exc})")
        write_task_error_log(
            output_dir,
            task_id,
            method=method,
            question_id=task.get("question_id"),
            benchmark="clarifycodebench_fullinfo",
            task_type="programming",
            step="task",
            error=exc,
            input_problem=complete_problem,
            qa_pairs=[],
            llm_call_logs=llm_call_logs,
        )
        return None
    finally:
        set_live_text_refresh(None)
        set_llm_call_log(None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, benchmark_choices=("clarifycodebench",))
    parser.add_argument("--release-version", default="v6")
    parser.add_argument("--livecodebench-root", type=Path, default=DEFAULT_LCB_ROOT)
    parser.add_argument("--not-fast", action="store_true")
    args = parser.parse_args()
    if args.question_only:
        parser.error("--question-only is not applicable to the full-information control")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    method = "fullinfo"
    client = client_from_args(args)
    tasks = load_tasks(args.data, args.limit, benchmark=args.benchmark, task_ids=args.task_ids)
    by_question_id = _load_livecodebench(
        lcb_root=args.livecodebench_root.resolve(),
        release_version=args.release_version,
        not_fast=args.not_fast,
    )
    missing = sorted({str(task.get("question_id")) for task in tasks} - set(by_question_id))
    if missing:
        raise ValueError(
            f"{len(missing)} ClarifyCodeBench question_id value(s) are absent from "
            f"LiveCodeBench {args.release_version}: {', '.join(missing[:10])}"
        )

    output_dir = method_output_dir(__file__, args.output_dir, benchmark="clarifycodebench")
    results_path = resolve_results_path(args.out, output_dir)
    print(f"[{method}][setup] loaded {len(tasks)} ClarifyCodeBench task(s)", flush=True)
    print(f"[{method}][setup] source=LiveCodeBench {args.release_version} question_content", flush=True)
    print(f"[{method}][setup] output -> {output_dir}", flush=True)
    existing_records = [] if args.no_resume else load_jsonl_records(results_path)
    done_task_ids = completed_task_ids(existing_records)
    if args.no_resume:
        reset_jsonl(results_path)
        reset_jsonl(cost_jsonl_path(output_dir))
        _clear_task_results(output_dir)
        clear_task_error_records(output_dir)
        write_json(summarize_cost_records([]), cost_summary_path(output_dir))
        write_json(summarize_passrate_records([], benchmark="clarifycodebench"), passrate_summary_path(output_dir))
    else:
        print(f"[{method}][setup] resume -> {len(done_task_ids)} completed task(s)", flush=True)
        for record in existing_records:
            _write_task_result(output_dir, record)
            remove_task_error_record(output_dir, record.get("task_id"))
        _refresh_costs(existing_records, output_dir)
        write_json(summarize_passrate_records(existing_records, benchmark="clarifycodebench"), passrate_summary_path(output_dir))

    records = list(existing_records)
    pending = [task for task in tasks if str(task.get("task_id")) not in done_task_ids]
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(
                    _process_task,
                    task,
                    complete_problem=by_question_id[str(task["question_id"])].question_content,
                    client=client,
                    args=args,
                    output_dir=output_dir,
                ): task
                for task in batch
            }
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    _refresh_costs(records, output_dir)
                    continue
                record, log_payload = result
                task_id = record.get("task_id")
                write_task_log(output_dir, task_id, log_payload)
                append_jsonl(record, results_path)
                _write_task_result(output_dir, record)
                records.append(record)
                _refresh_costs(records, output_dir)
                write_json(summarize_passrate_records(records, benchmark="clarifycodebench"), passrate_summary_path(output_dir))
                log_step(method, task_id, "task", "done")

    _refresh_costs(records, output_dir)
    write_json(summarize_passrate_records(records, benchmark="clarifycodebench"), passrate_summary_path(output_dir))
    print(f"[{method}][done] results -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
