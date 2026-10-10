from __future__ import annotations

import importlib
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

from .models import HilBenchTask


def _rows(db_path: Path, query: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        cursor = connection.execute(query)
        if cursor.description is None:
            raise ValueError("Submission must be a read-only query that returns rows")
        columns = [str(item[0]) for item in cursor.description]
        return columns, cursor.fetchall()


def evaluate_sql(task: HilBenchTask, query: str) -> dict[str, Any]:
    if task.database_path is None:
        return {"enabled": False, "passed": None, "reason": "SQL database is not downloaded"}
    if not task.ground_truth_sql_path.exists():
        return {"enabled": False, "passed": None, "reason": "ground_truth.sql is unavailable"}
    try:
        try:
            evaluator = _hil_module("hil_bench.utils.calculate_sql_pass_at_1")
        except ModuleNotFoundError as exc:
            if exc.name != "pandas":
                raise
            expected_columns, expected_rows = _rows(
                task.database_path,
                task.ground_truth_sql_path.read_text(encoding="utf-8"),
            )
            actual_columns, actual_rows = _rows(task.database_path, query)
            passed = (
                len(expected_columns) == len(actual_columns)
                and sorted(map(repr, expected_rows)) == sorted(map(repr, actual_rows))
            )
            details = {
                "status": "correct" if passed else "incorrect",
                "evaluator": "stdlib-compatible-fallback",
                "reason": "pandas is unavailable for the official HiL-Bench comparator",
            }
            return {
                "enabled": True,
                "passed": int(passed),
                "total": 1,
                "pass_rate": float(passed),
                "status": "pass" if passed else "fail",
                "results": [details],
                "details": details,
            }
        instance = {
            "env": {"base_db_path": str(task.database_path)},
            "problem_statement": {
                "ground_truth_query": task.ground_truth_sql_path.read_text(encoding="utf-8"),
                "expected_sorted": False,
                "required_unique": False,
                "no_extra_columns_allowed": False,
            },
        }
        _, details, passed = evaluator.evaluate_single_sql_instance(
            task.task_id,
            {"model_patch": query},
            instance,
            task.shared_dir,
        )
        return {
            "enabled": True,
            "passed": int(passed),
            "total": 1,
            "pass_rate": float(passed),
            "status": "pass" if passed else "fail",
            "results": [details],
            "details": details,
        }
    except Exception as exc:  # noqa: BLE001
        return {"enabled": True, "passed": 0, "total": 1, "pass_rate": 0.0, "status": "error", "results": [], "error": f"{type(exc).__name__}: {exc}"}


def _hil_module(name: str):
    benchmark_root = Path(__file__).resolve().parents[3] / "benchmarks" / "hil-bench"
    if str(benchmark_root) not in sys.path:
        sys.path.insert(0, str(benchmark_root))
    return importlib.import_module(name)


def evaluate_swe(task: HilBenchTask, patch: str, *, timeout: int = 1800) -> dict[str, Any]:
    if "diff --git " not in patch:
        return {
            "enabled": True,
            "passed": 0,
            "total": 1,
            "pass_rate": 0.0,
            "status": "error",
            "results": [],
            "reason": "submission is not a git-compatible unified diff",
        }
    metadata_path = task.shared_dir / "metadata.json"
    if not metadata_path.exists():
        return {"enabled": False, "passed": None, "reason": "SWE metadata is unavailable"}
    # SWE-agent submissions from older runs may have lost the diff's final newline.
    # The harness applies the patch with `patch`, which rejects such a final hunk.
    patch = patch if patch.endswith("\n") else patch + "\n"
    with tempfile.TemporaryDirectory(prefix=f"echocoder_eval_{task.task_id}_") as temporary:
        tasks_root = Path(temporary)
        evaluation_task = tasks_root / task.task_id
        evaluation_task.mkdir()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        (evaluation_task / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        evaluator = _hil_module("hil_bench.utils.custom_eval")
        report = evaluator.evaluate_custom_instances(
            {task.task_id: {"model_patch": patch, "model_name_or_path": "echocoder"}},
            tasks_root,
            run_id=f"echocoder-{task.task_id}",
            timeout=timeout,
            max_workers=1,
        )
    passed = task.task_id in report.get("resolved_ids", [])
    details = report.get("results", {}).get(task.task_id, {})
    return {
        "enabled": True,
        "passed": int(passed),
        "total": 1,
        "pass_rate": float(passed),
        "status": "pass" if passed else "fail",
        "results": [details],
        "details": details,
    }


def evaluate_final(task: HilBenchTask, candidate: str, *, timeout: int = 1800) -> dict[str, Any]:
    return evaluate_sql(task, candidate) if task.task_type == "sql" else evaluate_swe(task, candidate, timeout=timeout)
