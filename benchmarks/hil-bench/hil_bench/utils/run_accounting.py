from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .token_cost import combine_usage, price_usage


def attach_pipeline_usage(
    metrics: dict[str, Any], model_name: str, hil_metrics: dict[str, Any] | None
) -> None:
    hil_instances = (hil_metrics or {}).get("instances", {})
    for instance_id, instance_metrics in metrics.get("instances", {}).items():
        agent_instance_usage = price_usage(
            model_name,
            int(instance_metrics.get("input_tokens") or 0),
            int(instance_metrics.get("output_tokens") or 0),
        )
        judge_instance_usage = (hil_instances.get(instance_id) or {}).get("token_usage")
        pipeline_instance_usage = combine_usage(agent_instance_usage, judge_instance_usage)
        instance_metrics["provider_reported_agent_cost"] = instance_metrics.get("cost")
        instance_metrics["token_usage"] = {
            "agent": agent_instance_usage,
            "ask_human_judge": judge_instance_usage,
            "pipeline": pipeline_instance_usage,
        }
        instance_metrics["cost"] = pipeline_instance_usage.get("total_cost")

    provider_reported_cost = metrics.get("total_cost")
    agent_usage = price_usage(
        model_name,
        int(metrics.get("input_tokens") or 0),
        int(metrics.get("output_tokens") or 0),
    )
    judge_usage = (hil_metrics or {}).get("token_usage")
    pipeline_usage = combine_usage(agent_usage, judge_usage)
    metrics["provider_reported_agent_cost"] = provider_reported_cost
    metrics["token_usage"] = {
        "agent": agent_usage,
        "ask_human_judge": judge_usage,
        "pipeline": pipeline_usage,
    }
    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_cost",
        "output_cost",
        "total_cost",
        "cost_available",
    ):
        metrics[key] = pipeline_usage.get(key)


def prepare_task_resume(results_dir: Path) -> tuple[dict[str, Any], int]:
    prior_logs: dict[str, Any] = {}
    for logs_path in results_dir.glob("**/ask_human_logs.json"):
        try:
            payload = json.loads(logs_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            prior_logs.update(payload)

    resumed_ids: set[str] = set()
    for trajectory_path in results_dir.glob("**/*.traj"):
        instance_dir = trajectory_path.parent
        instance_id = trajectory_path.stem
        if instance_dir.name != instance_id or instance_dir.parent == results_dir:
            continue
        try:
            payload = json.loads(trajectory_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        exit_status = payload.get("info", {}).get("exit_status")
        if exit_status in (None, "early_exit"):
            continue
        if "__ask_human" in instance_id:
            trajectory = payload.get("trajectory", [])
            question_calls = sum(
                1
                for step in trajectory
                if isinstance(step, dict)
                and isinstance(step.get("action"), str)
                and step["action"].lstrip().startswith("ask_human ")
            )
            logged_questions = (prior_logs.get(instance_id) or {}).get("questions", [])
            if question_calls and (
                len(logged_questions) != question_calls
                or any("judge_usage" not in question for question in logged_questions)
            ):
                continue
        destination = results_dir / instance_id
        if not destination.exists():
            shutil.copytree(instance_dir, destination)
        resumed_ids.add(instance_id)

    return {key: value for key, value in prior_logs.items() if key in resumed_ids}, len(resumed_ids)


def _run_dir_activity(path: Path) -> float:
    activity = path.stat().st_mtime
    markers = [
        path / "run_batch_exit_statuses.yaml",
        path / "consolidated_metrics.json",
        path / "consolidated_results.json",
    ]
    markers.extend(path.glob("**/*.traj"))
    for marker in markers:
        try:
            activity = max(activity, marker.stat().st_mtime)
        except OSError:
            continue
    return activity


def resolve_run_output_dir(output_dir: Path, run_name: str) -> Path:
    canonical = output_dir / run_name
    candidates = [
        path
        for path in output_dir.glob(f"{run_name}*")
        if path.is_dir() and "__rerun_" not in path.name
    ]
    if not candidates:
        return canonical
    return max(candidates, key=_run_dir_activity)
