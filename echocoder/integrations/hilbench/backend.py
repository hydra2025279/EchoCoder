from __future__ import annotations

from pathlib import Path
from typing import Any

from echocoder.integrations.base import Mode

from .models import HilBenchTask
from .shared.backend import HilBenchBackend, augment_specification
from .sql.backend import SqlBackend
from .sweagent.backend import SweAgentBackend


def create_backend(
    task: HilBenchTask,
    *,
    mode: Mode,
    sweagent_artifact_root: Path,
    sweagent_config_path: Path,
    sweagent_model: str | None = None,
    sweagent_max_steps: int,
    sweagent_timeout: int,
    sweagent_cost_limit: float,
    sweagent_call_limit: int,
    progressive_edit: bool = False,
    **kwargs: Any,
) -> HilBenchBackend:
    if task.task_type == "sql":
        for key in (
            "embedding_persist_dir",
            "final_location_samples",
            "final_samples_per_location",
            "reproduction_samples",
        ):
            kwargs.pop(key, None)
        return SqlBackend(task, **kwargs)
    if mode == Mode.ECHOCODER_SWEAGENT:
        return SweAgentBackend(
            task,
            artifact_root=sweagent_artifact_root,
            sweagent_config_path=sweagent_config_path,
            sweagent_model=sweagent_model,
            sweagent_max_steps=sweagent_max_steps,
            sweagent_timeout=sweagent_timeout,
            sweagent_cost_limit=sweagent_cost_limit,
            sweagent_call_limit=sweagent_call_limit,
            progressive_edit=progressive_edit,
            **kwargs,
        )
    raise ValueError(f"Unsupported HiL-Bench mode: {mode}")


__all__ = [
    "HilBenchBackend",
    "SqlBackend",
    "SweAgentBackend",
    "augment_specification",
    "create_backend",
]
