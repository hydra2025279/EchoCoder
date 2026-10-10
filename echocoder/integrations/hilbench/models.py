"""Data models shared by the HiL-Bench adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


TaskType = Literal["sql", "swe"]


@dataclass(frozen=True)
class HilBenchTask:
    task_id: str
    task_type: TaskType
    problem: str
    mode_dir: Path
    shared_dir: Path
    metadata: dict[str, Any]
    qa_pairs: list[dict[str, Any]]
    blocker_registry_path: Path
    database_path: Path | None = None
    business_info: list[str] = field(default_factory=list)
    table_descriptions: list[Path] = field(default_factory=list)
    repository_path: Path | None = None

    @property
    def ground_truth_sql_path(self) -> Path:
        return self.shared_dir / "ground_truth.sql"
