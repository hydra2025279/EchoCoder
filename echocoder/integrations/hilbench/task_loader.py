

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

from baselines.common import load_tasks

from .models import HilBenchTask, TaskType


ASSET_ROOT_ENV = "HIL_BENCH_ASSET_ROOT"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_asset_root(asset_root: str | Path | None = None) -> Path | None:
    """Resolve the optional persistent HiL-Bench asset directory."""
    value = asset_root if asset_root is not None else os.environ.get(ASSET_ROOT_ENV)
    if value is None or not str(value).strip():
        return None
    return Path(value).expanduser().resolve()


def _download_hf_bucket(repo_id: str, filename: str, destination: Path) -> Path:
    """Stream a private HF bucket object without buffering it in memory."""
    try:
        from huggingface_hub import HfFileSystem
    except ImportError as exc:
        raise RuntimeError(
            "Downloading HiL-Bench assets requires huggingface_hub. "
            "Install the hilbench extra and authenticate with `hf auth login`."
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with HfFileSystem().open(
            f"hf://buckets/{repo_id}/{filename}", "rb"
        ) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def _business_info(shared_dir: Path) -> list[str]:
    path = shared_dir / "data" / "business_info.json"
    if not path.exists():
        return []
    raw = _read_json(path)
    if isinstance(raw, dict):
        raw = raw.get("business_info", [])
    return [str(item).strip() for item in raw if str(item).strip()] if isinstance(raw, list) else []


def _database_path(shared_dir: Path, database_name: str) -> Path | None:
    candidates = [
        shared_dir / "data" / "database.sqlite",
        shared_dir / "data" / f"{database_name}.sqlite",
        shared_dir / f"{database_name}.sqlite",
    ]
    return next((path for path in candidates if path.exists()), None)


def _sql_manifest(shared_dir: Path) -> dict[str, Any] | None:
    path = shared_dir / "data_archive.json"
    return _read_json(path) if path.exists() else None


def _sql_asset_state(
    shared_dir: Path,
    database_name: str,
    asset_root: Path | None,
) -> tuple[Path | None, list[Path]]:
    local_data = shared_dir / "data"
    database = _database_path(shared_dir, database_name)
    descriptions = {path.name: path for path in sorted(local_data.glob("*.csv"))}
    manifest = _sql_manifest(shared_dir)
    if asset_root is None or manifest is None:
        return database, list(descriptions.values())

    bucket_root = asset_root / "sql"
    db_candidate = bucket_root / str(manifest["db_artifact_path"])
    if database is None and db_candidate.exists():
        database = db_candidate
    for filename in map(str, manifest.get("desc_artifact_paths", [])):
        candidate = bucket_root / filename
        if candidate.exists() and candidate.name not in descriptions:
            descriptions[candidate.name] = candidate
    return database, [descriptions[name] for name in sorted(descriptions)]


def _selected(task_id: str, requested: set[str]) -> bool:
    if not requested:
        return True
    aliases = {
        task_id,
        task_id.removeprefix("sql_"),
        task_id.removeprefix("swe_"),
        task_id.removeprefix("hil-sql-"),
        task_id.removeprefix("hil-swe-"),
    }
    return bool(aliases & requested)


def load_hilbench_tasks(
    data_root: str | Path,
    *,
    task_type: TaskType,
    task_ids: Iterable[str] | None = None,
    limit: int | None = None,
    repository_root: str | Path | None = None,
    asset_root: str | Path | None = None,
) -> list[HilBenchTask]:
    root = Path(data_root).resolve()
    resolved_asset_root = resolve_asset_root(asset_root)
    requested = {
        item
        for value in (task_ids or [])
        for item in re.split(r"[\s,]+", value.strip())
        if item
    }
    raw_tasks = load_tasks(root, benchmark="hil-bench")
    tasks: list[HilBenchTask] = []
    for raw in raw_tasks:
        if raw.get("task_type") != task_type:
            continue
        task_id = str(raw["task_id"])
        if not _selected(task_id, requested):
            continue
        mode_dir = Path(raw["task_dir"]).resolve()
        shared_dir = mode_dir.parent / "shared"
        metadata_path = shared_dir / "metadata.json"
        metadata = _read_json(metadata_path) if metadata_path.exists() else {}
        if not raw.get("blocker_registry_path"):
            raise FileNotFoundError(f"Missing blocker registry for {task_id}")
        registry_path = Path(raw["blocker_registry_path"]).resolve()
        database_name = str(metadata.get("database_name", "database"))
        database_path, table_descriptions = _sql_asset_state(
            shared_dir,
            database_name,
            resolved_asset_root,
        ) if task_type == "sql" else (None, [])
        repo_path = None
        if repository_root:
            repo_root = Path(repository_root).resolve()
            candidates = [repo_root / task_id, repo_root / str(metadata.get("dataset_name", "")), repo_root]
            repo_path = next((candidate for candidate in candidates if candidate.is_dir()), None)
        tasks.append(
            HilBenchTask(
                task_id=task_id,
                task_type=task_type,
                problem=str(raw["modified_content"]),
                mode_dir=mode_dir,
                shared_dir=shared_dir,
                metadata=metadata,
                qa_pairs=list(raw.get("qa_pairs", [])),
                blocker_registry_path=registry_path,
                database_path=database_path,
                business_info=_business_info(shared_dir) if task_type == "sql" else [],
                table_descriptions=table_descriptions,
                repository_path=repo_path,
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    if requested:
        found = {task.task_id for task in tasks}
        missing = [item for item in requested if not any(_selected(task_id, {item}) for task_id in found)]
        if missing:
            raise ValueError(f"Unknown {task_type} task id(s): {', '.join(sorted(missing))}")
    return tasks


def download_sql_assets(
    task: HilBenchTask,
    asset_root: str | Path | None = None,
) -> Path:
    """Download only the database/schema artifacts declared by one SQL task."""
    if task.task_type != "sql":
        raise ValueError("SQL assets can only be downloaded for SQL tasks")
    archive_path = task.shared_dir / "data_archive.json"
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing SQL artifact manifest: {archive_path}")
    manifest = _read_json(archive_path)
    repo_id = str(manifest["hf_bucket"])
    files = [str(manifest["db_artifact_path"]), *map(str, manifest.get("desc_artifact_paths", []))]
    resolved_asset_root = resolve_asset_root(asset_root)
    data_dir = resolved_asset_root / "sql" if resolved_asset_root else task.shared_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in files:
        destination = data_dir / filename if resolved_asset_root else data_dir / Path(filename).name
        if destination.exists():
            continue
        _download_hf_bucket(repo_id, filename, destination)
    db_path = (
        data_dir / str(manifest["db_artifact_path"])
        if resolved_asset_root
        else data_dir / f"{task.metadata.get('database_name', 'database')}.sqlite"
    )
    if not db_path.exists():
        raise FileNotFoundError(f"Downloaded manifest did not produce {db_path}")
    return db_path


def download_swe_image(
    task: HilBenchTask,
    asset_root: str | Path | None = None,
) -> str:
    """Download and load the Docker image archive declared by one SWE task."""
    if task.task_type != "swe":
        raise ValueError("SWE images can only be downloaded for SWE tasks")
    image_name = str(task.metadata.get("image_name", "")).strip()
    if not image_name:
        raise ValueError(f"SWE task {task.task_id} has no image_name")
    inspect = subprocess.run(
        ["docker", "image", "inspect", image_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode == 0:
        return image_name
    archive = task.metadata.get("image_archive") or {}
    repo_id = str(archive.get("hf_bucket", "")).strip()
    filename = str(archive.get("artifact_path", "")).strip()
    if not repo_id or not filename:
        raise FileNotFoundError(f"No downloadable image archive is declared for {task.task_id}")
    resolved_asset_root = resolve_asset_root(asset_root)
    if resolved_asset_root:
        archive_path = resolved_asset_root / "swe" / filename
    else:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        archive_path = hf_home / "echocoder-assets" / repo_id / filename
    if not archive_path.exists():
        _download_hf_bucket(repo_id, filename, archive_path)
    if archive_path.name.endswith((".zst", ".tar.zst")):
        decompressor = subprocess.Popen(
            ["zstd", "-dc", str(archive_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        loaded = subprocess.run(
            ["docker", "load"],
            stdin=decompressor.stdout,
            capture_output=True,
            text=True,
            check=False,
        )
        if decompressor.stdout is not None:
            decompressor.stdout.close()
        decompress_stderr = (
            decompressor.stderr.read() if decompressor.stderr is not None else b""
        )
        decompressor.wait()
        if decompressor.returncode != 0:
            raise RuntimeError(f"Could not decompress {archive_path}: {decompress_stderr.decode(errors='replace').strip()}")
    else:
        loaded = subprocess.run(
            ["docker", "load", "-i", str(archive_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    if loaded.returncode != 0:
        raise RuntimeError(f"Could not load {archive_path}: {loaded.stderr.strip()}")
    verify = subprocess.run(
        ["docker", "image", "inspect", image_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if verify.returncode != 0:
        raise RuntimeError(
            f"Archive loaded, but expected image {image_name!r} is unavailable: "
            f"{loaded.stdout.strip()}"
        )
    return image_name
