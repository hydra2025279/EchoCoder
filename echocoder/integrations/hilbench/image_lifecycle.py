from __future__ import annotations

import subprocess
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections.abc import Callable
from threading import Lock

from .models import HilBenchTask
from .task_loader import download_swe_image


def image_id(reference: str) -> str | None:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", reference],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def image_ids() -> set[str]:
    result = subprocess.run(
        ["docker", "image", "ls", "--all", "--quiet", "--no-trunc"],
        capture_output=True, text=True, check=True,
    )
    return set(result.stdout.split())


def image_layers(reference: str) -> list[str]:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", reference],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return []
    layers = json.loads(result.stdout)
    return layers if isinstance(layers, list) else []


def owned_runtime_images(before: set[str], base_images: set[str]) -> dict[str, str]:
    parent_layers = [image_layers(base_image) for base_image in base_images]
    parent_layers = [layers for layers in parent_layers if layers]
    if not parent_layers:
        return {}
    runtime: dict[str, str] = {}
    for candidate in image_ids() - before - base_images:
        layers = image_layers(candidate)
        if not any(layers[:len(parent)] == parent for parent in parent_layers):
            continue
        history = subprocess.run(
            ["docker", "history", "--no-trunc", "--format", "{{.CreatedBy}}", candidate],
            capture_output=True, text=True, check=False,
        )
        if history.returncode == 0 and any(
            marker in history.stdout.lower()
            for marker in (
                "swe-rex", "python-build-standalone", "python-3.11.8", "python3.11/bin/swerex"
            )
        ):
            runtime[candidate] = candidate
    return runtime


def snapshot_swerex_artifacts() -> tuple[set[str], set[str]]:
    return image_ids(), set(build_cache_records())


def collect_swerex_runtime_images(
    tasks: list[HilBenchTask], before_images: set[str]
) -> dict[str, str]:
    base_images = {
        owned_id for task in tasks
        if (owned_id := image_id(str(task.metadata.get("image_name", ""))))
    }
    return owned_runtime_images(before_images, base_images)


def build_cache_records() -> dict[str, dict]:
    result = subprocess.run(
        ["docker", "buildx", "du", "--format=json"],
        capture_output=True, text=True, check=True,
    )
    records: dict[str, dict] = {}
    for line in result.stdout.splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("ID"):
                records[str(record["ID"])] = record
    return records


def cleanup_build_cache(cache_ids: set[str]) -> set[str]:
    records = build_cache_records()
    reclaimable = {
        cache_id for cache_id in cache_ids
        if records.get(cache_id, {}).get("Reclaimable") is True
    }
    if reclaimable:
        selector = "^(" + "|".join(sorted(reclaimable)) + ")$"
        subprocess.run(
            ["docker", "buildx", "prune", "--all", "--force", "--filter", f"id~={selector}"],
            capture_output=True, text=True, check=False,
        )
    remaining = build_cache_records()
    return cache_ids & remaining.keys()


def load_batch_images(
    tasks: list[HilBenchTask], asset_root: Path | None, owned: dict[str, str],
    on_owned: Callable[[], None] | None = None,
    max_workers: int = 1,
) -> None:
    selected: dict[str, HilBenchTask] = {}
    for task in tasks:
        reference = str(task.metadata.get("image_name", "")).strip()
        if reference and reference not in owned:
            selected.setdefault(reference, task)
    if not selected:
        return

    lock = Lock()

    def prepare(reference: str, task: HilBenchTask) -> str:
        before = image_id(reference)
        print(f"[echocoder_hilbench][images] loading {reference}", flush=True)
        try:
            download_swe_image(task, asset_root=asset_root)
        finally:
            after = image_id(reference)
            if before is None and after is not None:
                with lock:
                    owned[reference] = after
                    if on_owned is not None:
                        on_owned()
        return reference

    with ThreadPoolExecutor(max_workers=min(max_workers, len(selected))) as executor:
        futures = {
            executor.submit(prepare, reference, task): reference
            for reference, task in selected.items()
        }
        for index, future in enumerate(as_completed(futures), start=1):
            reference = futures[future]
            try:
                future.result()
            except Exception:
                print(
                    f"[echocoder_hilbench][images] failed {reference} ({index}/{len(selected)})",
                    flush=True,
                )
                raise
            print(
                f"[echocoder_hilbench][images] ready {reference} ({index}/{len(selected)})",
                flush=True,
            )


def load_cleanup_ledger(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in data.get("images", {}).items()}


def load_cache_cleanup_ledger(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(value) for value in data.get("build_cache", [])}


def write_cleanup_ledger(
    path: Path, owned: dict[str, str], build_cache: set[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"images": owned, "build_cache": sorted(build_cache or set())}, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def cleanup_batch_images(
    owned: dict[str, str], *, runtime_ids: set[str] | None = None
) -> dict[str, str]:
    failures: dict[str, str] = {}
    pending = dict(owned)
    for _ in range(2):
        failures = _cleanup_batch_images_once(pending, runtime_ids=runtime_ids or set())
        if not failures or len(failures) == len(pending):
            break
        pending = failures
    return failures


def _cleanup_batch_images_once(
    owned: dict[str, str], *, runtime_ids: set[str]
) -> dict[str, str]:
    failures: dict[str, str] = {}
    for reference, expected_id in sorted(
        owned.items(), key=lambda item: item[1] not in runtime_ids
    ):
        if image_id(expected_id) is None:
            continue
        containers = subprocess.run(
            [
                "docker", "ps", "-a", "--filter", f"ancestor={expected_id}",
                "--format", "{{.ID}}\t{{.Status}}",
            ],
            capture_output=True, text=True, check=False,
        )
        if containers.returncode != 0:
            failures[reference] = expected_id
            continue
        container_rows = [line.split("\t", 1) for line in containers.stdout.splitlines()]
        if expected_id not in runtime_ids and any(
            len(row) != 2 or not row[1].lower().startswith(("exited", "created"))
            for row in container_rows
        ):
            failures[reference] = expected_id
            continue
        container_ids = [row[0] for row in container_rows]
        if container_ids:
            removed = subprocess.run(
                ["docker", "rm", "-f", *container_ids],
                capture_output=True, text=True, check=False,
            )
            if removed.returncode != 0:
                failures[reference] = expected_id
                continue
        removed = subprocess.run(
            ["docker", "image", "rm", "--force", expected_id],
            capture_output=True, text=True, check=False,
        )
        if removed.returncode != 0 or image_id(expected_id) is not None:
            failures[reference] = expected_id
    return failures
