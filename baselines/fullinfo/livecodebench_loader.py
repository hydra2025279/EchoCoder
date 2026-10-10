from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_LCB_ROOT = Path(__file__).resolve().parents[2] / "benchmarks" / "ClarifyCodeBench" / "LiveCodeBench"


def _load_livecodebench(
    *, lcb_root: Path, release_version: str, not_fast: bool
) -> dict[str, Any]:
    if not (lcb_root / "lcb_runner").is_dir():
        raise FileNotFoundError(
            f"LiveCodeBench was not found at {lcb_root}. "
            "Pass --livecodebench-root with the cloned repository path."
        )
    sys.path.insert(0, str(lcb_root))

    try:
        from lcb_runner.benchmarks.code_generation import (
            CodeGenerationProblem,
            load_code_generation_dataset,
            load_code_generation_dataset_not_fast,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LiveCodeBench dependencies are not installed in this environment. "
            f"Activate its environment and run `pip install -e {lcb_root}`."
        ) from exc

    if not_fast:
        problems = load_code_generation_dataset_not_fast(release_version)
    elif release_version in {f"v{index}" for index in range(1, 7)}:
        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Fine-grained LiveCodeBench releases require huggingface_hub"
            ) from exc

        shard_number = release_version.removeprefix("v")
        filename = "test.jsonl" if shard_number == "1" else f"test{shard_number}.jsonl"
        shard_path = Path(
            hf_hub_download(
                repo_id="livecodebench/code_generation_lite",
                filename=filename,
                repo_type="dataset",
            )
        )
        problems = [
            CodeGenerationProblem(**json.loads(line))
            for line in shard_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(
            f"Loaded {len(problems)} problems directly from {filename}",
            flush=True,
        )
    else:
        problems = load_code_generation_dataset(release_version)

    by_id: dict[str, Any] = {}
    duplicate_ids: set[str] = set()
    for problem in problems:
        if problem.question_id in by_id:
            duplicate_ids.add(problem.question_id)
        by_id[problem.question_id] = problem
    if duplicate_ids:
        preview = ", ".join(sorted(duplicate_ids)[:5])
        raise RuntimeError(f"LiveCodeBench contains duplicate question IDs: {preview}")
    return by_id
