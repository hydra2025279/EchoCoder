"""Standard-input execution and output-vector clustering for ClarifyGPT."""

from __future__ import annotations

import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def execute_candidates(
    candidates: list[dict[str, Any]],
    input_cases: list[dict[str, str]],
    *,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    executions: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="clarifygpt_candidates_") as directory:
        root = Path(directory)
        for candidate in candidates:
            code_path = root / f"candidate_{candidate['candidate_id']}.py"
            code_path.write_text(candidate["code"], encoding="utf-8")
            outcomes = []
            for case_index, case in enumerate(input_cases, start=1):
                try:
                    process = subprocess.run(
                        [sys.executable, str(code_path)],
                        input=case["input"],
                        text=True,
                        capture_output=True,
                        timeout=timeout_seconds,
                        check=False,
                    )
                    if process.returncode == 0:
                        value = f"ok:{process.stdout.strip()[:4000]}"
                        status = "ok"
                    else:
                        # The upstream consistency check treats every exception as
                        # one observable error behavior rather than clustering by
                        # traceback text or exit status.
                        value = "error"
                        status = "error"
                except subprocess.TimeoutExpired:
                    value = "error"
                    status = "timeout"
                outcomes.append(
                    {
                        "case_index": case_index,
                        "kind": case["kind"],
                        "status": status,
                        "value": value,
                    }
                )
            executions.append({"candidate_id": candidate["candidate_id"], "outcomes": outcomes})
    return executions


def cluster_by_output(
    candidates: list[dict[str, Any]], executions: list[dict[str, Any]], *, random_seed: int
) -> list[dict[str, Any]]:
    by_candidate = {item["candidate_id"]: item["outcomes"] for item in executions}
    groups: dict[tuple[str, ...], list[str]] = {}
    for candidate in candidates:
        signature = tuple(item["value"] for item in by_candidate[candidate["candidate_id"]])
        groups.setdefault(signature, []).append(candidate["candidate_id"])
    random_generator = random.Random(random_seed)
    return [
        {
            "group_id": f"G{index}",
            "candidate_ids": candidate_ids,
            "representative_id": random_generator.choice(candidate_ids),
            "output_signature": list(signature),
        }
        for index, (signature, candidate_ids) in enumerate(groups.items(), start=1)
    ]
