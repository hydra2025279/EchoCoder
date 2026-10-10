"""Shared helpers for baseline pipelines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from openai import OpenAI

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARKS_ROOT = _WORKSPACE_ROOT / "benchmarks"
_CLARIFY_PROJECT_ROOT = _BENCHMARKS_ROOT / "ClarifyCodeBench"
if _CLARIFY_PROJECT_ROOT.exists() and str(_CLARIFY_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLARIFY_PROJECT_ROOT))

from clarifycodebench.llm import make_client
from clarifycodebench.prompts import JUDGE_PROMPT


HIL_BLOCKER_JUDGE_PROMPT = """\
Determine whether the agent's question identifies the same single, specific
missing requirement as the HIL-Bench blocker.

Reject the question if it is not a real question, is vague or generic, combines
multiple topics, merely asks for a definition without identifying the ambiguity,
or embeds an unsupported assumption and asks for confirmation. Accept it only if
it targets the precise blocker topic, even when the wording differs from the
examples.

Blocker description and examples:
{key_question}

Agent question:
{model_question}

Return only yes or no.
"""


_CURRENT_LLM_CALL_LOG: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "current_llm_call_log",
    default=None,
)
_CURRENT_LLM_CALL_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "current_llm_call_context",
    default={},
)
_CURRENT_LLM_LIVE_OUTPUT: ContextVar[dict[str, str] | None] = ContextVar(
    "current_llm_live_output",
    default=None,
)
_CURRENT_LIVE_TEXT_REFRESH: ContextVar[Callable[[], None] | None] = ContextVar(
    "current_live_text_refresh",
    default=None,
)


def set_llm_call_log(
    call_log: list[dict[str, Any]] | None,
    *,
    live_output_dir: str | Path | None = None,
    task_id: str | None = None,
) -> None:
    if call_log is None:
        previous_log = _CURRENT_LLM_CALL_LOG.get()
        if previous_log is not None and _CURRENT_LLM_LIVE_OUTPUT.get() is not None:
            _write_live_llm_state(previous_log, in_progress=False)
        _CURRENT_LLM_CALL_LOG.set(None)
        _CURRENT_LLM_LIVE_OUTPUT.set(None)
        _CURRENT_LIVE_TEXT_REFRESH.set(None)
        return
    _CURRENT_LLM_CALL_LOG.set(call_log)
    _CURRENT_LIVE_TEXT_REFRESH.set(None)
    if live_output_dir is None:
        _CURRENT_LLM_LIVE_OUTPUT.set(None)
        return
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id or "unknown_task")
    output_dir = Path(live_output_dir)
    calls_path = output_dir / "logs" / f"{safe_id}.calls.jsonl"
    cost_path = output_dir / "cost.live" / f"{safe_id}.json"
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    cost_path.parent.mkdir(parents=True, exist_ok=True)
    calls_path.write_text("", encoding="utf-8")
    _CURRENT_LLM_LIVE_OUTPUT.set(
        {"calls_path": str(calls_path), "cost_path": str(cost_path)}
    )
    _write_live_llm_state(call_log)


def set_live_text_refresh(callback: Callable[[], None] | None) -> None:
    _CURRENT_LIVE_TEXT_REFRESH.set(callback)


def _write_live_llm_state(
    call_log: list[dict[str, Any]], *, in_progress: bool = True
) -> None:
    live = _CURRENT_LLM_LIVE_OUTPUT.get()
    if live is None:
        return
    cost_path = Path(live["cost_path"])
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "in_progress": in_progress,
        "token_usage": summarize_llm_call_usage(call_log),
    }
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=cost_path.parent,
        prefix=f".{cost_path.name}.",
        delete=False,
    ) as temporary:
        json.dump(payload, temporary, ensure_ascii=False, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(cost_path)


@contextmanager
def llm_call_context(**values: Any):
    current = _CURRENT_LLM_CALL_CONTEXT.get()
    merged = {
        **current,
        **{key: value for key, value in values.items() if value is not None},
    }
    token = _CURRENT_LLM_CALL_CONTEXT.set(merged)
    try:
        yield
    finally:
        _CURRENT_LLM_CALL_CONTEXT.reset(token)


def current_llm_call_context() -> dict[str, Any]:
    return dict(_CURRENT_LLM_CALL_CONTEXT.get())


def append_external_llm_call(call: dict[str, Any]) -> None:
    call_log = _CURRENT_LLM_CALL_LOG.get()
    if call_log is None:
        return
    call_log.append(call)
    live = _CURRENT_LLM_LIVE_OUTPUT.get()
    if live is not None:
        with Path(live["calls_path"]).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(call, ensure_ascii=False) + "\n")
        _write_live_llm_state(call_log)
    refresh = _CURRENT_LIVE_TEXT_REFRESH.get()
    if refresh is not None:
        try:
            refresh()
        except OSError as error:
            print(f"[log] realtime TXT update failed: {error}", flush=True)


@dataclass
class AzureResponsesClient:
    """Azure Foundry client with Responses and Chat Completions compatibility."""

    endpoint: str
    api_key: str
    default_deployment: str | None = None

    @property
    def base_url(self) -> str:
        parsed = urlsplit(self.endpoint)
        if parsed.scheme != "https" or not parsed.netloc:
            raise RuntimeError("LLM_BASE_URL must be a valid HTTPS URL")
        return urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")

    @property
    def openai_v1_bases(self) -> list[str]:
        parsed = urlsplit(self.endpoint.strip())
        if parsed.scheme != "https" or not parsed.netloc:
            raise RuntimeError("LLM_BASE_URL must be a valid HTTPS URL")
        root = urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")
        path = parsed.path.rstrip("/")
        bases: list[str] = []
        if path.endswith("/openai/v1"):
            bases.append(f"{root}{path}")
        elif path:
            bases.append(f"{root}{path}/openai/v1")
        resource_base = f"{root}/openai/v1"
        if resource_base not in bases:
            bases.append(resource_base)
        return bases

    @staticmethod
    def _prefers_chat_completions(model: str) -> bool:
        return "deepseek" in model.lower()

    @staticmethod
    def _uses_completion_token_field(model: str) -> bool:
        name = model.lower()
        return any(value in name for value in ("deepseek", "kimi", "luna", "gpt-5"))

    @staticmethod
    def _unsupported_parameter(error: RuntimeError) -> str | None:
        match = re.search(r"Unsupported parameter: '([^']+)'", str(error))
        if match:
            return match.group(1)
        match = re.search(r'"param"\s*:\s*"([^"]+)"', str(error))
        return match.group(1) if match else None

    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "api-key": self.api_key,
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=120) as response:
                result = json.loads(response.read())
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Azure API returned HTTP {error.code}: {body}") from None
        except URLError as error:
            raise RuntimeError(
                f"Azure API connection failed: {type(error.reason).__name__}"
            ) from None
        if not isinstance(result, dict):
            raise RuntimeError("Azure returned an unexpected response")
        return result

    def _post_with_token_fallback(
        self, url: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        payload = dict(body)
        for _ in range(2):
            try:
                return self._post(url, payload)
            except RuntimeError as error:
                parameter = self._unsupported_parameter(error)
                if parameter == "max_tokens" and "max_tokens" in payload:
                    payload["max_completion_tokens"] = payload.pop("max_tokens")
                    continue
                if (
                    parameter == "max_completion_tokens"
                    and "max_completion_tokens" in payload
                ):
                    payload["max_tokens"] = payload.pop("max_completion_tokens")
                    continue
                raise
        raise RuntimeError("Azure rejected both completion-token parameter names")

    def responses_create(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        temperature: float,
        max_output_tokens: int | None,
    ) -> dict[str, Any]:
        del temperature
        deployment = self.default_deployment or model
        apis = (
            ("chat", "responses")
            if self._prefers_chat_completions(deployment)
            else ("responses", "chat")
        )
        last_error: RuntimeError | None = None
        for base in self.openai_v1_bases:
            for api in apis:
                if api == "responses":
                    body = {
                        "model": deployment,
                        "instructions": instructions,
                        "input": input_text,
                        "store": False,
                    }
                    if max_output_tokens is not None:
                        body["max_output_tokens"] = max_output_tokens
                    url = f"{base}/responses"
                else:
                    token_field = (
                        "max_completion_tokens"
                        if self._uses_completion_token_field(deployment)
                        else "max_tokens"
                    )
                    body = {
                        "model": deployment,
                        "messages": [
                            {"role": "system", "content": instructions},
                            {"role": "user", "content": input_text},
                        ],
                    }
                    if max_output_tokens is not None:
                        body[token_field] = max_output_tokens
                    url = f"{base}/chat/completions"
                try:
                    result = self._post_with_token_fallback(url, body)
                    _output_text(result)
                    return result
                except RuntimeError as error:
                    last_error = error
                    if any(f"HTTP {code}" in str(error) for code in (401, 403, 429)):
                        raise
        raise last_error or RuntimeError(
            f"Azure request failed for deployment {deployment}"
        )


def _normalise_benchmark(name: str | None) -> str:
    value = (name or "clarifycodebench").strip().lower().replace("_", "-")
    aliases = {
        "clarify": "clarifycodebench",
        "clarify-code-bench": "clarifycodebench",
        "clarifycodebench": "clarifycodebench",
        "hil": "hil-bench",
        "hilbench": "hil-bench",
        "hil-bench": "hil-bench",
    }
    try:
        return aliases[value]
    except KeyError:
        raise ValueError(f"Unsupported benchmark: {name!r}") from None


def _default_data_path(benchmark: str) -> Path:
    if benchmark == "hil-bench":
        return _BENCHMARKS_ROOT / "hil-bench"
    return _CLARIFY_PROJECT_ROOT / "data" / "ClarifyCodeBench.jsonl"


def _load_clarify_tasks(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if limit is not None and limit <= 0:
        return []
    tasks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
                if limit is not None and len(tasks) >= limit:
                    break
    return tasks


def _hil_mode_dirs(data_path: Path) -> list[Path]:
    if data_path.is_file():
        raise ValueError(f"HIL-Bench data path must be a task directory or directory: {data_path}")

    if (data_path / "instruction.md").exists():
        return [data_path]

    candidates = []
    for instruction in data_path.rglob("instruction.md"):
        mode_name = instruction.parent.name.lower()
        if mode_name == "baseline" or mode_name.endswith("-baseline"):
            candidates.append(instruction.parent)
    return sorted(set(candidates), key=lambda item: str(item))


def _hil_task_identity(mode_dir: Path) -> tuple[str, str]:
    name = mode_dir.name
    match = re.search(r"(?:hil-bench-)?(sql|swe)[_-](\d+)-baseline$", name, re.IGNORECASE)
    if match:
        task_type, index = match.groups()
        return f"hil-{task_type.lower()}-{index}", task_type.lower()

    if name == "baseline" and mode_dir.parent.name.startswith(("sql_", "swe_")):
        task_type, index = mode_dir.parent.name.split("_", 1)
        return f"hil-{task_type.lower()}-{index}", task_type.lower()

    parent_name = mode_dir.parent.name
    match = re.match(r"(sql|swe)_(\d+)$", parent_name, re.IGNORECASE)
    if match:
        task_type, index = match.groups()
        return f"hil-{task_type.lower()}-{index}", task_type.lower()
    return f"hil-{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}", "unknown"


def _find_hil_blocker_registry(mode_dir: Path, task_type: str) -> Path | None:
    candidates = [
        mode_dir / "blocker_registry.json",
        mode_dir.parent / "blocker_registry.json",
        mode_dir.parent / "shared" / "ask-human-data" / "blocker_registry.json",
        mode_dir.parent / "shared" / "blocker_registry.json",
    ]

    match = re.search(r"(?:sql|swe)[_-](\d+)", mode_dir.name, re.IGNORECASE)
    if match and task_type in {"sql", "swe"}:
        index = match.group(1)
        candidates.append(
            _BENCHMARKS_ROOT
            / "hil-bench"
            / f"harbor_{task_type}"
            / f"{task_type}_{index}"
            / "shared"
            / "ask-human-data"
            / "blocker_registry.json"
        )

    return next((candidate for candidate in candidates if candidate.exists()), None)


def _load_hil_task(mode_dir: Path) -> dict[str, Any]:
    task_id, task_type = _hil_task_identity(mode_dir)
    problem_path = mode_dir / "instruction.md"
    problem = problem_path.read_text(encoding="utf-8")
    registry_path = _find_hil_blocker_registry(mode_dir, task_type)
    blockers: list[dict[str, Any]] = []
    if registry_path:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
        blockers = raw.get("blockers", []) if isinstance(raw, dict) else []

    qa_pairs = []
    for blocker in blockers:
        if not isinstance(blocker, dict):
            continue
        examples = blocker.get("example_questions") or blocker.get("acceptable_questions") or []
        examples = [str(item).strip() for item in examples if str(item).strip()]
        question = examples[0] if examples else str(blocker.get("description", "")).strip()
        qa_pairs.append(
            {
                "question": question,
                "answer": blocker.get("resolution", ""),
                "blocker_id": blocker.get("id"),
                "description": blocker.get("description", ""),
                "example_questions": examples,
            }
        )

    metadata = {}
    metadata_path = mode_dir.parent / "shared" / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return {
        "task_id": task_id,
        "question_id": metadata.get("hf_task_id") or metadata.get("task_id") or task_id,
        "benchmark": "hil-bench",
        "task_type": task_type,
        "task_dir": str(mode_dir),
        "blocker_registry_path": str(registry_path) if registry_path else None,
        "blockers": blockers,
        "modified_content": problem,
        "qa_pairs": qa_pairs,
    }


def load_tasks(
    path: str | Path | None,
    limit: int | None = None,
    *,
    benchmark: str = "clarifycodebench",
    task_ids: str | Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    benchmark = _normalise_benchmark(benchmark)
    data_path = Path(path) if path else _default_data_path(benchmark)
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path

    if isinstance(task_ids, str):
        task_id_values = [task_ids]
    else:
        task_id_values = list(task_ids or [])
    requested_task_ids = {
        item
        for value in task_id_values
        for item in re.split(r"[\s,]+", value.strip())
        if item
    }

    if benchmark == "clarifycodebench":
        tasks = _load_clarify_tasks(data_path, None if requested_task_ids else limit)
    else:
        mode_dirs = _hil_mode_dirs(data_path)
        tasks = []
        seen_task_ids: set[str] = set()
        for mode_dir in mode_dirs:
            task = _load_hil_task(mode_dir)
            if task["task_id"] in seen_task_ids:
                continue
            seen_task_ids.add(task["task_id"])
            tasks.append(task)

    if requested_task_ids:
        matched_ids: set[str] = set()
        selected_tasks = []
        for task in tasks:
            task_id = str(task.get("task_id", ""))
            aliases = {task_id}
            if benchmark == "clarifycodebench" and task_id.startswith("task_"):
                aliases.add(task_id.removeprefix("task_"))
            matches = aliases & requested_task_ids
            if matches:
                selected_tasks.append(task)
                matched_ids.update(matches)
        missing_ids = sorted(requested_task_ids - matched_ids)
        if missing_ids:
            raise ValueError(f"Unknown task id(s): {', '.join(missing_ids)}")
        tasks = selected_tasks

    return tasks[:limit] if limit is not None else tasks


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return records
    changed = False
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                published = publish_clarify_artifact(item)
                changed |= published != item
                records.append(published)
    if changed:
        write_jsonl(records, p)
    return records


def completed_task_ids(records: Iterable[dict[str, Any]]) -> set[str]:
    return {str(record["task_id"]) for record in records if record.get("task_id") is not None}


def write_jsonl(records: Iterable[dict[str, Any]], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(publish_clarify_artifact(record), ensure_ascii=False) + "\n")


def reset_jsonl(out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("", encoding="utf-8")


def append_jsonl(record: dict[str, Any], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(publish_clarify_artifact(record), ensure_ascii=False) + "\n")


def write_json(data: dict[str, Any], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(publish_clarify_artifact(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _task_field(task: Any, name: str, default: Any = None) -> Any:
    return task.get(name, default) if isinstance(task, dict) else getattr(task, name, default)


def _safe_arguments(args: argparse.Namespace) -> dict[str, Any]:
    values = {}
    for key, value in vars(args).items():
        if "key" in key.lower() or "token" in key.lower() and key != "max_tokens":
            continue
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
            values[key] = value
    return values


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def initialize_run_manifest(
    output_dir: str | Path,
    *,
    args: argparse.Namespace,
    tasks: Iterable[Any],
    method: str,
    benchmark: str,
    prompt_file: str | Path,
) -> dict[str, Any]:
    out = Path(output_dir) / "run.json"
    task_list = list(tasks)
    task_ids = [str(_task_field(task, "task_id", "")) for task in task_list]
    dataset_payload = [
        {
            "task_id": task_id,
            "question_id": _task_field(task, "question_id"),
            "problem": _task_field(
                task,
                "modified_content",
                _task_field(task, "problem", ""),
            ),
        }
        for task_id, task in zip(task_ids, task_list, strict=True)
    ]
    prompt_path = Path(prompt_file)
    prompt_bytes = prompt_path.read_bytes() if prompt_path.exists() else b""
    prior = {}
    if out.exists() and not args.no_resume:
        try:
            prior = json.loads(out.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            prior = {}
    manifest = {
        "run_id": prior.get(
            "run_id", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        ),
        "status": "running",
        "started_at": prior.get("started_at", datetime.now(timezone.utc).isoformat()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "method": method,
        "benchmark": benchmark,
        "arguments": _safe_arguments(args),
        "prompt": {
            "version": "method_v3",
            "path": str(prompt_path),
            "sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        },
        "git": _git_metadata(),
        "dataset": {
            "task_count": len(task_ids),
            "task_ids": task_ids,
            "sha256": hashlib.sha256(
                json.dumps(dataset_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
        "progress": prior.get("progress", {}),
    }
    write_json(manifest, out)
    return manifest


def update_run_manifest(
    output_dir: str | Path,
    *,
    completed_records: Iterable[dict[str, Any]],
    status: str = "running",
) -> dict[str, Any]:
    path = Path(output_dir) / "run.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    completed_ids = completed_task_ids(completed_records)
    failures = load_task_error_records(output_dir)
    failed_ids = {
        str(record["task_id"])
        for record in failures
        if record.get("task_id") is not None and str(record["task_id"]) not in completed_ids
    }
    requested_ids = set(manifest.get("dataset", {}).get("task_ids", []))
    manifest["status"] = status
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["finished_at"] = (
        manifest["updated_at"] if status in {"completed", "interrupted"} else None
    )
    manifest["progress"] = {
        "requested_task_count": len(requested_ids),
        "completed_task_count": len(completed_ids & requested_ids),
        "failed_task_count": len(failed_ids & requested_ids),
        "pending_task_count": len(requested_ids - completed_ids - failed_ids),
        "completed_task_ids": sorted(completed_ids & requested_ids),
        "failed_task_ids": sorted(failed_ids & requested_ids),
    }
    write_json(manifest, path)
    return manifest


def method_output_dir(
    method_file: str | Path,
    output_dir: str | Path | None = None,
    *,
    benchmark: str = "clarifycodebench",
) -> Path:
    if output_dir:
        out = Path(output_dir)
    else:
        out = Path(method_file).resolve().parent / "output"
        if _normalise_benchmark(benchmark) != "clarifycodebench":
            out = out / _normalise_benchmark(benchmark)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    return out


def resolve_results_path(out_path: str | Path | None, output_dir: Path) -> Path:
    return Path(out_path) if out_path else output_dir / "results.jsonl"


def question_summary_path(output_dir: Path) -> Path:
    return output_dir / "summary.question.json"


def passrate_summary_path(output_dir: Path) -> Path:
    return output_dir / "summary.passrate.json"


def cost_jsonl_path(output_dir: Path) -> Path:
    return output_dir / "cost.jsonl"


def cost_summary_path(output_dir: Path) -> Path:
    return output_dir / "cost.json"


def task_log_path(output_dir: Path, task_id: str | None) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id or "unknown_task")
    return Path(output_dir) / "logs" / f"{safe_id}.txt"


def task_error_log_path(output_dir: Path, task_id: str | None) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id or "unknown_task")
    return Path(output_dir) / "logs" / f"{safe_id}.error.txt"


def task_error_record_path(output_dir: str | Path, task_id: str | None) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id or "unknown_task")
    return Path(output_dir) / "errors" / f"{safe_id}.json"


def load_task_error_records(output_dir: str | Path) -> list[dict[str, Any]]:
    error_dir = Path(output_dir) / "errors"
    records = []
    if not error_dir.exists():
        return records
    for path in sorted(error_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def clear_task_error_records(output_dir: str | Path) -> None:
    error_dir = Path(output_dir) / "errors"
    if error_dir.exists():
        for path in error_dir.glob("*.json"):
            path.unlink()


def remove_task_error_record(output_dir: str | Path, task_id: str | None) -> None:
    path = task_error_record_path(output_dir, task_id)
    if path.exists():
        path.unlink()


def write_task_log(output_dir: Path, task_id: str | None, payload: dict[str, Any]) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    path = task_log_path(output_dir, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_task_log(task_id=task_id, created_at=created_at, payload=publish_clarify_artifact(payload))
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(rendered)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def write_task_log_from_record(
    output_dir: str | Path, record: dict[str, Any]
) -> None:
    rounds = record.get("rounds", []) or []
    write_task_log(
        Path(output_dir),
        record.get("task_id"),
        {
            "method": record.get("method"),
            "workflow_version": record.get("workflow_version"),
            "benchmark": record.get("benchmark", "clarifycodebench"),
            "task_type": record.get("task_type"),
            "question_id": record.get("question_id"),
            "input_problem": record.get("original_problem", record.get("problem", "")),
            "qa_pairs": record.get("ground_truth", []),
            "steps": {
                "budgets": {
                    "K_r": record.get("reconstruction_budget"),
                    "K_q": record.get("question_call_budget"),
                },
                "rounds": rounds,
                "final_specification": record.get("final_specification"),
                "stop_reason": record.get("stop_reason"),
            },
            "generated_questions": record.get("generated_questions", []),
            "llm_call_logs": record.get("llm_calls", []),
            "question_metrics": record.get("question_metrics"),
            "pass_evaluation": record.get("pass_evaluation"),
            "record": record,
        },
    )


def write_task_error_log(
    output_dir: Path,
    task_id: str | None,
    *,
    method: str,
    question_id: str | None,
    benchmark: str = "clarifycodebench",
    task_type: str | None = None,
    step: str,
    error: Exception,
    input_problem: str,
    qa_pairs: list[dict[str, Any]],
    llm_call_logs: list[dict[str, Any]],
) -> None:
    payload = {
        "method": method,
        "benchmark": benchmark,
        "task_type": task_type,
        "question_id": question_id,
        "input_problem": input_problem,
        "qa_pairs": qa_pairs,
        "steps": {
            "failed_step": step,
            "error_type": type(error).__name__,
            "error": str(error),
        },
        "generated_questions": [],
        "llm_call_logs": llm_call_logs,
        "question_metrics": None,
    }
    created_at = datetime.now(timezone.utc).isoformat()
    task_error_log_path(output_dir, task_id).write_text(
        render_task_log(task_id=task_id, created_at=created_at, payload=payload),
        encoding="utf-8",
    )
    ground_truth = [
        {
            "ground_truth_id": str(
                qa.get("blocker_id") or qa.get("question_id") or f"GT_{index}"
            ),
            "question": qa.get("question"),
            "answer": qa.get("answer"),
        }
        for index, qa in enumerate(qa_pairs, start=1)
    ]
    error_record = {
        "task_id": task_id,
        "question_id": question_id,
        "method": method,
        "benchmark": benchmark,
        "task_type": task_type,
        "status": "failed",
        "failed_step": step,
        "error_type": type(error).__name__,
        "error": str(error),
        "created_at": created_at,
        "ground_truth": ground_truth,
        "llm_calls": llm_call_logs,
        "token_usage": summarize_llm_call_usage(llm_call_logs),
    }
    write_json(error_record, task_error_record_path(output_dir, task_id))


def _section(title: str, body: str) -> str:
    return f"\n{'=' * 80}\n{title}\n{'=' * 80}\n{body.strip() if body else '(empty)'}\n"


def _format_value(value: Any) -> str:
    if value is None:
        return "(none)"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _format_bullet_list(items: Iterable[Any]) -> str:
    values = [str(item).strip() for item in items if str(item).strip()]
    return "\n".join(f"- {item}" for item in values) if values else "- None"


def _format_ground_truth(qa_pairs: list[dict[str, Any]]) -> str:
    if not qa_pairs:
        return "(no annotated missing requirements)"

    chunks = []
    for idx, qa in enumerate(qa_pairs, start=1):
        blocker_lines = []
        if qa.get("blocker_id"):
            blocker_lines.append(f"Blocker ID: {qa['blocker_id']}")
        if qa.get("description"):
            blocker_lines.append(f"Blocker description: {qa['description']}")
        if qa.get("example_questions"):
            blocker_lines.append(
                "Example questions: "
                + " | ".join(str(item) for item in qa["example_questions"])
            )
        chunks.append(
            "\n".join(
                [
                    f"[{idx}] Key question:",
                    str(qa.get("question", "")).strip(),
                    *blocker_lines,
                    "",
                    "Ground-truth missing information:",
                    str(qa.get("answer", "")).strip(),
                ]
            )
        )
    return "\n\n".join(chunks)


def _render_iterative_trajectory(payload: dict[str, Any]) -> str:
    record = payload.get("record") or {}
    rounds = record.get("rounds")
    if rounds is None:
        rounds = (payload.get("steps") or {}).get("rounds")
    if not isinstance(rounds, list) or not rounds:
        return ""

    lines: list[str] = []
    for state_offset, state in enumerate(rounds):
        state_index = int(state.get("specification_round", state_offset))
        state_id = state.get("state_id") or f"S{state_index}"
        code_id = state.get("code_id") or f"C{state_index}"
        generation_id = state.get("generation_id") or f"{state_id}_GEN"
        operation = state.get("generation_operation") or (
            "initial_complete_generation"
            if state_index == 0
            else "complete_repair_or_regeneration_after_clarification"
        )
        lines.extend(
            [
                f"[SPECIFICATION STATE {state_id}]",
                f"state_index: {state_index}",
                f"generation_id: {generation_id}",
                f"generation_engine: {state.get('generation_engine') or '(unspecified)'}",
                f"generation_operation: {operation}",
                f"code_id: {code_id}",
                f"parent_code_id: {state.get('parent_code_id') or '(none)'}",
                "",
                (
                    f"--- CURRENT SPECIFICATION {state_id} "
                    "(ORIGINAL UNDERSPECIFIED SPECIFICATION) ---"
                    if state_index == 0
                    else f"--- CURRENT AUGMENTED SPECIFICATION {state_id} "
                    "(ORIGINAL + ACCEPTED CLARIFICATIONS) ---"
                ),
                str(state.get("specification", "")),
                "",
                f"--- COMPLETE CODE {code_id} PRODUCED BY {generation_id} ---",
                str(state.get("complete_code", "")),
                "",
                f"--- INITIAL QUESTIONS FOR {generation_id} ---",
                _format_value(state.get("initial_questions", [])),
            ]
        )

        for reconstruction_offset, reconstruction in enumerate(
            state.get("reconstructions", []) or [], start=1
        ):
            r = int(
                reconstruction.get("reconstruction_index", reconstruction_offset)
            )
            reconstruction_id = (
                reconstruction.get("reconstruction_id") or f"{state_id}_{code_id}_R{r}"
            )
            lines.extend(
                [
                    "",
                    f"[RECONSTRUCTION {reconstruction_id}]",
                    f"r: {r}",
                    f"source_code_id: {code_id}",
                    "",
                    f"--- RECONSTRUCTED SPECIFICATION {reconstruction_id} "
                    f"(DERIVED FROM CODE {code_id}) ---",
                    str(reconstruction.get("reconstructed_description", "")),
                    "",
                    f"--- ADDED POINTS {reconstruction_id} ---",
                    _format_bullet_list(
                        [str(item) for item in reconstruction.get("added_points", [])]
                    ),
                ]
            )

            for batch_offset, batch in enumerate(
                reconstruction.get("question_batches", []) or [], start=1
            ):
                q = int(batch.get("question_call_index", batch_offset))
                batch_id = batch.get("batch_id") or f"Q{r}_{q}"
                global_batch_id = (
                    batch.get("global_batch_id") or f"{state_id}_{code_id}_{batch_id}"
                )
                questions = batch.get("questions", []) or []
                valid_count = sum(
                    question.get("oracle_status") == "valid"
                    for question in questions
                )
                lines.extend(
                    [
                        "",
                        f"[QUESTION BATCH {global_batch_id}]",
                        f"local_batch_id: {batch_id}",
                        f"r: {r}",
                        f"q: {q}",
                        f"source_code_id: {code_id}",
                        f"question_count: {len(questions)}",
                        f"valid_question_count: {valid_count}",
                        f"batch_valid: {str(valid_count > 0).lower()}",
                        "",
                        f"--- DETECTOR REASONING {global_batch_id} ---",
                        extract_tag(
                            str(batch.get("detector_response", "")), "REASONING"
                        )
                        or "(none)",
                    ]
                )
                for question_offset, question in enumerate(questions, start=1):
                    local_question_id = (
                        question.get("question_id")
                        or f"{batch_id}_{question_offset}"
                    )
                    global_question_id = (
                        question.get("global_question_id")
                        or f"{global_batch_id}_{question_offset}"
                    )
                    lines.extend(
                        [
                            "",
                            f"[QUESTION {global_question_id}]",
                            f"local_question_id: {local_question_id}",
                            f"oracle_status: {question.get('oracle_status')}",
                            f"question: {question.get('question', '')}",
                            f"answer: {question.get('answer') or '(none)'}",
                            "matched_key_question: "
                            + str(question.get("matched_key_question") or "(none)"),
                            "matched_ground_truth_id: "
                            + str(
                                (question.get("oracle_evaluation") or {}).get(
                                    "matched_key_id"
                                )
                                or "(none)"
                            ),
                            "duplicate: "
                            + str(
                                bool(
                                    (question.get("oracle_evaluation") or {}).get(
                                        "duplicate"
                                    )
                                )
                            ).lower(),
                        ]
                    )

        accepted = state.get("accepted_clarifications", []) or []
        lines.extend(["", f"--- ACCEPTED CLARIFICATIONS FOR {state_id} ---"])
        if accepted:
            for clarification in accepted:
                lines.extend(
                    [
                        f"- event_id: {clarification.get('event_id')}",
                        f"  question_id: {clarification.get('global_question_id')}",
                        f"  question: {clarification.get('question')}",
                        f"  answer: {clarification.get('answer')}",
                    ]
                )
        else:
            lines.append("- None")
        if accepted:
            lines.extend(
                [
                    f"transition: {state_id} -> S{state_index + 1}",
                    "transition_reason: specification_updated_by_valid_clarification",
                ]
            )
        lines.extend(["", "-" * 80, ""])

    final_state_id = record.get("final_state_id") or f"S{len(rounds) - 1}"
    final_code_id = record.get("final_code_id") or f"C{len(rounds) - 1}"
    lines.extend(
        [
            f"stop_reason: {record.get('stop_reason') or '(unknown)'}",
            f"final_state_id: {final_state_id}",
            f"final_code_id: {final_code_id}",
        ]
    )
    return "\n".join(lines).strip()


def _render_llm_interactions(call_logs: list[dict[str, Any]]) -> str:
    if not call_logs:
        return "(no LLM calls recorded yet)"
    lines: list[str] = []
    for index, call in enumerate(call_logs, start=1):
        call_id = call.get("call_id") or f"call_{index:04d}"
        lines.extend(
            [
                f"[LLM CALL {call_id}]",
                f"phase: {call.get('phase') or call.get('step') or '(unknown)'}",
                f"step: {call.get('step') or '(unknown)'}",
                f"state_id: {call.get('state_id') or '(none)'}",
                f"generation_id: {call.get('generation_id') or '(none)'}",
                f"code_id: {call.get('code_id') or '(none)'}",
                f"reconstruction_id: {call.get('reconstruction_id') or '(none)'}",
                f"batch_id: {call.get('global_batch_id') or '(none)'}",
                f"question_id: {call.get('global_question_id') or '(none)'}",
                f"ground_truth_id: {call.get('ground_truth_id') or '(none)'}",
                f"status: {call.get('status') or '(unknown)'}",
                f"attempt: {call.get('attempt', 1)}/{call.get('max_retries', 1)}",
                "",
                f"--- SYSTEM PROMPT {call_id} ---",
                str(call.get("system_prompt") or "(none)"),
                "",
                f"--- USER PROMPT {call_id} ---",
                str(call.get("user_prompt") or "(none)"),
                "",
                f"--- REASONING {call_id} ---",
                str(call.get("reasoning") or "(none)"),
            ]
        )
        if call.get("error"):
            lines.extend(["", f"error: {call['error']}"])
        lines.extend(["", "-" * 80, ""])
    return "\n".join(lines).strip()


def render_task_log(*, task_id: str | None, created_at: str, payload: dict[str, Any]) -> str:
    lines = [
        f"task_id: {task_id or 'unknown_task'}",
        f"question_id: {payload.get('question_id') or ''}",
        f"method: {payload.get('method') or ''}",
        f"benchmark: {payload.get('benchmark') or 'clarifycodebench'}",
        f"task_type: {payload.get('task_type') or ''}",
        f"created_at: {created_at}",
    ]
    text = "\n".join(lines) + "\n"
    text += _section(
        "ORIGINAL UNDERSPECIFIED PROBLEM DESCRIPTION",
        _format_value(payload.get("input_problem")),
    )
    text += _section(
        "MISSING GROUND TRUTH",
        _format_ground_truth(payload.get("qa_pairs", []) or []),
    )

    flattened_trajectory = _render_iterative_trajectory(payload)
    if flattened_trajectory:
        text += _section("ITERATIVE TRAJECTORY (FLATTENED)", flattened_trajectory)

    record = payload.get("record") or {}
    rounds = record.get("rounds") or (payload.get("steps") or {}).get("rounds") or []
    has_accepted_clarifications = any(
        round_data.get("accepted_clarifications") for round_data in rounds
    )
    steps = payload.get("steps", {}) or {}
    for step_name, step_output in steps.items():
        if step_name == "rounds":
            continue
        if step_name == "final_specification":
            title = (
                "FINAL AUGMENTED SPECIFICATION (ORIGINAL + ACCEPTED CLARIFICATIONS)"
                if has_accepted_clarifications
                else "FINAL ORIGINAL SPECIFICATION (NO ACCEPTED CLARIFICATION)"
            )
            text += _section(title, _format_value(step_output))
            continue
        title = f"STEP: {str(step_name).replace('_', ' ').upper()}"
        text += _section(title, _format_value(step_output))

    text += _section(
        "LLM INTERACTIONS (PROMPTS AND REASONING)",
        _render_llm_interactions(payload.get("llm_call_logs", []) or []),
    )
    text += _section("QUESTION METRICS", _format_value(payload.get("question_metrics")))
    if payload.get("benchmark") in {"clarifycodebench", "clarifycodebench_fullinfo"}:
        value = payload.get("pass_at_1", record.get("pass_at_1"))
        if value is not None:
            text += _section("PASS@1", _format_value(value))
    if "pass_evaluation" in payload:
        text += _section("PASS EVALUATION", _format_value(payload.get("pass_evaluation")))

    return text


def log_step(method: str, task_id: str | None, step: str, status: str) -> None:
    print(f"[{method}][{task_id or 'unknown_task'}][{step}] {status}", flush=True)


def _escape_invalid_json_backslashes(text: str) -> str:
    """Escape invalid backslashes without changing valid JSON escapes.

    Models often put LaTeX (for example ``\\geq`` or ``\\circ``) in a JSON
    string.  A regex cannot reliably distinguish an already escaped ``\\\\``
    from the second character of that escape, so it can accidentally turn valid
    JSON into invalid JSON.  Scan the text and preserve every JSON escape as-is;
    only a backslash followed by a non-JSON escape is doubled.
    """
    valid_single = '"\\/bfnrt'
    output: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char != "\\":
            output.append(char)
            index += 1
            continue

        if index + 1 >= len(text):
            output.append("\\\\")
            index += 1
            continue

        next_char = text[index + 1]
        if next_char in valid_single:
            output.extend(("\\", next_char))
            index += 2
            continue
        if next_char == "u" and index + 5 < len(text):
            digits = text[index + 2 : index + 6]
            if re.fullmatch(r"[0-9a-fA-F]{4}", digits):
                output.extend(("\\", "u", digits))
                index += 6
                continue
        if next_char == "\\":
            output.extend(("\\", "\\"))
            index += 2
            continue

        output.extend(("\\", "\\"))
        index += 1
    return "".join(output)


def _escape_json_control_characters(text: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                output.append(char)
                escaped = False
            elif char == "\\":
                output.append(char)
                escaped = True
            elif char == '"':
                output.append(char)
                in_string = False
            elif char == "\n":
                output.append("\\n")
            elif char == "\r":
                output.append("\\r")
            elif char == "\t":
                output.append("\\t")
            else:
                output.append(char)
        else:
            output.append(char)
            if char == '"':
                in_string = True
    return "".join(output)


def _close_json_containers(text: str) -> str:
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char == "]" and stack and stack[-1] == "[":
            stack.pop()
        elif char == "}" and stack and stack[-1] == "{":
            stack.pop()
    if in_string or not stack:
        return text
    return text + "".join("}" if item == "{" else "]" for item in reversed(stack))


def _json_candidates(text: str) -> list[str]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    repaired = _escape_json_control_characters(_escape_invalid_json_backslashes(raw))
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    candidates = [repaired]
    if raw != repaired:
        candidates.append(raw)
    candidates.extend(
        closed
        for closed in [_close_json_containers(candidate) for candidate in candidates]
        if closed not in candidates
    )
    return candidates


def _parse_json_values(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    values: list[Any] = []
    seen: set[str] = set()
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
            values.append(parsed)
        except json.JSONDecodeError:
            pass

        for match in re.finditer(r"[\[{]", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            try:
                marker = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
            except (TypeError, ValueError):
                marker = repr(parsed)
            if marker not in seen:
                seen.add(marker)
                values.append(parsed)
    return values


def parse_json_object(
    text: str, *, required_keys: tuple[str, ...] = ()
) -> dict[str, Any] | None:
    for parsed in _parse_json_values(text):
        if isinstance(parsed, dict) and all(key in parsed for key in required_keys):
            return parsed
    return None


def extract_questions_from_output(method: str, output_text: str) -> list[str]:
    questions: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            questions.append(value.strip())
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            if isinstance(value.get("clarifying_questions"), list):
                collect(value["clarifying_questions"])
            for field in ("clarifying_question", "question"):
                if isinstance(value.get(field), str):
                    questions.append(value[field].strip())
            if method == "divergence" and isinstance(value.get("contradictions"), list):
                collect(value["contradictions"])

    for value in _parse_json_values(output_text):
        collect(value)

    if not questions:
        for line in (output_text or "").splitlines():
            cleaned = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
            if cleaned.endswith("?"):
                questions.append(cleaned)

    unique: list[str] = []
    seen: set[str] = set()
    for question in questions:
        cleaned = " ".join(question.split()).strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            unique.append(cleaned)
    return unique


def _judge_question_match(
    client: OpenAI | AzureResponsesClient,
    *,
    model: str,
    candidate_question: str,
    key_question: str,
    consistency: int,
    max_tokens: int | None,
    max_retries: int,
    benchmark: str = "clarifycodebench",
) -> bool:
    prompt_template = HIL_BLOCKER_JUDGE_PROMPT if benchmark == "hil-bench" else JUDGE_PROMPT
    prompt = prompt_template.format(
        key_question=key_question,
        model_question=candidate_question,
    )
    yes = no = 0
    for _ in range(consistency):
        verdict = chat_text(
            client,
            model=model,
            system_prompt="Return only yes or no.",
            user_prompt=prompt,
            temperature=1.0,
            max_tokens=max_tokens,
            max_retries=max_retries,
            step_name="question_judge",
            response_format="yes_no",
        ).strip().lower()
        verdict = extract_tag(verdict, "ANSWER").strip().lower()
        if "yes" in verdict:
            yes += 1
        elif "no" in verdict:
            no += 1
    return yes > no


def evaluate_questions(
    client: OpenAI | AzureResponsesClient,
    *,
    judge_model: str,
    generated_questions: list[str],
    qa_pairs: list[dict[str, Any]],
    consistency: int,
    max_tokens: int | None,
    max_retries: int,
    question_metadata: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if question_metadata is not None and len(question_metadata) != len(
        generated_questions
    ):
        raise ValueError("question_metadata must align with generated_questions")
    benchmark = "hil-bench" if any(qa.get("blocker_id") for qa in qa_pairs) else "clarifycodebench"
    matched_keys: set[str] = set()
    evaluations: list[dict[str, Any]] = []

    inherited_context = current_llm_call_context()
    for question_index, question in enumerate(generated_questions, start=1):
        metadata = (
            dict(question_metadata[question_index - 1])
            if question_metadata is not None
            else {}
        )
        batch_id = inherited_context.get("batch_id")
        global_batch_id = inherited_context.get("global_batch_id")
        question_id = (
            str(metadata.get("question_id"))
            if metadata.get("question_id") is not None
            else f"{batch_id}_{question_index}" if batch_id else f"Q_{question_index}"
        )
        global_question_id = (
            str(metadata.get("global_question_id"))
            if metadata.get("global_question_id") is not None
            else f"{global_batch_id}_{question_index}"
            if global_batch_id
            else question_id
        )
        context_metadata = {
            key: value
            for key, value in metadata.items()
            if key not in {"question_id", "global_question_id", "ground_truth_id"}
        }
        matched_key = None
        matched_key_id = None
        is_duplicate = False
        for ground_truth_index, qa in enumerate(qa_pairs, start=1):
            key_question = str(qa.get("question", "")).strip()
            if not key_question:
                continue
            ground_truth_id = str(
                qa.get("blocker_id")
                or qa.get("question_id")
                or f"GT_{ground_truth_index}"
            )
            judge_key_question = key_question
            if qa.get("blocker_id"):
                examples = qa.get("example_questions") or [key_question]
                judge_key_question = "\n".join(
                    [
                        f"Blocker ID: {qa['blocker_id']}",
                        f"Description: {qa.get('description', '')}",
                        "Example questions:",
                        *[str(example) for example in examples],
                    ]
                )
            with llm_call_context(
                **context_metadata,
                question_id=question_id,
                global_question_id=global_question_id,
                ground_truth_id=ground_truth_id,
            ):
                matched = _judge_question_match(
                    client,
                    model=judge_model,
                    candidate_question=question,
                    key_question=judge_key_question,
                    consistency=consistency,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    benchmark=benchmark,
                )
            if matched:
                matched_key = key_question
                matched_key_id = ground_truth_id
                is_duplicate = key_question in matched_keys
                if not is_duplicate:
                    matched_keys.add(key_question)
                break

        evaluations.append(
            {
                "question": question,
                "correct": matched_key is not None and not is_duplicate,
                "duplicate": is_duplicate,
                "matched_key_question": matched_key,
                "matched_key_id": matched_key_id,
            }
        )

    total_predicted = len(generated_questions)
    correct = sum(1 for item in evaluations if item["correct"])
    total_key = len(qa_pairs)
    precision = correct / total_predicted if total_predicted else 0.0
    recall = correct / total_key if total_key else (1.0 if total_predicted == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "total_key_questions": total_key,
        "total_predicted_questions": total_predicted,
        "correct_questions": correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "evaluations": evaluations,
    }


def evaluate_pass_rate(
    *,
    pass_command: str | None,
    task_id: str | None,
    question_id: str | None,
    codes: list[str],
) -> dict[str, Any]:
    if not pass_command:
        return {
            "enabled": False,
            "pass": None,
            "avg_pass_rate": None,
            "pass_rate": None,
            "reason": "pass command was not provided",
            "results": [],
        }
    if not codes:
        return {
            "enabled": True,
            "pass": None,
            "avg_pass_rate": None,
            "pass_rate": None,
            "reason": "no generated code to evaluate",
            "results": [],
        }

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="clarify_pass_") as tmp:
        tmp_dir = Path(tmp)
        for idx, code in enumerate(codes, start=1):
            code_file = tmp_dir / f"{task_id or question_id or 'task'}_{idx}.py"
            code_file.write_text(code, encoding="utf-8")
            command = pass_command
            for placeholder, value in (
                ("{code_file}", str(code_file)),
                ("{task_id}", task_id or ""),
                ("{question_id}", question_id or ""),
                ("{sample_idx}", str(idx - 1)),
            ):
                command = command.replace(placeholder, value)
            proc = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
                timeout=300,
                check=False,
            )
            results.append(
                {
                    "sample_idx": idx - 1,
                    "code_file": str(code_file),
                    "command": command,
                    "passed": proc.returncode == 0,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout[-4000:],
                    "stderr": proc.stderr[-4000:],
                }
            )

    passed = sum(1 for item in results if item["passed"])
    pass_rate = passed / len(results) if results else None
    return {
        "enabled": True,
        "pass": bool(results) and passed == len(results),
        "avg_pass_rate": pass_rate,
        "pass_rate": pass_rate,
        "passed": passed,
        "total": len(results),
        "results": results,
    }


def skipped_pass_evaluation(reason: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "pass": None,
        "avg_pass_rate": None,
        "pass_rate": None,
        "reason": reason,
        "results": [],
    }


def summarize_question_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    samples = []
    total_key = 0
    total_predicted = 0
    total_correct = 0
    macro_precision_sum = 0.0
    macro_recall_sum = 0.0
    macro_f1_sum = 0.0
    macro_recall_ques_sum = 0.0
    macro_precision_ques_sum = 0.0
    macro_precision_level_sum = 0.0
    macro_ask_f1_ques_sum = 0.0
    question_level_scored_count = 0
    legacy_scored_count = 0
    total_covered_key_questions = 0
    total_generated_questions = 0
    total_correct_generated_questions = 0
    total_nonempty_batches = 0
    total_valid_nonempty_batches = 0
    total_tkqr = 0.0
    total_ora = 0.0
    interaction_metric_count = 0
    scored_count = 0
    skipped_count = 0

    for record in records:
        metrics = record.get("question_metrics")
        sample = {
            "task_id": record.get("task_id"),
            "question_id": record.get("question_id"),
            "method": record.get("method"),
            "benchmark": record.get("benchmark", "clarifycodebench"),
            "task_type": record.get("task_type"),
            "generated_question_count": len(record.get("generated_questions", []) or []),
            "scored": metrics is not None,
        }
        if metrics is None:
            skipped_count += 1
            sample["reason"] = "question metrics were skipped"
            samples.append(sample)
            continue

        scored_count += 1
        key = int(metrics.get("total_key_questions", 0))
        metric_unit = str(metrics.get("metric_unit", "individual_question"))
        has_legacy_metrics = all(
            field in metrics
            for field in (
                "total_predicted_questions",
                "correct_questions",
                "precision",
                "recall",
                "ask_f1",
            )
        )
        has_question_level_metrics = all(
            key in metrics
            for key in (
                "recall_ques",
                "precision_ques",
                "precision_level",
                "ask_f1_ques",
                "covered_key_questions",
                "total_generated_questions",
                "correct_generated_questions",
                "nonempty_batches",
                "valid_nonempty_batches",
            )
        )

        total_key += key
        sample.update({
            "total_key_questions": key,
            "metric_unit": metric_unit,
            "evaluations": metrics.get("evaluations", []),
        })
        if has_legacy_metrics:
            predicted = int(metrics["total_predicted_questions"])
            correct = int(metrics["correct_questions"])
            precision = float(metrics["precision"])
            recall = float(metrics["recall"])
            f1 = float(metrics["ask_f1"])
            total_predicted += predicted
            total_correct += correct
            macro_precision_sum += precision
            macro_recall_sum += recall
            macro_f1_sum += f1
            legacy_scored_count += 1
            sample.update(
                {
                    "total_predicted_questions": predicted,
                    "correct_questions": correct,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "ask_f1": f1,
                }
            )
        if has_question_level_metrics:
            recall_ques = float(metrics["recall_ques"])
            precision_ques = float(metrics["precision_ques"])
            precision_level = float(metrics["precision_level"])
            ask_f1_ques = float(metrics["ask_f1_ques"])
            covered_key_questions = int(metrics["covered_key_questions"])
            generated_questions = int(metrics["total_generated_questions"])
            correct_generated_questions = int(
                metrics["correct_generated_questions"]
            )
            nonempty_batches = int(metrics["nonempty_batches"])
            valid_nonempty_batches = int(metrics["valid_nonempty_batches"])
            sample.update(
                {
                    "recall_ques": recall_ques,
                    "precision_ques": precision_ques,
                    "precision_level": precision_level,
                    "ask_f1_ques": ask_f1_ques,
                    "covered_key_questions": covered_key_questions,
                    "total_generated_questions": generated_questions,
                    "correct_generated_questions": correct_generated_questions,
                    "nonempty_batches": nonempty_batches,
                    "valid_nonempty_batches": valid_nonempty_batches,
                }
            )
            macro_recall_ques_sum += recall_ques
            macro_precision_ques_sum += precision_ques
            macro_precision_level_sum += precision_level
            macro_ask_f1_ques_sum += ask_f1_ques
            question_level_scored_count += 1
            total_covered_key_questions += covered_key_questions
            total_generated_questions += generated_questions
            total_correct_generated_questions += correct_generated_questions
            total_nonempty_batches += nonempty_batches
            total_valid_nonempty_batches += valid_nonempty_batches
        if has_legacy_metrics and metric_unit == "Q_r_q_batch":
            sample["total_predicted_batches"] = predicted
            sample["valid_batches"] = correct
        for interaction_metric in ("TKQR", "ORA"):
            if interaction_metric in metrics:
                sample[interaction_metric] = float(metrics[interaction_metric])
        if "TKQR" in metrics and "ORA" in metrics:
            total_tkqr += float(metrics["TKQR"])
            total_ora += float(metrics["ORA"])
            interaction_metric_count += 1
        if sample["benchmark"] == "hil-bench":
            sample.update(
                {
                    "blockers_present": key,
                    "blockers_discovered": (
                        covered_key_questions
                        if has_question_level_metrics
                        else int(metrics.get("correct_questions", 0))
                    ),
                }
            )
        samples.append(sample)

    summary = {
        "sample_count": len(records),
        "benchmarks": sorted({record.get("benchmark", "clarifycodebench") for record in records}),
        "scored_sample_count": scored_count,
        "skipped_sample_count": skipped_count,
        "totals": {
            "key_questions": total_key,
        },
        "micro": {},
        "macro": {},
        "samples": samples,
    }
    if legacy_scored_count:
        micro_precision = total_correct / total_predicted if total_predicted else 0.0
        micro_recall = (
            total_correct / total_key
            if total_key
            else (1.0 if total_predicted == 0 else 0.0)
        )
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if micro_precision + micro_recall
            else 0.0
        )
        summary["totals"].update(
            {
                "predicted_questions": total_predicted,
                "correct_questions": total_correct,
            }
        )
        summary["micro"].update(
            {
                "precision": micro_precision,
                "recall": micro_recall,
                "f1": micro_f1,
                "ask_f1": micro_f1,
            }
        )
        summary["macro"].update(
            {
                "precision": macro_precision_sum / legacy_scored_count,
                "recall": macro_recall_sum / legacy_scored_count,
                "f1": macro_f1_sum / legacy_scored_count,
                "ask_f1": macro_f1_sum / legacy_scored_count,
            }
        )
    metric_units = {
        sample.get("metric_unit")
        for sample in samples
        if sample.get("scored") and sample.get("metric_unit")
    }
    if len(metric_units) == 1:
        summary["metric_unit"] = next(iter(metric_units))
    if legacy_scored_count and metric_units == {"Q_r_q_batch"}:
        summary["totals"]["generated_batches"] = total_predicted
        summary["totals"]["valid_batches"] = total_correct
    if question_level_scored_count:
        micro_recall_ques = (
            total_covered_key_questions / total_key if total_key else 0.0
        )
        micro_precision_ques = (
            total_correct_generated_questions / total_generated_questions
            if total_generated_questions
            else 0.0
        )
        micro_ask_f1_ques = (
            2
            * micro_precision_ques
            * micro_recall_ques
            / (micro_precision_ques + micro_recall_ques)
            if micro_precision_ques + micro_recall_ques
            else 0.0
        )
        summary["totals"].update(
            {
                "covered_key_questions": total_covered_key_questions,
                "generated_questions": total_generated_questions,
                "correct_generated_questions": total_correct_generated_questions,
                "nonempty_batches": total_nonempty_batches,
                "valid_nonempty_batches": total_valid_nonempty_batches,
            }
        )
        summary["micro"].update(
            {
                "recall_ques": micro_recall_ques,
                "precision_ques": micro_precision_ques,
                "precision_level": (
                    total_valid_nonempty_batches / total_nonempty_batches
                    if total_nonempty_batches
                    else 0.0
                ),
                "ask_f1_ques": micro_ask_f1_ques,
            }
        )
        summary["macro"].update(
            {
                "recall_ques": macro_recall_ques_sum / question_level_scored_count,
                "precision_ques": (
                    macro_precision_ques_sum / question_level_scored_count
                ),
                "precision_level": (
                    macro_precision_level_sum / question_level_scored_count
                ),
                "ask_f1_ques": (
                    macro_ask_f1_ques_sum / question_level_scored_count
                ),
            }
        )
    if interaction_metric_count:
        summary["TKQR"] = total_tkqr / interaction_metric_count
        summary["ORA"] = total_ora / interaction_metric_count
    if "hil-bench" in summary["benchmarks"]:
        if question_level_scored_count:
            summary["hil"] = {
                "blockers_present": total_key,
                "blockers_discovered": total_covered_key_questions,
                "recall_ques": summary["micro"]["recall_ques"],
                "precision_ques": summary["micro"]["precision_ques"],
                "ask_f1_ques": summary["micro"]["ask_f1_ques"],
                "precision_level": summary["micro"]["precision_level"],
            }
        elif legacy_scored_count:
            summary["hil"] = {
                "blockers_present": total_key,
                "blockers_discovered": total_correct,
                "precision": summary["micro"]["precision"],
                "recall": summary["micro"]["recall"],
                "f1": summary["micro"]["f1"],
            }
    return summary


def summarize_llm_call_usage(call_logs: list[dict[str, Any]]) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    by_step: dict[str, dict[str, int]] = {}
    by_model: dict[str, dict[str, int]] = {}
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    metered_call_count = 0

    def add_usage(bucket: dict[str, dict[str, int]], key: str, usage: dict[str, int]) -> None:
        values = bucket.setdefault(
            key,
            {
                "call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        )
        values["call_count"] += 1
        for field in totals:
            values[field] += usage[field]

    for log in call_logs:
        raw_usage = log.get("usage")
        usage = _normalise_token_usage(raw_usage)
        call = {
            key: log.get(key)
            for key in (
                "call_id",
                "specification_state",
                "state_id",
                "generation_id",
                "generation_operation",
                "parent_code_id",
                "finalization_id",
                "r",
                "reconstruction_id",
                "q",
                "code_id",
                "batch_id",
                "global_batch_id",
                "question_id",
                "global_question_id",
                "ground_truth_id",
                "phase",
                "attempt",
                "max_retries",
                "duration_ms",
            )
            if log.get(key) is not None
        }
        call.update({
            "step": str(log.get("step", "llm_call")),
            "model": str(log.get("model", "unknown")),
            "status": str(log.get("status", "completed")),
            "input_tokens": usage["input_tokens"] if usage else None,
            "output_tokens": usage["output_tokens"] if usage else None,
            "total_tokens": usage["total_tokens"] if usage else None,
        })
        calls.append(call)
        if usage is None:
            continue
        metered_call_count += 1
        for field in totals:
            totals[field] += usage[field]
        add_usage(by_step, call["step"], usage)
        add_usage(by_model, call["model"], usage)

    return _enrich_token_usage_costs({
        "usage_available": metered_call_count > 0,
        "call_count": len(calls),
        "metered_call_count": metered_call_count,
        **totals,
        "by_step": by_step,
        "by_model": by_model,
        "calls": calls,
    })


def _load_cost_config() -> dict[str, Any]:
    path = Path(os.environ.get("ECHOCODER_COST_CONFIG", _WORKSPACE_ROOT / "cost.yaml"))
    if not path.exists():
        return {
            "path": str(path),
            "sha256": None,
            "currency": "USD",
            "token_unit": 1_000_000,
            "models": {},
            "error": "cost configuration file does not exist",
        }
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]

        payload = yaml.safe_load(text)
    except ModuleNotFoundError:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return {
                "path": str(path),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "currency": "USD",
                "token_unit": 1_000_000,
                "models": {},
                "error": f"PyYAML is required for non-JSON YAML: {exc}",
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "path": str(path),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "currency": "USD",
            "token_unit": 1_000_000,
            "models": {},
            "error": str(exc),
        }
    if not isinstance(payload, dict):
        payload = {}
    models = payload.get("models")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "currency": str(payload.get("currency", "USD")),
        "token_unit": int(payload.get("token_unit", 1_000_000)),
        "models": models if isinstance(models, dict) else {},
        "error": None,
    }


def _price_usage(
    *, input_tokens: int, output_tokens: int, model: str, config: dict[str, Any]
) -> dict[str, Any]:
    raw_price = config["models"].get(model)
    if not isinstance(raw_price, dict):
        return {
            "cost_available": False,
            "input_price": None,
            "output_price": None,
            "input_cost": None,
            "output_cost": None,
            "total_cost": None,
        }
    input_price = raw_price.get("input_price")
    output_price = raw_price.get("output_price")
    if (
        isinstance(input_price, bool)
        or isinstance(output_price, bool)
        or not isinstance(input_price, (int, float))
        or not isinstance(output_price, (int, float))
        or input_price < 0
        or output_price < 0
    ):
        return {
            "cost_available": False,
            "input_price": input_price,
            "output_price": output_price,
            "input_cost": None,
            "output_cost": None,
            "total_cost": None,
        }
    token_unit = config["token_unit"]
    input_cost = input_tokens * float(input_price) / token_unit
    output_cost = output_tokens * float(output_price) / token_unit
    return {
        "cost_available": True,
        "input_price": float(input_price),
        "output_price": float(output_price),
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def _enrich_token_usage_costs(usage: dict[str, Any]) -> dict[str, Any]:
    config = _load_cost_config()
    calls = usage.get("calls", []) or []
    unpriced_models: set[str] = set()
    priced_call_count = 0
    for call in calls:
        if call.get("input_tokens") is None or call.get("output_tokens") is None:
            call.update(
                {
                    "cost_available": False,
                    "input_cost": None,
                    "output_cost": None,
                    "total_cost": None,
                }
            )
            continue
        cost = _price_usage(
            input_tokens=int(call["input_tokens"]),
            output_tokens=int(call["output_tokens"]),
            model=str(call.get("model", "unknown")),
            config=config,
        )
        call.update(cost)
        if cost["cost_available"]:
            priced_call_count += 1
        else:
            unpriced_models.add(str(call.get("model", "unknown")))

    by_model = usage.get("by_model", {}) or {}
    for model, bucket in by_model.items():
        cost = _price_usage(
            input_tokens=int(bucket.get("input_tokens", 0)),
            output_tokens=int(bucket.get("output_tokens", 0)),
            model=str(model),
            config=config,
        )
        bucket.update(cost)
        if not cost["cost_available"]:
            unpriced_models.add(str(model))

    by_step = usage.get("by_step", {}) or {}
    for step, bucket in by_step.items():
        step_calls = [
            call
            for call in calls
            if call.get("step") == step and call.get("input_tokens") is not None
        ]
        complete = bool(step_calls) and all(
            call.get("cost_available") for call in step_calls
        )
        bucket.update(
            {
                "cost_available": complete,
                "input_cost": (
                    sum(float(call["input_cost"]) for call in step_calls)
                    if complete
                    else None
                ),
                "output_cost": (
                    sum(float(call["output_cost"]) for call in step_calls)
                    if complete
                    else None
                ),
                "total_cost": (
                    sum(float(call["total_cost"]) for call in step_calls)
                    if complete
                    else None
                ),
            }
        )

    metered_call_count = int(usage.get("metered_call_count", 0))
    cost_available = metered_call_count > 0 and priced_call_count == metered_call_count
    model_costs = list(by_model.values())
    usage.update(
        {
            "cost_available": cost_available,
            "priced_call_count": priced_call_count,
            "unpriced_models": sorted(unpriced_models),
            "currency": config["currency"],
            "token_unit": config["token_unit"],
            "pricing_config": config["path"],
            "pricing_config_sha256": config["sha256"],
            "pricing_error": config["error"],
            "input_cost": (
                sum(float(item["input_cost"]) for item in model_costs)
                if cost_available
                else None
            ),
            "output_cost": (
                sum(float(item["output_cost"]) for item in model_costs)
                if cost_available
                else None
            ),
        }
    )
    usage["total_cost"] = (
        float(usage["input_cost"]) + float(usage["output_cost"])
        if cost_available
        else None
    )
    return usage


def cost_record_from_result(record: dict[str, Any]) -> dict[str, Any]:
    usage = _enrich_token_usage_costs(dict(record.get("token_usage") or {}))
    return {
        "task_id": record.get("task_id"),
        "question_id": record.get("question_id"),
        "method": record.get("method"),
        "benchmark": record.get("benchmark", "clarifycodebench"),
        "task_type": record.get("task_type"),
        "status": record.get("status", "completed"),
        "failed_step": record.get("failed_step"),
        "error_type": record.get("error_type"),
        "error": record.get("error"),
        "usage_available": bool(usage.get("usage_available")),
        "call_count": int(usage.get("call_count", 0)),
        "metered_call_count": int(usage.get("metered_call_count", 0)),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost_available": bool(usage.get("cost_available")),
        "priced_call_count": int(usage.get("priced_call_count", 0)),
        "unpriced_models": usage.get("unpriced_models", []),
        "currency": usage.get("currency"),
        "token_unit": usage.get("token_unit"),
        "pricing_config": usage.get("pricing_config"),
        "pricing_config_sha256": usage.get("pricing_config_sha256"),
        "pricing_error": usage.get("pricing_error"),
        "input_cost": usage.get("input_cost"),
        "output_cost": usage.get("output_cost"),
        "total_cost": usage.get("total_cost"),
        "by_step": usage.get("by_step", {}),
        "by_model": usage.get("by_model", {}),
        "calls": usage.get("calls", []),
    }


def summarize_cost_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    by_step: dict[str, dict[str, int]] = {}
    by_model: dict[str, dict[str, int]] = {}
    available_count = 0
    call_count = 0
    metered_call_count = 0
    status_counts: dict[str, int] = {}
    all_calls: list[dict[str, Any]] = []

    def merge_buckets(target: dict[str, dict[str, int]], source: dict[str, Any]) -> None:
        for key, raw_values in source.items():
            values = target.setdefault(
                str(key),
                {
                    "call_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            )
            for field in values:
                values[field] += int(raw_values.get(field, 0))

    for record in records:
        status = str(record.get("status", "completed"))
        status_counts[status] = status_counts.get(status, 0) + 1
        call_count += int(record.get("call_count", 0))
        metered_call_count += int(record.get("metered_call_count", 0))
        all_calls.extend(record.get("calls", []) or [])
        if not record.get("usage_available"):
            continue
        available_count += 1
        for field in totals:
            totals[field] += int(record.get(field, 0) or 0)
        merge_buckets(by_step, record.get("by_step", {}))
        merge_buckets(by_model, record.get("by_model", {}))

    enriched = _enrich_token_usage_costs(
        {
            "usage_available": metered_call_count > 0,
            "call_count": call_count,
            "metered_call_count": metered_call_count,
            **totals,
            "by_step": by_step,
            "by_model": by_model,
            "calls": all_calls,
        }
    )
    totals.update(
        {
            "input_cost": enriched["input_cost"],
            "output_cost": enriched["output_cost"],
            "total_cost": enriched["total_cost"],
            "currency": enriched["currency"],
        }
    )
    divisor = available_count or 1
    return {
        "sample_count": len(records),
        "usage_available_sample_count": available_count,
        "missing_usage_sample_count": len(records) - available_count,
        "call_count": call_count,
        "metered_call_count": metered_call_count,
        "priced_call_count": enriched["priced_call_count"],
        "cost_available": enriched["cost_available"],
        "unpriced_models": enriched["unpriced_models"],
        "currency": enriched["currency"],
        "token_unit": enriched["token_unit"],
        "pricing_config": enriched["pricing_config"],
        "pricing_config_sha256": enriched["pricing_config_sha256"],
        "pricing_error": enriched["pricing_error"],
        "status_counts": status_counts,
        "totals": totals,
        "average_per_metered_sample": {
            field: (
                totals[field] / divisor
                if available_count and isinstance(totals[field], (int, float))
                else None
            )
            for field in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "input_cost",
                "output_cost",
                "total_cost",
            )
        },
        "by_step": enriched["by_step"],
        "by_model": enriched["by_model"],
    }


_CLARIFY_RAW_PASS_KEYS = frozenset({
    "pass_evaluation", "pass_rate", "code_evaluation", "code_state_evaluations",
    "initial_code_evaluation", "final_code_evaluation", "initial_code_passed",
    "final_code_passed", "final_correct", "code_iterations", "code_correctness",
    "code_correctness_transition", "code_passed", "code_pass_rate",
    "initial_code_pass@1", "final_code_pass@1", "pass@1",
})


def clarify_pass_at_1(record: dict[str, Any]) -> float | None:
    if "pass_at_1" in record:
        value = record["pass_at_1"]
        return None if value is None else float(value)
    evaluation = record.get("pass_evaluation") or record.get("final_code_evaluation")
    if evaluation is None:
        submitted = (record.get("pass_rate") or {}).get("submitted_patch") or {}
        evaluation = submitted.get("evaluation")
        if isinstance(evaluation, dict) and "evaluation" in evaluation:
            evaluation = evaluation["evaluation"]
    if not isinstance(evaluation, dict) or not evaluation.get("enabled"):
        return None
    code_passed = evaluation.get("pass") is True
    if not code_passed:
        passed = evaluation.get("passed")
        total = evaluation.get("total")
        code_passed = isinstance(passed, int) and isinstance(total, int) and total > 0 and passed == total
    if record.get("benchmark") == "clarifycodebench_fullinfo":
        return float(code_passed)
    metrics = record.get("question_metrics") or (record.get("ask_result") or {}).get("metrics")
    if not isinstance(metrics, dict):
        return None
    covered = metrics.get("covered_key_questions")
    required = metrics.get("total_key_questions")
    if covered is None or required is None:
        return None
    return float(code_passed and int(covered) >= int(required))


def publish_clarify_artifact(value: Any) -> Any:
    def convert(item: Any, active: bool) -> Any:
        if isinstance(item, list):
            return [convert(child, active) for child in item]
        if not isinstance(item, dict):
            return item
        own_benchmark = item.get("benchmark") in {"clarifycodebench", "clarifycodebench_fullinfo"}
        active = active or own_benchmark
        has_pass_data = own_benchmark and (
            any(key in item for key in _CLARIFY_RAW_PASS_KEYS) or "pass_at_1" in item
        )
        result = {
            key: convert(child, active)
            for key, child in item.items()
            if not active or key not in _CLARIFY_RAW_PASS_KEYS
        }
        if has_pass_data:
            result["pass_at_1"] = clarify_pass_at_1(item)
        return result

    return convert(value, False)


def summarize_passrate_records(
    records: list[dict[str, Any]], *, benchmark: str | None = None
) -> dict[str, Any]:
    if benchmark in {"clarifycodebench", "clarifycodebench_fullinfo"} or (
        records and all(
            record.get("benchmark", "clarifycodebench").startswith("clarifycodebench")
            for record in records
        )
    ):
        samples = []
        for record in records:
            value = clarify_pass_at_1(record)
            samples.append({
                "task_id": record.get("task_id"),
                "question_id": record.get("question_id"),
                "method": record.get("method"),
                "benchmark": record.get("benchmark", "clarifycodebench"),
                "task_type": record.get("task_type"),
                "enabled": value is not None,
                "pass_at_1": value,
            })
        evaluated = [sample["pass_at_1"] for sample in samples if sample["enabled"]]
        passed = sum(evaluated)
        aggregate = passed / len(evaluated) if evaluated else None
        return {
            "sample_count": len(records),
            "evaluated_sample_count": len(evaluated),
            "skipped_sample_count": len(records) - len(evaluated),
            "totals": {"passed": passed, "evaluated": len(evaluated)},
            "micro": {"pass_at_1": aggregate},
            "macro": {"pass_at_1": aggregate},
            "samples": samples,
        }
    samples = []
    enabled_count = 0
    skipped_count = 0
    total_passed = 0
    total_evaluated = 0
    macro_sum = 0.0

    for record in records:
        evaluation = record.get("pass_evaluation")
        sample = {
            "task_id": record.get("task_id"),
            "question_id": record.get("question_id"),
            "method": record.get("method"),
            "benchmark": record.get("benchmark", "clarifycodebench"),
            "task_type": record.get("task_type"),
            "enabled": bool(evaluation and evaluation.get("enabled")),
        }
        if not evaluation or not evaluation.get("enabled"):
            skipped_count += 1
            sample["pass_rate"] = None
            sample["reason"] = (evaluation or {}).get("reason", "pass-rate was not evaluated")
            samples.append(sample)
            continue

        enabled_count += 1
        code_passed = int(evaluation.get("passed", 0))
        total = int(evaluation.get("total", 0))
        code_pass_rate = evaluation.get("pass_rate")
        passed = code_passed
        pass_rate = code_pass_rate
        if sample["benchmark"] == "clarifycodebench":
            question_metrics = record.get("question_metrics")
            if question_metrics is None:
                question_metrics = (record.get("ask_result") or {}).get("metrics")
            has_coverage = isinstance(question_metrics, dict) and all(
                key in question_metrics
                for key in ("covered_key_questions", "total_key_questions")
            )
            covered = (
                int(question_metrics["covered_key_questions"])
                if has_coverage
                else None
            )
            required = (
                int(question_metrics["total_key_questions"])
                if has_coverage
                else None
            )
            all_resolved = (
                covered is not None
                and required is not None
                and covered >= required
            )
            sample.update(
                {
                    "covered_key_questions": covered,
                    "total_key_questions": required,
                    "all_key_questions_resolved": all_resolved,
                    "code_passed": code_passed,
                    "code_pass_rate": code_pass_rate,
                }
            )
            if not all_resolved:
                passed = 0
                pass_rate = 0.0
        total_passed += passed
        total_evaluated += total
        if pass_rate is not None:
            macro_sum += float(pass_rate)
        sample.update(
            {
                "passed": passed,
                "total": total,
                "pass_rate": pass_rate,
                "results": evaluation.get("results", []),
            }
        )
        samples.append(sample)

    return {
        "sample_count": len(records),
        "evaluated_sample_count": enabled_count,
        "skipped_sample_count": skipped_count,
        "totals": {
            "passed": total_passed,
            "evaluated": total_evaluated,
        },
        "micro": {
            "pass_rate": total_passed / total_evaluated if total_evaluated else None,
        },
        "macro": {
            "pass_rate": macro_sum / enabled_count if enabled_count else None,
        },
        "samples": samples,
    }


def extract_tag(text: str, tag: str) -> str:
    pattern = re.compile(
        rf"\[{re.escape(tag)}\](.*?)(?:\[/{re.escape(tag)}\]|$)",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text or "")
    return match.group(1).strip() if match else (text or "").strip()


def _output_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct

    parts: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"} and isinstance(
                content.get("text"), str
            ):
                parts.append(content["text"])
    if parts:
        return "".join(parts)

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):
                    texts = [
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict)
                        and item.get("type") in {"output_text", "text"}
                        and isinstance(item.get("text"), str)
                    ]
                    if texts:
                        return "".join(texts)
            text = choice.get("text")
            if isinstance(text, str) and text.strip():
                return text
    raise RuntimeError("LLM response contained no output text")


def _reasoning_prompt(system_prompt: str, response_format: str) -> str:
    if response_format == "yes_no":
        suffix = """\

For logging, include a brief visible rationale, not hidden chain-of-thought.
Return exactly:

[REASONING]
<brief reason for the decision>
[/REASONING]

[ANSWER]
yes|no
[/ANSWER]
"""
    else:
        suffix = """\

For logging, include a brief visible rationale before the requested output.
Do not reveal hidden chain-of-thought; summarize only the key assumptions or
decision basis that are useful for debugging.

[REASONING]
<brief visible reasoning>
[/REASONING]
"""
    return f"{system_prompt.rstrip()}\n\n{suffix.strip()}"


def _usage_value(usage: Any, *names: str) -> int | None:
    for name in names:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _normalise_token_usage(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    input_tokens = _usage_value(usage, "input_tokens", "prompt_tokens")
    output_tokens = _usage_value(usage, "output_tokens", "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _record_llm_call(
    *,
    step_name: str,
    model: str,
    temperature: float,
    max_tokens: int | None,
    system_prompt: str,
    user_prompt: str,
    response: str,
    response_format: str,
    usage: Any = None,
    status: str = "completed",
    error: str | None = None,
    attempt: int = 1,
    max_retries: int = 1,
    duration_ms: float | None = None,
) -> None:
    call_log = _CURRENT_LLM_CALL_LOG.get()
    if call_log is None:
        return
    item = {
        "call_id": f"call_{len(call_log) + 1:04d}",
        **current_llm_call_context(),
        "step": step_name,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": response_format,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "reasoning": (
            extract_tag(response, "REASONING")
            if re.search(r"\[REASONING\]", response or "", flags=re.IGNORECASE)
            else ""
        ),
        "response": response,
        "status": status,
        "usage": _normalise_token_usage(usage),
        "attempt": attempt,
        "max_retries": max_retries,
        "duration_ms": duration_ms,
    }
    if error:
        item["error"] = error
    call_log.append(item)
    live = _CURRENT_LLM_LIVE_OUTPUT.get()
    if live is not None:
        with Path(live["calls_path"]).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        _write_live_llm_state(call_log)
    refresh = _CURRENT_LIVE_TEXT_REFRESH.get()
    if refresh is not None:
        try:
            refresh()
        except OSError as error:
            print(f"[log] realtime TXT update failed: {error}", flush=True)


def chat_text(
    client: OpenAI | AzureResponsesClient,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int | None = 4096,
    max_retries: int = 5,
    step_name: str = "llm_call",
    response_format: str = "default",
    include_visible_reasoning: bool = True,
) -> str:
    requested_system_prompt = system_prompt
    if include_visible_reasoning:
        system_prompt = _reasoning_prompt(system_prompt, response_format)
    if isinstance(client, AzureResponsesClient):
        for attempt in range(max_retries):
            started_at = time.perf_counter()
            usage = None
            try:
                response_data = client.responses_create(
                    model=model,
                    instructions=system_prompt,
                    input_text=user_prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                usage = response_data.get("usage")
                response = _output_text(response_data)
                _record_llm_call(
                    step_name=step_name,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=requested_system_prompt,
                    user_prompt=user_prompt,
                    response=response,
                    response_format=response_format,
                    usage=usage,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    duration_ms=(time.perf_counter() - started_at) * 1000,
                )
                return response
            except Exception as exc:  # noqa: BLE001
                _record_llm_call(
                    step_name=step_name,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=requested_system_prompt,
                    user_prompt=user_prompt,
                    response="",
                    response_format=response_format,
                    usage=usage,
                    status="failed",
                    error=str(exc),
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    duration_ms=(time.perf_counter() - started_at) * 1000,
                )
                print(f"[azure] attempt {attempt + 1}/{max_retries} failed: {exc}", flush=True)
                if attempt + 1 < max_retries:
                    delay = min(2**attempt, 30)
                    if "HTTP 429" in str(exc):
                        hint = re.search(r"Please wait\s+(\d+(?:\.\d+)?)\s+seconds?", str(exc), re.I)
                        if hint:
                            delay = max(delay, float(hint.group(1)))
                    time.sleep(delay)
        raise RuntimeError(f"Azure LLM call failed after {max_retries} retries")

    kwargs: dict[str, Any] = {"temperature": temperature, "stream": False}
    if max_tokens is not None:
        if "gpt" in model.lower():
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
    for attempt in range(max_retries):
        started_at = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **kwargs,
            )
            text = response.choices[0].message.content or ""
            _record_llm_call(
                step_name=step_name,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=requested_system_prompt,
                user_prompt=user_prompt,
                response=text,
                response_format=response_format,
                usage=getattr(response, "usage", None),
                attempt=attempt + 1,
                max_retries=max_retries,
                duration_ms=(time.perf_counter() - started_at) * 1000,
            )
            return text
        except Exception as exc:  # noqa: BLE001
            _record_llm_call(
                step_name=step_name,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=requested_system_prompt,
                user_prompt=user_prompt,
                response="",
                response_format=response_format,
                status="failed",
                error=str(exc),
                attempt=attempt + 1,
                max_retries=max_retries,
                duration_ms=(time.perf_counter() - started_at) * 1000,
            )
            print(f"[llm] attempt {attempt + 1}/{max_retries} failed: {exc}", flush=True)
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"LLM call failed after {max_retries} retries")


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    benchmark_choices: tuple[str, ...] = ("clarifycodebench", "hil-bench"),
    data_help: str | None = None,
) -> None:
    parser.add_argument(
        "--benchmark",
        "--dataset",
        default="clarifycodebench",
        choices=benchmark_choices,
        help="benchmark backend to run",
    )
    parser.add_argument(
        "--data",
        default=None,
        help=data_help or (
            "ClarifyCodeBench JSONL or HIL-Bench task directory; defaults to the "
            "selected benchmark under the workspace root"
        ),
    )
    parser.add_argument("--out", default=None, help="output JSONL; defaults to <method>/output/results.jsonl")
    parser.add_argument("--output-dir", default=None, help="method output directory; defaults to <method>/output")
    parser.add_argument("--model", required=True, help="OpenAI-compatible model id")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N tasks")
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        help=(
            "run only these task ids; accepts comma- or space-separated values "
            "such as task_8,task_14 or 8 14"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="number of tasks to run concurrently in each batch (default: 3)",
    )
    parser.add_argument("--max-tokens", type=int, default=4096, help="LLM output token budget")
    parser.add_argument("--max-retries", type=int, default=5, help="LLM retries per call")
    parser.add_argument("--question-judge-model", default=None, help="model used to judge generated questions")
    parser.add_argument("--question-consistency", type=int, default=3, help="judge samples per generated question")
    parser.add_argument("--skip-question-metrics", action="store_true", help="do not judge question correctness")
    parser.add_argument("--question-only", action="store_true", help="generate and evaluate questions only; skip pass-rate")
    parser.add_argument("--no-resume", action="store_true", help="ignore existing results and rerun from scratch")
    parser.add_argument(
        "--pass-command",
        default=None,
        help=(
            "optional external pass evaluator command. Placeholders: "
            "{code_file}, {task_id}, {question_id}, {sample_idx}. Exit 0 means pass."
        ),
    )
    parser.add_argument("--api-key", default=None, help="override LLM_API_KEY")
    parser.add_argument("--base-url", default=None, help="override LLM_BASE_URL")
    parser.add_argument(
        "--azure-deployment",
        default=None,
        help="override AZURE_OPENAI_DEPLOYMENT; defaults to --model",
    )


def client_from_args(args: argparse.Namespace) -> OpenAI | AzureResponsesClient:
    base_url = args.base_url or os.environ.get("LLM_BASE_URL")
    api_key = args.api_key or os.environ.get("LLM_API_KEY")
    hostname = (urlsplit(base_url).hostname or "").lower() if base_url else ""
    if hostname.endswith((".services.ai.azure.com", ".openai.azure.com")):
        if not api_key:
            raise RuntimeError("LLM_API_KEY is required")
        return AzureResponsesClient(
            endpoint=base_url,
            api_key=api_key,
            default_deployment=args.azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        )
    return make_client(api_key=api_key, base_url=base_url)
