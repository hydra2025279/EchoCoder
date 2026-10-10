from __future__ import annotations

import base64
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from baselines.common import append_external_llm_call, current_llm_call_context

from ..models import HilBenchTask


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SWE_AGENT_ROOT = REPOSITORY_ROOT / "benchmarks" / "hil-bench" / "SWE-agent"
HIL_BENCH_PYTHON = REPOSITORY_ROOT / "benchmarks" / "hil-bench" / ".venv" / "bin" / "python"
DEFAULT_CONFIG = REPOSITORY_ROOT / "benchmarks" / "hil-bench" / "configs" / "swe" / "default_swe_config.yaml"


@dataclass(frozen=True)
class SweAgentRunConfig:
    model: str
    max_steps: int = 200
    timeout: int = 21600
    per_instance_cost_limit: float = 100.0
    per_instance_call_limit: int = 0
    config_path: Path = DEFAULT_CONFIG


@dataclass
class SweAgentRunResult:
    patch: str
    raw_response: str
    exit_status: str | None
    trajectory_path: str
    prediction_path: str
    config_path: str
    agent_steps: int
    model_stats: dict[str, Any]
    duration_ms: float
    stdout_path: str
    stderr_path: str
    parent_patch_path: str | None = None


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _absolute_tool_paths(agent: dict[str, Any]) -> None:
    bundles = agent.get("tools", {}).get("bundles", [])
    for bundle in bundles:
        path = Path(str(bundle.get("path", "")))
        if path and not path.is_absolute():
            bundle["path"] = str((SWE_AGENT_ROOT / path).resolve())


def _azure_model_name(model: str) -> str:
    return model if "/" in model else f"azure/{model}"


def _sweagent_model_name(model: str) -> str:
    if "deepseek" in model.lower() and "/" not in model:
        return f"openai/{model}"
    return _azure_model_name(model)


def _created_paths(patch: str) -> list[str]:
    """Find paths added by a git patch so baseline untracked copies can be cleared."""
    paths: list[str] = []
    path: str | None = None
    for line in patch.splitlines():
        if line.startswith("diff --git a/"):
            path = None
            match = re.fullmatch(r"diff --git a/(.+) b/\1", line)
            if match:
                candidate = match.group(1)
                parts = candidate.split("/")
                if (
                    candidate
                    and all(part not in ("", ".", "..") for part in parts)
                    and ".git" not in parts
                ):
                    path = candidate
        elif line.startswith("new file mode ") and path is not None:
            paths.append(path)
    return paths


class SweAgentRunner:
    def __init__(
        self,
        task: HilBenchTask,
        *,
        client: Any,
        config: SweAgentRunConfig,
        artifact_root: Path,
    ) -> None:
        self.task = task
        self.client = client
        self.config = config
        self.artifact_root = artifact_root

    def _render_config(
        self, specification: str, specification_round: int, state_root: Path,
        previous_patch: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "SWE-agent requires PyYAML; install the project with pip install -e '.[hilbench]'"
            ) from exc
        source = yaml.safe_load(self.config.config_path.read_text(encoding="utf-8"))
        agent = dict(source["agent"])
        _absolute_tool_paths(agent)
        model = dict(agent.get("model", {}))
        model.update(
            {
                "name": _sweagent_model_name(self.config.model),
                "api_key": "$LLM_API_KEY",
                "temperature": None,
                "per_instance_cost_limit": self.config.per_instance_cost_limit,
                "per_instance_call_limit": self.config.per_instance_call_limit,
                "total_cost_limit": 0.0,
                "enable_model_call_logging": True,
            }
        )
        endpoint = getattr(self.client, "base_url", None)
        if "deepseek" in self.config.model.lower():
            openai_v1_bases = getattr(self.client, "openai_v1_bases", None)
            if openai_v1_bases:
                endpoint = openai_v1_bases[0]
        if endpoint:
            model["api_base"] = str(endpoint)
        api_version = os.environ.get("AZURE_API_VERSION", "").strip()
        if api_version:
            model["api_version"] = api_version
        agent["model"] = model
        agent["max_steps"] = self.config.max_steps
        repo_name = str(self.task.metadata.get("repo_name", "app")).strip() or "app"
        instance_id = _safe_id(f"{self.task.task_id}_state_{specification_round}")
        if previous_patch:
            specification += (
                "\n\n## IMPLEMENTATION CONTEXT\n"
                "The previous candidate patch is already applied in this checkout. "
                "Inspect its assumptions against the confirmed clarifications and "
                "edit the existing implementation. Submit the complete resulting patch."
            )
        rendered = {
            "agent": agent,
            "env": {
                "repo": {
                    "type": "preexisting",
                    "repo_name": repo_name,
                    "base_commit": str(self.task.metadata.get("base_commit", "HEAD")),
                    "reset": False,
                },
                "deployment": {
                    "type": "docker",
                    "image": str(self.task.metadata.get("image_name", "")),
                    "python_standalone_dir": "/root",
                    "docker_args": ["--entrypoint="],
                },
            },
            "problem_statement": {
                "type": "text",
                "id": instance_id,
                "text": specification,
            },
            "output_dir": str(state_root),
        }
        if previous_patch:
            # The submitted patch is stripped when read from the trajectory.
            # git apply needs the final hunk line to end with a newline.
            patch_to_apply = previous_patch if previous_patch.endswith("\n") else previous_patch + "\n"
            encoded = base64.b64encode(patch_to_apply.encode("utf-8")).decode("ascii")
            patch_file = f"/tmp/echocoder_parent_{instance_id}.b64"
            commands = rendered["env"].setdefault("post_startup_commands", [])
            for path in _created_paths(previous_patch):
                # Some HiL images contain untracked source files. SWE-agent includes
                # them as additions in C0, so clear only those paths before replay.
                commands.append(
                    f"git -C {shlex.quote('/' + repo_name)} clean -f -- {shlex.quote(path)}"
                )
            for offset in range(0, len(encoded), 12000):
                chunk = encoded[offset:offset + 12000]
                commands.append(f"printf '%s' '{chunk}' >> {patch_file}")
            commands.append(
                f"git -C {shlex.quote('/' + repo_name)} update-index --refresh && "
                f"base64 -d < {patch_file} | git -C {shlex.quote('/' + repo_name)} apply --index -"
            )
        return rendered, instance_id

    def _command(self, config_path: Path) -> list[str]:
        interpreter = Path(os.environ.get("ECHOCODER_SWEAGENT_PYTHON", str(HIL_BENCH_PYTHON))).expanduser()
        if not interpreter.is_file():
            raise FileNotFoundError(
                f"SWE-agent Python not found at {interpreter}; run 'cd benchmarks/hil-bench && uv sync' "
                "or set ECHOCODER_SWEAGENT_PYTHON"
            )
        return [
            str(interpreter),
            "-m",
            "sweagent.run.run",
            "run",
            "--config",
            str(config_path),
        ]

    def _record_calls(
        self,
        trajectory: list[dict[str, Any]],
        model_stats: dict[str, Any],
        specification_round: int,
        duration_ms: float,
    ) -> None:
        context = current_llm_call_context()
        total_input = int(model_stats.get("tokens_sent", 0) or 0)
        total_output = int(model_stats.get("tokens_received", 0) or 0)
        for index, step in enumerate(trajectory, start=1):
            messages = list(step.get("query") or [])
            system_prompt = "\n\n".join(
                str(item.get("content", ""))
                for item in messages
                if item.get("role") == "system"
            )
            user_prompt = "\n\n".join(
                str(item.get("content", ""))
                for item in messages
                if item.get("role") != "system"
            )
            usage = {
                "input_tokens": total_input if index == len(trajectory) else 0,
                "output_tokens": total_output if index == len(trajectory) else 0,
                "total_tokens": (
                    total_input + total_output if index == len(trajectory) else 0
                ),
            }
            append_external_llm_call(
                {
                    "call_id": f"sweagent_state_{specification_round}_step_{index}",
                    "phase": "sweagent_agent",
                    "step": f"sweagent.step_{index}",
                    "model": self.config.model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "messages": messages,
                    "response": step.get("response", ""),
                    "reasoning": step.get("thought", ""),
                    "tool_action": step.get("action", ""),
                    "tool_observation": step.get("observation", ""),
                    "status": "completed",
                    "usage": usage,
                    "duration_ms": duration_ms if index == len(trajectory) else 0.0,
                    **context,
                }
            )

    def run(
        self, specification: str, specification_round: int,
        previous_patch: str | None = None,
    ) -> SweAgentRunResult:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "SWE-agent requires PyYAML; install the project with pip install -e '.[hilbench]'"
            ) from exc
        state_root = self.artifact_root / f"state_{specification_round}"
        state_root.mkdir(parents=True, exist_ok=True)
        rendered, instance_id = self._render_config(
            specification, specification_round, state_root, previous_patch
        )
        parent_patch_path = None
        if previous_patch:
            parent_patch = state_root / "parent.diff"
            parent_patch.write_text(previous_patch.rstrip() + "\n", encoding="utf-8")
            parent_patch_path = str(parent_patch)
        config_path = state_root / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(rendered, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        stdout_path = state_root / "stdout.log"
        stderr_path = state_root / "stderr.log"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                [str(SWE_AGENT_ROOT), environment.get("PYTHONPATH", "")],
            )
        )
        environment.setdefault("LITELLM_USER", "echocoder-sweagent")
        started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(
                self._command(config_path),
                cwd=SWE_AGENT_ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=self.config.timeout,
                check=False,
            )
        duration_ms = (time.perf_counter() - started) * 1000
        instance_root = state_root / instance_id
        trajectory_path = instance_root / f"{instance_id}.traj"
        prediction_path = instance_root / f"{instance_id}.pred"
        if not trajectory_path.exists():
            detail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(
                f"SWE-agent exited with code {completed.returncode} without a trajectory: {detail}"
            )
        data = json.loads(trajectory_path.read_text(encoding="utf-8"))
        info = dict(data.get("info") or {})
        trajectory = list(data.get("trajectory") or [])
        model_stats = dict(info.get("model_stats") or {})
        self._record_calls(
            trajectory,
            model_stats,
            specification_round,
            duration_ms,
        )
        if model_stats.get("api_calls") == 0:
            raise RuntimeError(
                "SWE-agent submitted without making a model call "
                f"(exit_status={info.get('exit_status')}); ignoring the checkout patch"
            )
        patch = str(info.get("submission") or "")
        if not patch and prediction_path.exists():
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            patch = str(prediction.get("model_patch") or "")
        if "diff --git " not in patch:
            raise RuntimeError(
                f"SWE-agent did not submit a git-compatible patch (exit_status={info.get('exit_status')})"
            )
        if not patch.endswith("\n"):
            patch += "\n"
        result = SweAgentRunResult(
            patch=patch,
            raw_response=str(trajectory[-1].get("response", "")) if trajectory else "",
            exit_status=info.get("exit_status"),
            trajectory_path=str(trajectory_path),
            prediction_path=str(prediction_path),
            config_path=str(config_path),
            agent_steps=len(trajectory),
            model_stats=model_stats,
            duration_ms=duration_ms,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            parent_patch_path=parent_patch_path,
        )
        (state_root / "metadata.json").write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (state_root / "submission.diff").write_text(patch, encoding="utf-8")
        return result
