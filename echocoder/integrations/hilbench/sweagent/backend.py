from __future__ import annotations

from pathlib import Path
from typing import Any

from echocoder.integrations.base import LocalizedContext
from echocoder.schemas import GenerationResult

from ..models import HilBenchTask
from ..shared.backend import HilBenchBackend
from .adapter import SweAgentRunConfig, SweAgentRunResult, SweAgentRunner


class SweAgentBackend(HilBenchBackend):
    engine = "sweagent"

    def __init__(
        self,
        task: HilBenchTask,
        *,
        artifact_root: Path,
        sweagent_config_path: Path,
        sweagent_model: str | None = None,
        sweagent_max_steps: int,
        sweagent_timeout: int,
        sweagent_cost_limit: float,
        sweagent_call_limit: int,
        progressive_edit: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(task, **kwargs)
        self.runner = SweAgentRunner(
            task,
            client=self.client,
            config=SweAgentRunConfig(
                model=sweagent_model or self.model,
                max_steps=sweagent_max_steps,
                timeout=sweagent_timeout,
                per_instance_cost_limit=sweagent_cost_limit,
                per_instance_call_limit=sweagent_call_limit,
                config_path=sweagent_config_path,
            ),
            artifact_root=artifact_root,
        )
        self.specification_round = -1
        self.progressive_edit = progressive_edit
        self.runs: list[SweAgentRunResult] = []

    def localize(
        self, specification: str, specification_round: int
    ) -> LocalizedContext:
        del specification
        self.specification_round = specification_round
        return LocalizedContext(
            metadata={
                "localizer": "upstream-sweagent-internal",
                "specification_round": specification_round,
            }
        )

    def generate_complete_candidate(
        self,
        specification: str,
        context: LocalizedContext,
        previous_candidate: str | None,
    ) -> GenerationResult:
        del context
        if self.specification_round < 0:
            raise RuntimeError("SWE-agent state must be initialized before generation")
        if self.progressive_edit and previous_candidate:
            run = self.runner.run(
                specification, self.specification_round,
                previous_patch=previous_candidate,
            )
        else:
            run = self.runner.run(specification, self.specification_round)
        self.runs.append(run)
        return GenerationResult(
            candidate=run.patch,
            initial_questions=[],
            raw_response=run.raw_response,
        )

    def final_artifacts(self) -> list[dict[str, Any]]:
        return [
            {
                "trajectory_path": item.trajectory_path,
                "prediction_path": item.prediction_path,
                "config_path": item.config_path,
                "exit_status": item.exit_status,
                "agent_steps": item.agent_steps,
                "model_stats": item.model_stats,
                "duration_ms": item.duration_ms,
                "parent_patch_path": item.parent_patch_path,
            }
            for item in self.runs
        ]
