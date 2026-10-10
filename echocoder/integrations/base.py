
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

from echocoder.core import AnalyzeAccepted, AppendAnalysis, Snapshot, run_specification_completion
from echocoder.schemas import GenerationResult, OracleResult, ReconstructionResult, WorkflowConfig, WorkflowResult


class Mode(str, Enum):
    ECHOCODER_SWEAGENT = "echocoder_sweagent"


@dataclass
class LocalizedContext:
    files: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    schema_items: list[str] = field(default_factory=list)
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class EchoCoderBackend(Protocol):
    def localize(self, specification: str, specification_round: int) -> LocalizedContext: ...
    def generate_complete_candidate(self, specification: str, context: LocalizedContext, previous_candidate: str | None) -> GenerationResult: ...
    def reconstruct_requirements(self, specification: str, context: LocalizedContext, candidate: str) -> ReconstructionResult: ...
    def generate_questions(self, specification: str, context: LocalizedContext, candidate: str, reconstruction: ReconstructionResult, initial_questions: list[str], previous_valid_clarifications: list[dict[str, str]], previous_invalid_questions: list[str], mode: Mode) -> tuple[str, list[str]]: ...
    def ask_human(self, question: str) -> OracleResult: ...
    def update_specification(self, specification: str, question: str, answer: str) -> str: ...


@dataclass(frozen=True)
class IntegrationConfig:
    mode: Mode = Mode.ECHOCODER_SWEAGENT
    reconstruction_budget: int = 3
    question_call_budget: int = 2
    semantic_stop_on_empty_questions: bool = False


def run_integrated_echocoder(
    *,
    problem: str,
    backend: EchoCoderBackend,
    config: IntegrationConfig,
    progress: Callable[[int, int | None, int | None, str, str], None] | None = None,
    snapshot: Snapshot | None = None,
    analyze_accepted: AnalyzeAccepted | None = None,
    append_analysis: AppendAnalysis | None = None,
) -> WorkflowResult:
    active_context: LocalizedContext | None = None

    def generate(
        specification: str,
        previous: str | None,
        round_index: int,
        previous_valid_clarifications: list[dict[str, str]],
        previous_invalid_questions: list[str],
    ) -> GenerationResult:
        nonlocal active_context
        del previous_valid_clarifications, previous_invalid_questions
        active_context = backend.localize(specification, round_index)
        result = backend.generate_complete_candidate(specification, active_context, previous)
        result.localized_context = asdict(active_context)
        return result

    def reconstruct(specification: str, candidate: str) -> ReconstructionResult:
        if active_context is None:
            raise RuntimeError("Localization must complete before reconstruction")
        return backend.reconstruct_requirements(specification, active_context, candidate)

    def detect(specification: str, candidate: str, reconstruction: ReconstructionResult, initial: list[str], valid: list[dict[str, str]], invalid: list[str]) -> tuple[str, list[str]]:
        if active_context is None:
            raise RuntimeError("Localization must complete before clarification")
        return backend.generate_questions(specification, active_context, candidate, reconstruction, initial, valid, invalid, config.mode)

    return run_specification_completion(
        problem=problem,
        generate=generate,
        reconstruct=reconstruct,
        detect=detect,
        ask_oracle_batch=lambda questions: [
            backend.ask_human(question) for question in questions
        ],
        update_specification=backend.update_specification,
        config=WorkflowConfig(
            reconstruction_budget=config.reconstruction_budget,
            question_call_budget=config.question_call_budget,
            semantic_stop_on_empty_questions=config.semantic_stop_on_empty_questions,
        ),
        progress=progress,
        snapshot=snapshot,
        analyze_accepted=analyze_accepted,
        append_analysis=append_analysis,
    )
