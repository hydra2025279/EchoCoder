from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


OracleStatus = Literal["valid", "invalid"]
StopReason = Literal[
    "no_questions_from_detector",
    "no_valid_clarification_within_budget",
]


@dataclass(frozen=True)
class WorkflowConfig:
    
    reconstruction_budget: int = 3
    question_call_budget: int = 2
    semantic_stop_on_empty_questions: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("reconstruction_budget", self.reconstruction_budget),
            ("question_call_budget", self.question_call_budget),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")


@dataclass
class GenerationResult:
    candidate: str
    initial_questions: list[str] = field(default_factory=list)
    raw_response: str | None = None
    localized_context: dict[str, Any] | None = None


@dataclass
class ReconstructionResult:
    description: str
    added_points: list[str] = field(default_factory=list)


@dataclass
class OracleResult:
    status: OracleStatus
    answer: str | None = None
    matched_key_question: str | None = None
    evaluation: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionRecord:
    question: str
    oracle_status: OracleStatus
    answer: str | None = None
    matched_key_question: str | None = None
    oracle_evaluation: dict[str, Any] | None = None
    semantic_axis: str | None = None
    question_id: str | None = None
    global_question_id: str | None = None


@dataclass
class QuestionBatchRecord:
    question_call_index: int
    detector_response: str
    questions: list[QuestionRecord]
    prior_questions: list[str] = field(default_factory=list)
    prior_valid_clarifications: list[dict[str, str]] = field(default_factory=list)
    prior_invalid_questions: list[str] = field(default_factory=list)
    batch_id: str | None = None
    global_batch_id: str | None = None


@dataclass
class ReconstructionRecord:
    reconstruction_index: int
    reconstructed_description: str
    added_points: list[str]
    question_batches: list[QuestionBatchRecord] = field(default_factory=list)
    reconstruction_id: str | None = None


@dataclass
class SpecificationStateRecord:
    specification_round: int
    specification: str
    complete_code: str
    initial_questions: list[str]
    localized_context: dict[str, Any] | None
    reconstructions: list[ReconstructionRecord] = field(default_factory=list)
    accepted_clarifications: list["ClarificationEvent"] = field(default_factory=list)
    assumption_updates: list[dict[str, str]] = field(default_factory=list)
    state_id: str | None = None
    generation_id: str | None = None
    code_id: str | None = None
    parent_code_id: str | None = None
    generation_operation: str | None = None


@dataclass
class ClarificationEvent:
    specification_round: int
    reconstruction_index: int
    question_call_index: int
    question: str
    answer: str
    matched_key_question: str | None = None
    event_id: str | None = None
    state_id: str | None = None
    code_id: str | None = None
    reconstruction_id: str | None = None
    batch_id: str | None = None
    global_batch_id: str | None = None
    question_id: str | None = None
    global_question_id: str | None = None


@dataclass
class WorkflowResult:
    original_problem: str
    rounds: list[SpecificationStateRecord]
    clarification_history: list[ClarificationEvent]
    rejected_questions: list[str]
    stop_reason: StopReason
    final_specification: str
    final_solution: str
    total_oracle_questions: int
    total_question_calls: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
