from __future__ import annotations

from typing import Any

from echocoder.clarification import generate_questions
from echocoder.generation import generate_complete_candidate
from echocoder.integrations.base import LocalizedContext, Mode
from echocoder.iterative.oracle import query_clarification_oracle
from echocoder.reconstruction import reconstruct_requirements
from echocoder.schemas import GenerationResult, OracleResult, ReconstructionResult

from ..models import HilBenchTask


def augment_specification(specification: str, question: str, answer: str) -> str:
    heading = "## CLARIFICATIONS"
    separator = "\n\n" + heading + "\n" if heading not in specification else "\n"
    return (
        f"{specification.rstrip()}{separator}"
        f"- **Question:** {question.strip()}\n"
        f"  **Answer:** {answer.strip()}\n"
    )


class HilBenchBackend:
    engine = "echocoder"

    def __init__(
        self,
        task: HilBenchTask,
        *,
        client: Any,
        model: str,
        judge_model: str,
        max_tokens: int | None,
        max_retries: int,
        question_consistency: int,
    ) -> None:
        self.task = task
        self.client = client
        self.model = model
        self.judge_model = judge_model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.question_consistency = question_consistency

    def _artifact_instruction(self) -> str:
        if self.task.task_type == "sql":
            return "Return one read-only SQLite SELECT query in the [CODE] block."
        return "Return a complete git-compatible unified diff in the [CODE] block. Use paths relative to the repository root."

    def _generation_specification(
        self,
        specification: str,
        context: LocalizedContext,
        previous_candidate: str | None,
    ) -> str:
        parts = [
            specification.rstrip(),
            "\nEnvironment context available to the implementer:\n" + context.content,
            self._artifact_instruction(),
        ]
        if previous_candidate:
            parts.append(
                "Previous complete candidate to replace or improve:\n```\n"
                + previous_candidate
                + "\n```"
            )
        return "\n\n".join(parts)

    def generate_complete_candidate(
        self,
        specification: str,
        context: LocalizedContext,
        previous_candidate: str | None,
    ) -> GenerationResult:
        return generate_complete_candidate(
            self.client,
            model=self.model,
            specification=self._generation_specification(
                specification, context, previous_candidate
            ),
            task_type=self.task.task_type,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
        )

    def reconstruct_requirements(
        self,
        specification: str,
        context: LocalizedContext,
        candidate: str,
    ) -> ReconstructionResult:
        return reconstruct_requirements(
            self.client,
            model=self.model,
            specification=specification,
            solution=candidate,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            prompt_profile="hilbench",
        )

    def generate_questions(
        self,
        specification: str,
        context: LocalizedContext,
        candidate: str,
        reconstruction: ReconstructionResult,
        initial_questions: list[str],
        previous_valid_clarifications: list[dict[str, str]],
        previous_invalid_questions: list[str],
        mode: Mode,
    ) -> tuple[str, list[str]]:
        del context, mode
        return generate_questions(
            self.client,
            model=self.model,
            specification=specification,
            solution=candidate,
            reconstruction=reconstruction,
            initial_questions=initial_questions,
            previous_valid_clarifications=previous_valid_clarifications,
            previous_invalid_questions=previous_invalid_questions,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            prompt_profile="hilbench",
        )

    def ask_human(self, question: str) -> OracleResult:
        return OracleResult(
            **query_clarification_oracle(
                self.client,
                question=question,
                qa_pairs=self.task.qa_pairs,
                judge_model=self.judge_model,
                consistency=self.question_consistency,
                max_tokens=self.max_tokens,
                max_retries=self.max_retries,
            )
        )

    def update_specification(
        self, specification: str, question: str, answer: str
    ) -> str:
        return augment_specification(specification, question, answer)

    def close(self) -> None:
        return None
