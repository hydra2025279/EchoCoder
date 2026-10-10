from __future__ import annotations

import logging
from typing import Any

from baselines.common import chat_text, parse_json_object
from echocoder.schemas import ClarificationEvent, ReconstructionResult

LOGGER = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are analyzing a software requirement after confirmed clarifications.
Compare the current issue, the requirement reconstructed from an earlier patch,
and the confirmed question/answer pairs for this batch. The issue may contain
clarifications from earlier batches; do not analyze those again. For each pair
in this batch, identify one assumption made by the reconstruction that the answer
corrects. An omitted requirement may be described as an assumption that the
behavior was unconstrained. Do not claim
to have inspected code; no code is provided. Use the confirmed answer as the
authority. Be specific and concise.

Return JSON only:
{"assumption_updates": {"<question_id>": {"wrong_assumption": "...", "intended_behavior": "...", "difference": "..."}}}
Use each question's ID as its key. Return one entry for every question ID in
the current batch. Do not return entries for earlier clarifications.
"""


def _question_key(event: ClarificationEvent, index: int) -> str:
    if event.global_question_id:
        return event.global_question_id
    if event.question_id:
        return f"S{event.specification_round}_{event.question_id}"
    return (
        f"S{event.specification_round}_R{event.reconstruction_index}_"
        f"Q{event.question_call_index}_{index}"
    )


def _pairs_for_prompt(events: list[tuple[str, ClarificationEvent]]) -> str:
    return "\n".join(
        f"Question ID: {question_id}\nQuestion: {event.question}\nAnswer: {event.answer}"
        for question_id, event in events
    )


def _validated_updates(response: str, expected_ids: set[str]) -> dict[str, dict[str, str]]:
    data = parse_json_object(response, required_keys=("assumption_updates",))
    updates = data.get("assumption_updates") if data else None
    if not isinstance(updates, dict):
        return {}
    return {
        question_id: update
        for question_id, update in updates.items()
        if question_id in expected_ids
        and isinstance(update, dict)
        and all(
            isinstance(update.get(field), str) and update[field].strip()
            for field in ("wrong_assumption", "intended_behavior", "difference")
        )
    }


def analyze_accepted(
    client: Any,
    *,
    model: str,
    specification: str,
    reconstruction: ReconstructionResult,
    accepted: list[ClarificationEvent],
    max_tokens: int | None,
    max_retries: int,
) -> list[dict[str, str]]:
    current = [
        (_question_key(item, index), item)
        for index, item in enumerate(accepted, start=1)
    ]
    pairs = _pairs_for_prompt(current)
    points = "\n".join(f"- {point}" for point in reconstruction.added_points) or "- None"
    response = chat_text(
        client,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=(
            f"Current issue:\n{specification}\n\n"
            f"Reconstructed requirement:\n{reconstruction.description}\n\n"
            f"Added points:\n{points}\n\n"
            f"Confirmed clarifications for this batch:\n{pairs}"
        ),
        max_tokens=max_tokens,
        max_retries=max_retries,
        step_name="analyze_reconstruction_assumptions",
    )
    expected_ids = {question_id for question_id, _ in current}
    updates = _validated_updates(response, expected_ids)
    missing = [(question_id, event) for question_id, event in current if question_id not in updates]
    if missing:
        LOGGER.warning(
            "Assumption analysis omitted %d current question ID(s); retrying only those IDs",
            len(missing),
        )
        response = chat_text(
            client,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=(
                f"Reconstructed added points:\n{points}\n\n"
                f"Confirmed clarifications for these question IDs only:\n"
                f"{_pairs_for_prompt(missing)}"
            ),
            max_tokens=max_tokens,
            max_retries=max_retries,
            step_name="analyze_reconstruction_assumptions_retry",
        )
        updates.update(_validated_updates(response, expected_ids - updates.keys()))
    if len(updates) < len(expected_ids):
        LOGGER.warning(
            "Assumption analysis still lacks %d current question ID(s); keeping their "
            "confirmed Q/A without optional assumption notes",
            len(expected_ids) - len(updates),
        )
    result = []
    for question_id, event in current:
        update = updates.get(question_id)
        if update is None:
            continue
        result.append({
            "question_id": question_id,
            "question": event.question,
            "answer": event.answer,
            "wrong_assumption": update["wrong_assumption"].strip(),
            "intended_behavior": update["intended_behavior"].strip(),
            "difference": update["difference"].strip(),
        })
    return result


def append_analysis(specification: str, updates: list[dict[str, str]]) -> str:
    result = specification
    for update in updates:
        clarification = (
            f"- **Question:** {update['question'].strip()}\n"
            f"  **Answer:** {update['answer'].strip()}\n"
        )
        before, separator, after = result.rpartition(clarification)
        if not separator:
            raise RuntimeError("Confirmed answer missing from updated issue")
        result = (
            before + separator
            + f"  **Previous inferred assumption:** {update['wrong_assumption']}\n"
            + f"  **Intended behavior:** {update['intended_behavior']}\n"
            + f"  **Difference:** {update['difference']}\n"
            + after
        )
    return result
