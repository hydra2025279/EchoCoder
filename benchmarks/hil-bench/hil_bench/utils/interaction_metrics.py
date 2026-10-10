from __future__ import annotations

import math
from typing import Any


def calculate_tkqr(hit_sequence: list[int], total_key_questions: int) -> float:
    if not hit_sequence or total_key_questions <= 0:
        return 0.0
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hit_sequence))
    ideal_len = min(len(hit_sequence), total_key_questions)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_len))
    return dcg / idcg if idcg else 0.0


def calculate_ora(question_turns: int, total_key_questions: int) -> float:
    if question_turns < 0 or total_key_questions < 0:
        raise ValueError("question_turns and total_key_questions must be non-negative")
    sigma = 0.425 * total_key_questions if total_key_questions > 0 else 0.3
    return math.exp(
        -((question_turns - total_key_questions) ** 2) / (2.0 * sigma**2)
    )


def interaction_metrics(
    questions: list[dict[str, Any]], total_key_questions: int
) -> dict[str, Any]:
    covered: set[str] = set()
    hit_sequence: list[int] = []
    for question in questions:
        blocker_name = question.get("blocker_name")
        is_new_hit = bool(blocker_name) and str(blocker_name) not in covered
        hit_sequence.append(1 if is_new_hit else 0)
        if blocker_name:
            covered.add(str(blocker_name))

    question_turns = len(questions)
    tkqr = calculate_tkqr(hit_sequence + [0], total_key_questions)
    if total_key_questions == 0 and question_turns == 0:
        tkqr = 1.0
    return {
        "TKQR": tkqr,
        "ORA": calculate_ora(question_turns, total_key_questions),
        "question_turns": question_turns,
        "key_questions": total_key_questions,
        "diff": question_turns - total_key_questions,
        "hit_sequence": hit_sequence,
        "turn_unit": "ask_human_question",
        "final_answer_turn_included_in_tkqr": True,
    }
