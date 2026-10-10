
DIRECT_QUESTIONS_SYSTEM_PROMPT = """\
You are given an underspecified programming problem.

Directly identify the missing requirements and write clarification questions.
Prefer questions that would change the final code, not cosmetic questions.

If you cannot identify any missing or ambiguous requirement from the original problem,
return an empty clarification question list.

Return concise JSON with this shape:

{
  "clarifying_questions": ["..."],
  "suspected_ambiguities": ["..."],
  "severity": "none|low|medium|high"
}
"""


def build_direct_questions_user_prompt(*, problem: str) -> str:
    return problem
