"""Prompt templates for EchoCoder."""

from typing import Literal


PromptProfile = Literal["clarifycodebench", "hilbench"]


def _normalize_question_history(
    previous_valid_clarifications: list[dict[str, str]] | None,
    previous_invalid_questions: list[str] | None,
) -> tuple[list[dict[str, str]], list[str]]:
    valid: list[dict[str, str]] = []
    valid_keys: set[str] = set()
    for item in previous_valid_clarifications or []:
        question = " ".join(str(item.get("question", "")).split()).strip()
        answer = " ".join(str(item.get("answer", "")).split()).strip()
        key = question.casefold()
        if question and answer and key not in valid_keys:
            valid_keys.add(key)
            valid.append({"question": question, "answer": answer})

    invalid: list[str] = []
    invalid_keys: set[str] = set()
    for item in previous_invalid_questions or []:
        question = " ".join(str(item).split()).strip()
        key = question.casefold()
        if question and key not in valid_keys and key not in invalid_keys:
            invalid_keys.add(key)
            invalid.append(question)
    return valid, invalid


def _bullet_list(items: list[str], *, empty: str = "- None") -> str:
    cleaned = [" ".join(str(item).split()).strip() for item in items]
    nonempty = [item for item in cleaned if item]
    return "\n".join(f"- {item}" for item in nonempty) if nonempty else empty


def _valid_clarification_bullets(items: list[dict[str, str]]) -> str:
    if not items:
        return "- None"
    return "\n".join(
        f"- Question: {item['question']}\n  Answer: {item['answer']}"
        for item in items
    )


def _first_valid_question_requirement(
    valid_clarifications: list[dict[str, str]],
) -> str:
    if valid_clarifications:
        return ""
    return (
        "No clarification question has been accepted yet, so return at least one "
        "clarification question and do not return an empty clarifying_questions list."
    )

GENERATE_CODE_SYSTEM_PROMPT = """\
You are a helpful coding assistant.

You are given a competitive-programming problem.

Generate a complete, runnable Python solution that implements the problem.

Return

[CODE]
<complete Python code>
[/CODE]
"""


GENERATE_SQL_SYSTEM_PROMPT = """\
You are a helpful coding assistant.

You are given a SQL task.

Generate a complete, executable SQL query that implements the task.

Return

[CODE]
<SQL query>
[/CODE]
"""


GENERATE_SWE_SYSTEM_PROMPT = """\
You are a helpful coding assistant.

You are given a software-engineering task.

Generate a complete implementation or patch-oriented solution that addresses the task.

Return

[CODE]
<implementation artifact>
[/CODE]
"""


def get_generate_code_system_prompt(task_type: str) -> str:
    if task_type == "sql":
        return GENERATE_SQL_SYSTEM_PROMPT
    if task_type == "swe":
        return GENERATE_SWE_SYSTEM_PROMPT
    return GENERATE_CODE_SYSTEM_PROMPT


INITIAL_DETECTION_AND_CODE_SYSTEM_PROMPT = """\
You are a helpful coding assistant.

You are given an underspecified {task_kind} task.

First generate distinct clarification questions that could change the implementation
or its output. You may generate more than one question, but each question must address
exactly one ambiguity.

If no further clarification question is needed, return an empty clarification question
list.

Then generate {artifact_description}.

Return

[INITIAL_QUESTIONS]
{{"clarifying_questions": ["..."]}}
[/INITIAL_QUESTIONS]

[CODE]
{code_placeholder}
[/CODE]
"""


def get_initial_detection_and_code_system_prompt(task_type: str) -> str:
    if task_type == "sql":
        task_kind = "SQL"
        artifact_description = "a complete, executable SQL query that implements the task"
        code_placeholder = "<complete SQL query>"
    elif task_type == "swe":
        task_kind = "software-engineering"
        artifact_description = (
            "a complete implementation or patch-oriented solution that addresses "
            "the task"
        )
        code_placeholder = "<complete implementation artifact>"
    else:
        task_kind = "competitive-programming"
        artifact_description = (
            "a complete, runnable Python solution that implements the problem"
        )
        code_placeholder = "<complete Python code>"

    return INITIAL_DETECTION_AND_CODE_SYSTEM_PROMPT.format(
        task_kind=task_kind,
        artifact_description=artifact_description,
        code_placeholder=code_placeholder,
    )


def build_generate_code_user_prompt(
    *,
    problem: str,
    task_type: str = "programming",
    previous_valid_clarifications: list[dict[str, str]] | None = None,
    previous_invalid_questions: list[str] | None = None,
    require_first_valid_question: bool = True,
    include_history: bool = True,
) -> str:
    valid, invalid = _normalize_question_history(
        previous_valid_clarifications, previous_invalid_questions
    )
    first_question_requirement = (
        _first_valid_question_requirement(valid)
        if require_first_valid_question
        else ""
    )
    if not include_history:
        return f"""\
Task type: {task_type}

Problem:
{problem}

{first_question_requirement}
"""
    return f"""\
Task type: {task_type}

Problem:
{problem}

Previously accepted clarification questions and answers:
{_valid_clarification_bullets(valid)}
Do not generate these questions again. Treat their answers as confirmed requirements.

Previously rejected clarification questions:
{_bullet_list(invalid)}
Do not generate these questions again or paraphrase the same semantic decisions.

{first_question_requirement}
"""


BACK_DESCRIBE_SYSTEM_PROMPT = """\
You are a helpful coding assistant.

You are given an original problem description and a generated solution artifact.

{reconstruction_audit}

The updated description will be used to detect underspecified points in the original
requirements. If you are unsure whether the original description specifies a rule,
include that rule in the updated description to maximize recall.

If the generated code implements no problem-level behavior beyond what the original
description already states, preserve the original description unchanged and return an
empty added_points list.

Use exactly the same format as the original problem and augment it directly. Also
list every requirement that you added. Each added point must state exactly one
problem-level requirement. Keep separate requirements in separate list items, even
when they are related. Do not combine multiple requirements in one item, and do not
include duplicate or paraphrased points.

Return only valid JSON with exactly this shape:

{
  "description": "<complete original description augmented with the inferred requirements>",
  "added_points": [
    "<one atomic requirement added to the description>"
  ]
}
"""


def build_back_describe_user_prompt(*, original_problem: str, code: str) -> str:
    return f"""\
Original problem description to preserve and augment:
{original_problem}

Generated solution artifact:
```
{code}
```
"""


DETECT_HALLUCINATION_SYSTEM_PROMPT = """\
You are a helpful coding assistant.

{question_intro} The generated solution, augmented description, and added points are evidence
only; the original problem is the sole authority.

Perform this audit internally before writing questions:

{question_audit}

{question_focus}

Ignore parser permissiveness, malformed or extra input, internal data structures, and
time or memory limits unless the original makes them required behavior. Do not invent
weak adjacent ambiguities when a stronger evidence-backed candidate exists.

{empty_question_rule}

{question_output_rule}

{"clarifying_questions": ["..."]}
"""


DETECT_HALLUCINATION_WITH_INITIAL_QUESTIONS_SYSTEM_PROMPT = """\
You are a helpful coding assistant.

{question_intro} The generated solution, augmented description, added points, and initial
questions are evidence only; the original problem is the sole authority.

Perform this audit internally before writing questions:

{question_audit}

{question_focus}

Ignore parser permissiveness, malformed or extra input, internal data structures, and
time or memory limits unless the original makes them required behavior. Do not invent
weak adjacent ambiguities when a stronger evidence-backed candidate exists.

{empty_question_rule}

{question_output_rule}

{"clarifying_questions": ["..."]}
"""


_CLARIFY_RECONSTRUCTION_AUDIT = """\
Preserve the complete original description and add problem-level requirements exposed
by the generated code. Read the code carefully to identify assumptions implemented in
the code but absent from the original description, including requirements related to
input, output format, terminology, edge cases, behavior, indices and ranges, ordering
and atomicity, numerical precision (such as rounding, tolerances, and error-handling
requirements), units, comparison rules, and collection semantics."""

_CLARIFY_QUESTION_INTRO = """\
Find clarification questions for requirements missing or ambiguous in the original
problem."""

_CLARIFY_QUESTION_AUDIT = """\
1. Build a candidate inventory from a direct scan of the original, every atomic added
   point, each behavioral decision in the generated solution, and every difference
   between the augmented and original descriptions.
2. Assign each candidate one semantic slot: terminology, value domain, collection
   semantics, operation semantics, objective, ordering, edge case, numerical rule, or
   output schema.
3. Classify every candidate as missing, partial, explicit, or conflicting. It is
   explicit only when the original fully states the same rule in its corresponding
   semantic context. If the answer appears only in another section, such as a
   constraint, formula, example, or output clause, treat that as partial evidence even
   when it strongly implies the answer. A convention is also only partial evidence.
4. Keep every missing, partial, or conflicting candidate. Drop only candidates that
   are fully explicit or concern implementation internals. Account for every candidate;
   never silently skip one or replace it with a nearby issue.

Use a complete semantic scan:

- For each input entity, check its data type, allowed values or range, units,
  distinctness, and guarantees, especially when these are only implied elsewhere.
- For each operation, recover its state transition and alternatives: insertion with
  shifting versus replacement, the amount moved, whether zero operations are allowed,
  how a group value is selected, and whether order or multiplicity matters.
- For each objective word such as optimally, result, length, possible, minimum, or
  maximum, identify the selected quantity and whether an arbitrary or extremal result
  is required.
- For each collection, check membership, duplicates, distinctness, multiplicity,
  ordering, and whether positions or only values matter. Do not replace a collection
  question with a question about equality among variables in one tuple.
- Audit output as a schema: printed values, count or header, ordering, number of lines,
  line-to-query mapping, separators, empty/no-solution behavior, and precision. Prefer
  the missing field itself over an adjacent formatting edge case.
- Do not dismiss a generic term such as progression, subsequence, rectangle, or set as
  standard when an unstated property could change its interpretation."""

_CLARIFY_QUESTION_AUDIT_WITH_INITIAL = """\
1. Build a candidate inventory from every initial question, every atomic added point,
   a direct scan of the original, each behavioral decision in the generated solution,
   and every difference between the augmented and original descriptions.
2. Assign each candidate one semantic slot: terminology, value domain, collection
   semantics, operation semantics, objective, ordering, edge case, numerical rule, or
   output schema.
3. Classify every candidate as missing, partial, explicit, or conflicting. It is
   explicit only when the original fully states the same rule in its corresponding
   semantic context. If the answer appears only in another section, such as a
   constraint, formula, example, or output clause, treat that as partial evidence even
   when it strongly implies the answer. A convention is also only partial evidence.
4. Keep every missing, partial, or conflicting candidate. Drop only candidates that
   are fully explicit or concern implementation internals. Account for every candidate;
   never silently skip one or replace it with a nearby issue.

Reconsider each initial question separately. Keep it unless the original fully and
directly answers its semantic decision. Rewrite it only to remove artifact references,
make it atomic, or restore the decisive alternatives. Do not remove it because its
answer is conventional, indirectly inferable, selected by the solution, or happens not
to change the current solution's output.

Use a complete semantic scan:

- For each input entity, check its data type, allowed values or range, units,
  distinctness, and guarantees, especially when these are only implied elsewhere.
- For each operation, recover its state transition and alternatives: insertion with
  shifting versus replacement, the amount moved, whether zero operations are allowed,
  how a group value is selected, and whether order or multiplicity matters.
- For each objective word such as optimally, result, length, possible, minimum, or
  maximum, identify the selected quantity and whether an arbitrary or extremal result
  is required.
- For each collection, check membership, duplicates, distinctness, multiplicity,
  ordering, and whether positions or only values matter. Do not replace a collection
  question with a question about equality among variables in one tuple.
- Audit output as a schema: printed values, count or header, ordering, number of lines,
  line-to-query mapping, separators, empty/no-solution behavior, and precision. Prefer
  the missing field itself over an adjacent formatting edge case.
- Do not dismiss a generic term such as progression, subsequence, rectangle, or set as
  standard when an unstated property could change its interpretation."""

_CLARIFY_QUESTION_FOCUS = """\
Ask about the underlying user-level requirement, not a data structure, derived formula,
truth table, or other consequence chosen by the solution. Infer latent requirements
behind code behavior: a status-to-label table may reveal missing entity relationships,
causal assumptions, uniqueness, or the meaning of the requested label.

Write from the perspective of a user reading only the original. Never mention the code,
solution, reconstruction, artifact, or assumptions made by them. Each question must
cover one independently answerable decision and preserve its decisive alternatives.
Never merge separate candidates with "and" or "or" merely to shorten the result.
Deduplicate only questions that resolve the same semantic slot and decision."""

_CLARIFY_EMPTY_QUESTION_RULE = """\
If the reconstructed description contains no requirement that is absent or ambiguous
in the original problem, return an empty clarification question list."""

_CLARIFY_QUESTION_OUTPUT_RULE = """\
Return the complete set of distinct clarification questions as valid JSON:"""

_CLARIFY_QUESTION_OUTPUT_RULE_WITH_INITIAL = """\
Return the complete final set as valid JSON:"""

# HiL-Bench paper, Appendix A (taxonomy) and B (blocker quality criteria).
_HILBENCH_RECONSTRUCTION_AUDIT = """\
Preserve the complete original description and surface task-critical decisions
that the generated solution artifact makes without authority from the task. The
artifact may be a code patch or a SQL query: treat its choices as evidence of
assumptions, not as proof that those choices are required. Inspect especially:

- Missing information: an exact value, threshold, name, format, API behavior,
  domain definition, or schema meaning needed to implement the requested result
  but not supplied by the task or available environment.
- Ambiguous requests: two or more plausible interpretations of the requested
  behavior, scope, precedence, timing, or data mapping that lead to materially
  different correct solutions.
- Contradictory information: task statements or available context that cannot
  all be satisfied, including a precedence rule silently chosen by the artifact.

Add only user-level decisions that affect core task correctness. Keep each
decision atomic. Do not promote an implementation detail, ordinary engineering
default, optional polish, or choice already determined by the task or available
environment into a new requirement."""


_HILBENCH_QUESTION_AUDIT = """\
1. Make an internal coverage ledger for every atomic added point, every
   task-level choice in the solution artifact, and every gap or conflict found
   by scanning the original task directly. Re-evaluate any initial questions
   against the same ledger; do not rely on the reconstruction to find all gaps.
2. For each candidate decision, identify the artifact's choice, the exact
   original-task statement or accepted clarification that settles that same
   decision (if any), and a plausible alternative that would materially change
   the required patch or query. Check concrete repository, test, schema, and
   data evidence as well. Existing behavior or a convention settles a gap only
   when the task adopts that contract and the evidence uniquely fixes its answer.
   The reconstruction and artifact reveal possible gaps; their choices are not
   authoritative answers. Account for each candidate before selecting questions.
3. Classify unresolved candidates as missing information (a required value or
   meaning is unavailable), ambiguous requests (multiple plausible outcomes
   remain), or contradictory information (applicable instructions cannot all
   hold). Check conflicts even when no added point describes them.
4. Keep independent decisions needed for core task correctness. Exact values,
   thresholds, source fields, API arguments, enum tokens, cache keys,
   serialization names, import paths, activation conditions, precedence, and
   error or fallback behavior may be task-level requirements: do not discard
   them merely because they look code-specific. Drop choices already resolved
   by authoritative evidence, routine internal defaults, optional refinements,
   speculative edge cases, and duplicates of the same decision. Do not turn
   every artifact choice into a question."""


_HILBENCH_QUESTION_INTRO = """\
Find clarification questions for task-critical missing information, ambiguous
requests, or contradictory requirements."""


_HILBENCH_QUESTION_FOCUS = """\
Ask about the task-level decision, including an API value or data meaning when
that is the unresolved requirement. Write for a task owner familiar with the
issue and the relevant environment. Do not mention the generated solution,
reconstruction, artifact, hidden tests, or benchmark. Each question must cover
one independently answerable decision and preserve its decisive alternatives.
Deduplicate questions that resolve the same decision. Rank the remaining gaps by
their impact on the core result and the strength of the task evidence. Return
one question for each eligible independent decision, up to five. Return fewer
than five only if fewer than five eligible decisions remain after the complete
audit; do not stop after finding the first good question."""


_HILBENCH_EMPTY_QUESTION_RULE = """\
If no task-critical missing information, ambiguity, or contradiction remains,
return an empty clarification question list."""


_HILBENCH_QUESTION_OUTPUT_RULE = "Return only the distinct questions that pass this audit as valid JSON:"


def _render_system_prompt(template: str, **sections: str) -> str:
    for name, content in sections.items():
        placeholder = "{" + name + "}"
        if template.count(placeholder) != 1:
            raise ValueError(f"Expected one {placeholder} in system prompt template")
        template = template.replace(placeholder, content, 1)
    return template


def get_back_describe_system_prompt(profile: PromptProfile = "clarifycodebench") -> str:
    if profile == "clarifycodebench":
        audit = _CLARIFY_RECONSTRUCTION_AUDIT
    elif profile == "hilbench":
        audit = _HILBENCH_RECONSTRUCTION_AUDIT
    else:
        raise ValueError(f"Unknown prompt profile: {profile}")
    return _render_system_prompt(
        BACK_DESCRIBE_SYSTEM_PROMPT, reconstruction_audit=audit
    )


def get_detect_hallucination_system_prompt(
    profile: PromptProfile = "clarifycodebench", *, with_initial_questions: bool
) -> str:
    template = (
        DETECT_HALLUCINATION_WITH_INITIAL_QUESTIONS_SYSTEM_PROMPT
        if with_initial_questions else DETECT_HALLUCINATION_SYSTEM_PROMPT
    )
    if profile == "clarifycodebench":
        sections = {
            "question_intro": _CLARIFY_QUESTION_INTRO,
            "question_audit": (
                _CLARIFY_QUESTION_AUDIT_WITH_INITIAL
                if with_initial_questions else _CLARIFY_QUESTION_AUDIT
            ),
            "question_focus": _CLARIFY_QUESTION_FOCUS,
            "empty_question_rule": _CLARIFY_EMPTY_QUESTION_RULE,
            "question_output_rule": (
                _CLARIFY_QUESTION_OUTPUT_RULE_WITH_INITIAL
                if with_initial_questions else _CLARIFY_QUESTION_OUTPUT_RULE
            ),
        }
    elif profile == "hilbench":
        sections = {
            "question_intro": _HILBENCH_QUESTION_INTRO,
            "question_audit": _HILBENCH_QUESTION_AUDIT,
            "question_focus": _HILBENCH_QUESTION_FOCUS,
            "empty_question_rule": _HILBENCH_EMPTY_QUESTION_RULE,
            "question_output_rule": _HILBENCH_QUESTION_OUTPUT_RULE,
        }
    else:
        raise ValueError(f"Unknown prompt profile: {profile}")
    return _render_system_prompt(template, **sections)


def build_detect_hallucination_user_prompt(
    *,
    original_problem: str,
    reconstructed_problem: str,
    added_points: list[str],
    generated_code: str,
    previous_valid_clarifications: list[dict[str, str]] | None = None,
) -> str:
    valid, _ = _normalize_question_history(previous_valid_clarifications, None)
    return f"""\
Original problem description:
{original_problem}

Original description augmented with semantics from generated code:
{reconstructed_problem}

Atomic requirements added during reconstruction:
{_bullet_list(added_points)}

Generated solution artifact:
```
{generated_code}
```

{_first_valid_question_requirement(valid)}
"""


def build_detect_hallucination_with_initial_questions_user_prompt(
    *,
    original_problem: str,
    reconstructed_problem: str,
    added_points: list[str],
    generated_code: str,
    initial_questions: list[str],
    previous_valid_clarifications: list[dict[str, str]] | None = None,
    previous_invalid_questions: list[str] | None = None,
    include_history: bool = True,
) -> str:
    valid, invalid = _normalize_question_history(
        previous_valid_clarifications, previous_invalid_questions
    )
    if not include_history:
        return f"""\
Original problem description:
{original_problem}

Original description augmented with semantics from generated code:
{reconstructed_problem}

Atomic requirements added during reconstruction:
{_bullet_list(added_points)}

Generated solution artifact:
```
{generated_code}
```

Initial clarification questions to reconsider:
{_bullet_list(initial_questions)}

{_first_valid_question_requirement(valid)}
"""
    return f"""\
Original problem description:
{original_problem}

Original description augmented with semantics from generated code:
{reconstructed_problem}

Atomic requirements added during reconstruction:
{_bullet_list(added_points)}

Generated solution artifact:
```
{generated_code}
```

Initial clarification questions to reconsider:
{_bullet_list(initial_questions)}

Previously accepted clarification questions and answers:
{_valid_clarification_bullets(valid)}
Do not generate these questions again. Treat their answers as confirmed requirements.

Previously rejected clarification questions:
{_bullet_list(invalid)}
Do not generate these questions again or paraphrase the same semantic decisions.

{_first_valid_question_requirement(valid)}
"""
