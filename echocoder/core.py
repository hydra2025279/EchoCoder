

from __future__ import annotations

from collections.abc import Callable

from baselines.common import llm_call_context

from echocoder.schemas import (
    ClarificationEvent,
    GenerationResult,
    OracleResult,
    QuestionBatchRecord,
    QuestionRecord,
    ReconstructionRecord,
    ReconstructionResult,
    SpecificationStateRecord,
    WorkflowConfig,
    WorkflowResult,
)


ValidClarification = dict[str, str]
Generate = Callable[
    [str, str | None, int, list[ValidClarification], list[str]],
    GenerationResult,
]
Reconstruct = Callable[[str, str], ReconstructionResult]
Detect = Callable[
    [
        str,
        str,
        ReconstructionResult,
        list[str],
        list[ValidClarification],
        list[str],
    ],
    tuple[str, list[str]],
]
AskOracleBatch = Callable[[list[str]], list[OracleResult]]
UpdateSpecification = Callable[[str, str, str], str]
Progress = Callable[[int, int | None, int | None, str, str], None]
Snapshot = Callable[[list[SpecificationStateRecord], str | None], None]
AnalyzeAccepted = Callable[[str, ReconstructionResult, list[ClarificationEvent]], list[dict[str, str]]]
AppendAnalysis = Callable[[str, list[dict[str, str]]], str]


def run_specification_completion(
    *,
    problem: str,
    generate: Generate,
    reconstruct: Reconstruct,
    detect: Detect,
    ask_oracle_batch: AskOracleBatch,
    update_specification: UpdateSpecification,
    config: WorkflowConfig,
    progress: Progress | None = None,
    snapshot: Snapshot | None = None,
    analyze_accepted: AnalyzeAccepted | None = None,
    append_analysis: AppendAnalysis | None = None,
) -> WorkflowResult:
    specification = problem
    previous_candidate: str | None = None
    rounds: list[SpecificationStateRecord] = []
    history: list[ClarificationEvent] = []
    rejected_questions: list[str] = []
    question_history: list[str] = []
    resolved_match_keys: set[str] = set()
    total_oracle_questions = 0
    total_question_calls = 0
    specification_round = 0

    def valid_history() -> list[ValidClarification]:
        return [
            {"question": item.question, "answer": item.answer}
            for item in history
        ]

    def emit(
        reconstruction_index: int | None,
        question_call_index: int | None,
        step: str,
        status: str,
    ) -> None:
        if progress is not None:
            progress(
                specification_round,
                reconstruction_index,
                question_call_index,
                step,
                status,
            )

    def emit_snapshot(stop_reason: str | None = None) -> None:
        if snapshot is not None:
            snapshot(rounds, stop_reason)

    while True:
        emit(None, None, "generate_complete_candidate", "start")
        state_id = f"S{specification_round}"
        generation_id = f"{state_id}_GEN"
        code_id = f"C{specification_round}"
        parent_code_id = (
            f"C{specification_round - 1}" if specification_round > 0 else None
        )
        generation_operation = (
            "initial_complete_generation"
            if specification_round == 0
            else "complete_repair_or_regeneration_after_clarification"
        )
        with llm_call_context(
            specification_state=specification_round,
            state_id=state_id,
            generation_id=generation_id,
            code_id=code_id,
            parent_code_id=parent_code_id,
            generation_operation=generation_operation,
            phase="generation",
        ):
            generation = generate(
                specification,
                previous_candidate,
                specification_round,
                valid_history(),
                list(rejected_questions),
            )
        if not generation.candidate.strip():
            raise RuntimeError("Generation returned an empty candidate")
        candidate = generation.candidate
        emit(None, None, "generate_complete_candidate", "done")

        state = SpecificationStateRecord(
            specification_round=specification_round,
            specification=specification,
            complete_code=candidate,
            initial_questions=generation.initial_questions,
            localized_context=generation.localized_context,
            state_id=state_id,
            generation_id=generation_id,
            code_id=code_id,
            parent_code_id=parent_code_id,
            generation_operation=generation_operation,
        )
        rounds.append(state)
        emit_snapshot()
        accepted: list[ClarificationEvent] = []

        for reconstruction_index in range(1, config.reconstruction_budget + 1):
            reconstruction_id = f"{state_id}_{code_id}_R{reconstruction_index}"
            emit(reconstruction_index, None, "reconstruct_requirements", "start")
            with llm_call_context(
                specification_state=specification_round,
                state_id=state_id,
                r=reconstruction_index,
                reconstruction_id=reconstruction_id,
                code_id=code_id,
                phase="reconstruction",
            ):
                reconstruction = reconstruct(specification, candidate)
            emit(reconstruction_index, None, "reconstruct_requirements", "done")
            reconstruction_record = ReconstructionRecord(
                reconstruction_index=reconstruction_index,
                reconstructed_description=reconstruction.description,
                added_points=reconstruction.added_points,
                reconstruction_id=reconstruction_id,
            )
            state.reconstructions.append(reconstruction_record)
            emit_snapshot()

            for question_call_index in range(1, config.question_call_budget + 1):
                emit(
                    reconstruction_index,
                    question_call_index,
                    "generate_question_batch",
                    "start",
                )
                prior_questions = list(question_history)
                prior_valid_clarifications = valid_history()
                prior_invalid_questions = list(rejected_questions)
                batch_id = f"Q{reconstruction_index}_{question_call_index}"
                global_batch_id = f"{state_id}_{code_id}_{batch_id}"
                with llm_call_context(
                    specification_state=specification_round,
                    state_id=state_id,
                    r=reconstruction_index,
                    reconstruction_id=reconstruction_id,
                    q=question_call_index,
                    code_id=code_id,
                    batch_id=batch_id,
                    global_batch_id=global_batch_id,
                    phase="question_generation",
                ):
                    detector_response, questions = detect(
                        specification,
                        candidate,
                        reconstruction,
                        generation.initial_questions,
                        prior_valid_clarifications,
                        prior_invalid_questions,
                    )
                total_question_calls += 1
                emit(
                    reconstruction_index,
                    question_call_index,
                    "generate_question_batch",
                    f"done ({len(questions)} question(s))",
                )

                if not questions and config.semantic_stop_on_empty_questions:
                    reconstruction_record.question_batches.append(
                        QuestionBatchRecord(
                            question_call_index=question_call_index,
                            detector_response=detector_response,
                            questions=[],
                            prior_questions=prior_questions,
                            prior_valid_clarifications=prior_valid_clarifications,
                            prior_invalid_questions=prior_invalid_questions,
                            batch_id=batch_id,
                            global_batch_id=global_batch_id,
                        )
                    )
                    emit_snapshot("no_questions_from_detector")
                    return WorkflowResult(
                        original_problem=problem,
                        rounds=rounds,
                        clarification_history=history,
                        rejected_questions=rejected_questions,
                        stop_reason="no_questions_from_detector",
                        final_specification=specification,
                        final_solution=candidate,
                        total_oracle_questions=total_oracle_questions,
                        total_question_calls=total_question_calls,
                    )

                emit(
                    reconstruction_index,
                    question_call_index,
                    "evaluate_question_batch",
                    "start",
                )
                with llm_call_context(
                    specification_state=specification_round,
                    state_id=state_id,
                    r=reconstruction_index,
                    reconstruction_id=reconstruction_id,
                    q=question_call_index,
                    code_id=code_id,
                    batch_id=batch_id,
                    global_batch_id=global_batch_id,
                    phase="oracle_evaluation",
                ):
                    oracle_results = ask_oracle_batch(questions)
                if len(oracle_results) != len(questions):
                    raise RuntimeError(
                        "Oracle returned a different number of results than questions"
                    )
                total_oracle_questions += len(questions)

                question_records: list[QuestionRecord] = []
                batch_accepted: list[ClarificationEvent] = []
                for question_index, (question, oracle) in enumerate(
                    zip(questions, oracle_results, strict=True), start=1
                ):
                    question_id = f"{batch_id}_{question_index}"
                    global_question_id = f"{global_batch_id}_{question_index}"
                    if (
                        oracle.status == "valid"
                        and oracle.matched_key_question
                        and oracle.matched_key_question in resolved_match_keys
                    ):
                        oracle = OracleResult(
                            status="invalid",
                            matched_key_question=oracle.matched_key_question,
                            evaluation={
                                **oracle.evaluation,
                                "duplicate_resolved_requirement": True,
                            },
                        )

                    question_records.append(
                        QuestionRecord(
                            question=question,
                            oracle_status=oracle.status,
                            answer=oracle.answer,
                            matched_key_question=oracle.matched_key_question,
                            oracle_evaluation=oracle.evaluation,
                            question_id=question_id,
                            global_question_id=global_question_id,
                        )
                    )
                    if oracle.status == "valid" and oracle.answer:
                        if oracle.matched_key_question:
                            resolved_match_keys.add(oracle.matched_key_question)
                        batch_accepted.append(
                            ClarificationEvent(
                                specification_round=specification_round,
                                reconstruction_index=reconstruction_index,
                                question_call_index=question_call_index,
                                question=question,
                                answer=oracle.answer,
                                matched_key_question=oracle.matched_key_question,
                                event_id=f"{global_question_id}_CLARIFICATION",
                                state_id=state_id,
                                code_id=code_id,
                                reconstruction_id=reconstruction_id,
                                batch_id=batch_id,
                                global_batch_id=global_batch_id,
                                question_id=question_id,
                                global_question_id=global_question_id,
                            )
                        )
                        if question not in question_history:
                            question_history.append(question)
                    else:
                        rejected_questions.append(question)
                        if question not in question_history:
                            question_history.append(question)

                reconstruction_record.question_batches.append(
                    QuestionBatchRecord(
                        question_call_index=question_call_index,
                        detector_response=detector_response,
                        questions=question_records,
                        prior_questions=prior_questions,
                        prior_valid_clarifications=prior_valid_clarifications,
                        prior_invalid_questions=prior_invalid_questions,
                        batch_id=batch_id,
                        global_batch_id=global_batch_id,
                    )
                )
                emit_snapshot()
                emit(
                    reconstruction_index,
                    question_call_index,
                    "evaluate_question_batch",
                    f"done ({len(batch_accepted)} valid)",
                )

                if batch_accepted:
                    accepted = batch_accepted
                    break

            if accepted:
                break

        if not accepted:
            emit_snapshot("no_valid_clarification_within_budget")
            return WorkflowResult(
                original_problem=problem,
                rounds=rounds,
                clarification_history=history,
                rejected_questions=rejected_questions,
                stop_reason="no_valid_clarification_within_budget",
                final_specification=specification,
                final_solution=candidate,
                total_oracle_questions=total_oracle_questions,
                total_question_calls=total_question_calls,
            )

        state.accepted_clarifications.extend(accepted)
        history.extend(accepted)
        if analyze_accepted is not None:
            emit(reconstruction_index, question_call_index, "analyze_accepted_assumptions", "start")
            with llm_call_context(
                specification_state=specification_round,
                state_id=state_id,
                code_id=code_id,
                reconstruction_id=reconstruction_id,
                r=reconstruction_index,
                q=question_call_index,
                phase="assumption_analysis",
            ):
                state.assumption_updates = analyze_accepted(
                    specification, reconstruction, accepted
                )
            emit(reconstruction_index, question_call_index, "analyze_accepted_assumptions", "done")
            emit_snapshot()
        for clarification in accepted:
            specification = update_specification(
                specification,
                clarification.question,
                clarification.answer,
            )
        if append_analysis is not None and state.assumption_updates:
            specification = append_analysis(specification, state.assumption_updates)
        previous_candidate = candidate
        specification_round += 1
        emit_snapshot()
