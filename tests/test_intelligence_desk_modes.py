import pandas as pd

from app.pages import nlq_chat
from app.state import DrillState, FilterState


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_ticker": ["AAA", "BBB", "CCC"],
            "channel": ["RIA", "Bank", "RIA"],
            "src_country": ["US", "US", "BR"],
            "nnb": [120.0, 80.0, -15.0],
            "end_aum": [1000.0, 850.0, 400.0],
        }
    )


def test_dataset_qa_rejects_generic_questions_without_general_fallback(monkeypatch):
    df = _sample_df()

    def fail_general_answer(*args, **kwargs):
        raise AssertionError("generic fallback should not be used in Dataset Q&A")

    monkeypatch.setattr(nlq_chat, "claude_generate_general_answer", fail_general_answer, raising=False)

    out = nlq_chat.answer_intelligence_desk(
        "What is an ETF?",
        nlq_chat.INTEL_MODE_DATASET_QA,
        df,
    )

    assert out["mode"] == nlq_chat.INTEL_MODE_DATASET_QA
    assert out["source"] == "internal_dataset"
    assert "generic finance chat" in out["answer"]
    assert out["error"] == "not_dataset_question"
    assert out["requested_mode"] == nlq_chat.INTEL_MODE_DATASET_QA
    assert out["resolved_mode"] == nlq_chat.INTEL_MODE_DATASET_QA


def test_claude_analyst_uses_internal_dataset_context(monkeypatch):
    df = _sample_df()
    subset = df.head(2).copy()

    monkeypatch.setattr(
        nlq_chat,
        "retrieve_intelligence_desk_context",
        lambda question, work_df: (subset, subset.to_markdown(index=False)),
    )
    monkeypatch.setattr(
        nlq_chat,
        "claude_generate_grounded",
        lambda system_prompt, user_message, model, max_tokens: "Grounded internal analysis.",
    )
    monkeypatch.setattr(
        nlq_chat,
        "claude_generate_general_answer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("general fallback should not be used")),
        raising=False,
    )

    out = nlq_chat.answer_intelligence_desk(
        "Summarize flow trends from the dataset.",
        nlq_chat.INTEL_MODE_CLAUDE_ANALYST,
        df,
    )

    assert out["mode"] == nlq_chat.INTEL_MODE_CLAUDE_ANALYST
    assert out["source"] == "internal_dataset"
    assert out["answer"] == "Grounded internal analysis."
    assert out["subset_df"].equals(subset)
    assert out["columns_used"] == list(subset.columns)
    assert out["row_count"] == len(subset)
    assert out["grounded"] is True


def test_market_intelligence_does_not_answer_internal_data_questions(monkeypatch):
    df = _sample_df()

    monkeypatch.setattr(
        nlq_chat,
        "search_market_context",
        lambda question: (_ for _ in ()).throw(AssertionError("external market search should not run for data questions")),
    )

    out = nlq_chat.answer_intelligence_desk(
        "Which ETF had the highest NNB?",
        nlq_chat.INTEL_MODE_MARKET_INTEL,
        df,
    )

    assert out["mode"] == nlq_chat.INTEL_MODE_MARKET_INTEL
    assert out["source"] == "external_market"
    assert "does not answer questions from your internal dataset" in out["answer"]
    assert out["error"] == "wrong_mode"


def test_selected_segment_scope_does_not_fallback_to_broader_dataset(monkeypatch):
    df = _sample_df()
    state = FilterState.from_dict({})

    monkeypatch.setattr(nlq_chat, "_load_intelligence_desk_df", lambda *args, **kwargs: df)
    monkeypatch.setattr(nlq_chat, "get_drill_state", lambda: DrillState())

    base_df, segment_df = nlq_chat._load_intelligence_desk_df_by_scope(
        nlq_chat.INTEL_SCOPE_SEGMENT,
        state,
        nlq_chat.ROOT,
    )

    assert base_df.equals(df)
    assert isinstance(segment_df, pd.DataFrame)
    assert segment_df.empty


def test_auto_resolves_direct_internal_questions_to_dataset_qa() -> None:
    resolution = nlq_chat.resolve_intelligence_mode(
        "Which ETF had the highest NNB above $100k?",
        nlq_chat.INTEL_MODE_AUTO,
    )

    assert resolution.requested_mode == nlq_chat.INTEL_MODE_AUTO
    assert resolution.resolved_mode == nlq_chat.INTEL_MODE_DATASET_QA
    assert resolution.question_shape == "direct_data"
    assert resolution.requires_clarification is False


def test_auto_resolves_narrative_internal_questions_to_claude_analyst() -> None:
    resolution = nlq_chat.resolve_intelligence_mode(
        "Summarize the NNB trend in our data and explain the drivers.",
        nlq_chat.INTEL_MODE_AUTO,
    )

    assert resolution.resolved_mode == nlq_chat.INTEL_MODE_CLAUDE_ANALYST
    assert resolution.question_shape == "narrative_data"
    assert resolution.requires_clarification is False


def test_auto_resolves_external_questions_to_market_intelligence() -> None:
    resolution = nlq_chat.resolve_intelligence_mode(
        "What is the market outlook for ETF flows given inflation and Fed expectations?",
        nlq_chat.INTEL_MODE_AUTO,
    )

    assert resolution.resolved_mode == nlq_chat.INTEL_MODE_MARKET_INTEL
    assert resolution.question_shape == "market"
    assert resolution.requires_clarification is False


def test_auto_requires_clarification_for_mixed_questions() -> None:
    resolution = nlq_chat.resolve_intelligence_mode(
        "What is our NNB trend and how does that compare to the market?",
        nlq_chat.INTEL_MODE_AUTO,
    )

    assert resolution.resolved_mode == nlq_chat.INTEL_MODE_AUTO
    assert resolution.question_shape == "mixed"
    assert resolution.requires_clarification is True


def test_explicit_mode_is_preserved_by_resolver() -> None:
    resolution = nlq_chat.resolve_intelligence_mode(
        "What is our NNB trend and how does that compare to the market?",
        nlq_chat.INTEL_MODE_DATASET_QA,
    )

    assert resolution.requested_mode == nlq_chat.INTEL_MODE_DATASET_QA
    assert resolution.resolved_mode == nlq_chat.INTEL_MODE_DATASET_QA
    assert resolution.requires_clarification is False


def test_answer_payload_preserves_requested_and_resolved_mode(monkeypatch):
    df = _sample_df()
    resolution = nlq_chat.resolve_intelligence_mode(
        "Summarize flow trends from the dataset.",
        nlq_chat.INTEL_MODE_AUTO,
    )
    subset = df.head(2).copy()

    monkeypatch.setattr(
        nlq_chat,
        "retrieve_intelligence_desk_context",
        lambda question, work_df: (subset, subset.to_markdown(index=False)),
    )
    monkeypatch.setattr(
        nlq_chat,
        "claude_generate_grounded",
        lambda system_prompt, user_message, model, max_tokens: "Grounded internal analysis.",
    )

    out = nlq_chat.answer_intelligence_desk(
        "Summarize flow trends from the dataset.",
        resolution.resolved_mode,
        df,
        requested_mode=resolution.requested_mode,
        source_scope=nlq_chat.INTEL_SCOPE_CURRENT,
        resolution=resolution,
    )

    assert out["requested_mode"] == nlq_chat.INTEL_MODE_AUTO
    assert out["resolved_mode"] == nlq_chat.INTEL_MODE_CLAUDE_ANALYST
    assert out["source_scope"] == nlq_chat.INTEL_SCOPE_CURRENT
    assert out["summary_table"] is not None
    assert out["columns_used"] == list(subset.columns)
    assert out["row_count"] == len(subset)
    assert out["grounded"] is True


def test_answer_payload_returns_clarification_for_mixed_auto_request() -> None:
    df = _sample_df()
    resolution = nlq_chat.resolve_intelligence_mode(
        "What is our NNB trend and how does that compare to the market?",
        nlq_chat.INTEL_MODE_AUTO,
    )

    out = nlq_chat.answer_intelligence_desk(
        "What is our NNB trend and how does that compare to the market?",
        resolution.resolved_mode,
        df,
        requested_mode=resolution.requested_mode,
        source_scope=nlq_chat.INTEL_SCOPE_CURRENT,
        resolution=resolution,
    )

    assert out["requested_mode"] == nlq_chat.INTEL_MODE_AUTO
    assert out["resolved_mode"] == nlq_chat.INTEL_MODE_AUTO
    assert out["requires_clarification"] is True
    assert out["error"] == "clarification_required"
    assert "split it into two questions" in out["answer"]
    assert out["original_question"] == "What is our NNB trend and how does that compare to the market?"
    assert [opt["mode"] for opt in out["clarification_options"]] == [
        nlq_chat.INTEL_MODE_DATASET_QA,
        nlq_chat.INTEL_MODE_CLAUDE_ANALYST,
        nlq_chat.INTEL_MODE_MARKET_INTEL,
    ]


def test_format_selection_is_presentation_only_for_resolved_mode() -> None:
    resolution = nlq_chat.resolve_intelligence_mode(
        "Summarize flow trends from the dataset.",
        nlq_chat.INTEL_MODE_AUTO,
    )
    out = {
        "answer": "Grounded internal analysis.",
        "mode": resolution.resolved_mode,
        "requested_mode": resolution.requested_mode,
        "resolved_mode": resolution.resolved_mode,
        "source_scope": nlq_chat.INTEL_SCOPE_CURRENT,
        "grounded": True,
        "source": "internal_dataset",
        "error": None,
        "resolution_reason": resolution.reason,
    }

    standard = nlq_chat._build_intelligence_assistant_message(
        question="Summarize flow trends from the dataset.",
        out=out,
        requested_scope=nlq_chat.INTEL_SCOPE_CURRENT,
        format_style=nlq_chat.INTEL_FORMAT_STANDARD,
        drill_state=DrillState(),
    )
    bullets = nlq_chat._build_intelligence_assistant_message(
        question="Summarize flow trends from the dataset.",
        out=out,
        requested_scope=nlq_chat.INTEL_SCOPE_CURRENT,
        format_style=nlq_chat.INTEL_FORMAT_BULLETS,
        drill_state=DrillState(),
    )

    assert standard["resolved_mode"] == nlq_chat.INTEL_MODE_CLAUDE_ANALYST
    assert bullets["resolved_mode"] == nlq_chat.INTEL_MODE_CLAUDE_ANALYST
    assert standard["format_style"] == nlq_chat.INTEL_FORMAT_STANDARD
    assert bullets["format_style"] == nlq_chat.INTEL_FORMAT_BULLETS


def test_metadata_payload_contains_expected_fields() -> None:
    payload = nlq_chat._build_intelligence_metadata_payload(
        {
            "requested_mode": nlq_chat.INTEL_MODE_AUTO,
            "resolved_mode": nlq_chat.INTEL_MODE_DATASET_QA,
            "format_style": nlq_chat.INTEL_FORMAT_COMPACT,
            "source_scope": nlq_chat.INTEL_SCOPE_CURRENT,
            "grounded": True,
        }
    )

    assert payload == {
        "requested_mode": nlq_chat.INTEL_MODE_AUTO,
        "resolved_mode": nlq_chat.INTEL_MODE_DATASET_QA,
        "format_style": nlq_chat.INTEL_FORMAT_COMPACT,
        "scope_used": nlq_chat.INTEL_SCOPE_LABELS[nlq_chat.INTEL_SCOPE_CURRENT],
        "grounded": "Yes",
    }


def test_render_metadata_row_handles_legacy_history_items(monkeypatch) -> None:
    markdown_calls: list[str] = []
    monkeypatch.setattr(nlq_chat.st, "markdown", lambda text, **kwargs: markdown_calls.append(text))

    nlq_chat._render_intelligence_metadata_row({"role": "assistant", "text": "Legacy answer"})

    assert markdown_calls
    assert "Requested:" in markdown_calls[0]
    assert "Format:" in markdown_calls[0]
    assert "Scope used:" in markdown_calls[0]
    assert "Grounded:" in markdown_calls[0]


def test_clarification_card_preserves_original_question_with_partial_metadata(monkeypatch) -> None:
    info_calls: list[str] = []
    monkeypatch.setattr(nlq_chat.st, "info", lambda text, **kwargs: info_calls.append(text))

    nlq_chat._render_intelligence_clarification_card(
        {
            "requires_clarification": True,
            "original_question": "What is our NNB trend and how does that compare to the market?",
        }
    )

    assert info_calls
    assert "What is our NNB trend and how does that compare to the market?" in info_calls[0]
    assert "From your data" in info_calls[0]
    assert "External market view" in info_calls[0]


def test_selected_segment_scope_without_drill_is_honest() -> None:
    text = nlq_chat._get_scope_reminder_text(
        nlq_chat.INTEL_MODE_DATASET_QA,
        nlq_chat.INTEL_SCOPE_SEGMENT,
        DrillState(),
    )

    assert text == "Selected segment only (no channel or ticker drill-down selected)"


def test_auto_metadata_still_renders_correctly_after_ui_additions() -> None:
    resolution = nlq_chat.resolve_intelligence_mode(
        "Summarize flow trends from the dataset.",
        nlq_chat.INTEL_MODE_AUTO,
    )
    assistant = nlq_chat._build_intelligence_assistant_message(
        question="Summarize flow trends from the dataset.",
        out={
            "answer": "Grounded internal analysis.",
            "mode": resolution.resolved_mode,
            "requested_mode": resolution.requested_mode,
            "resolved_mode": resolution.resolved_mode,
            "source_scope": nlq_chat.INTEL_SCOPE_CURRENT,
            "grounded": True,
            "source": "internal_dataset",
            "error": None,
            "resolution_reason": resolution.reason,
        },
        requested_scope=nlq_chat.INTEL_SCOPE_CURRENT,
        format_style=nlq_chat.INTEL_FORMAT_STANDARD,
        drill_state=DrillState(),
    )
    metadata = nlq_chat._build_intelligence_metadata_payload(assistant)

    assert metadata["requested_mode"] == nlq_chat.INTEL_MODE_AUTO
    assert metadata["resolved_mode"] == nlq_chat.INTEL_MODE_CLAUDE_ANALYST
    assert metadata["format_style"] == nlq_chat.INTEL_FORMAT_STANDARD
    assert metadata["scope_used"] == nlq_chat.INTEL_SCOPE_LABELS[nlq_chat.INTEL_SCOPE_CURRENT]
    assert metadata["grounded"] == "Yes"
