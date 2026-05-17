"""
Evaluation tests for ASR summary quality using ROUGE and BLEU metrics.

These tests validate _summarize_asr_with_llm() output against reference
summaries using automatic NLG metrics, without requiring a live LLM:
  - ROUGE-1 / ROUGE-2 / ROUGE-L  (recall, precision, F1)
  - BLEU-1 through BLEU-4        (ngram precision with brevity penalty)

Usage:
    pytest tests/unit/test_asr_summary_evals.py -v
    pytest tests/unit/test_asr_summary_evals.py -v -k "rouge"
    pytest tests/unit/test_asr_summary_evals.py -v --tb=short
"""
from __future__ import annotations

import types
from typing import Any

import pytest


# ─── Metric helpers ──────────────────────────────────────────────────────────

def _rouge_scores(hypothesis: str, reference: str) -> dict[str, dict[str, float]]:
    """Return {rouge1, rouge2, rougeL} each with {precision, recall, fmeasure}."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        key: {
            "precision": round(val.precision, 4),
            "recall":    round(val.recall,    4),
            "fmeasure":  round(val.fmeasure,  4),
        }
        for key, val in scores.items()
    }


def _bleu_scores(hypothesis: str, reference: str) -> dict[str, float]:
    """Return BLEU-1 through BLEU-4 corpus scores."""
    import nltk
    # Ensure punkt tokenizer data is available
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    ref_tokens = nltk.word_tokenize(reference.lower())

    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1

    return {
        "bleu1": round(sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth), 4),
        "bleu2": round(sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth), 4),
        "bleu3": round(sentence_bleu([ref_tokens], hyp_tokens, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth), 4),
        "bleu4": round(sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth), 4),
    }


# ─── Stub LLM client ────────────────────────────────────────────────────────

def _make_stub_client(response_text: str):
    """
    Create a minimal LLMClient-compatible stub that returns a fixed response.
    Avoids any real LLM call during testing.
    """
    ChatResponse = types.SimpleNamespace
    client = types.SimpleNamespace(
        chat=lambda prompt, temperature=0.2, **kw: ChatResponse(text=response_text)
    )
    return client


# ─── Reference data ──────────────────────────────────────────────────────────

# Each fixture is (transcript, speaker_turns, generated_summary, reference_summary).
# The reference summary represents the expected content — written by a human
# reviewer for the given transcript.  Add new fixtures as datasets grow.

FIXTURES: list[dict[str, Any]] = [
    {
        "id": "tech_meeting",
        "transcript": (
            "Alice: Good morning everyone. Today we'll discuss the Q3 roadmap. "
            "Bob: Thanks Alice. I think we need to prioritize the API redesign. "
            "Alice: Agreed. We also need to finalize the deployment timeline. "
            "Bob: I'd suggest we target end of July for the first release. "
            "Alice: That works. Let's schedule a follow-up next Friday."
        ),
        "speaker_turns": [
            {"speaker_name": "Alice", "start_s": 0.0,  "end_s": 5.0,  "text": "Good morning everyone. Today we'll discuss the Q3 roadmap."},
            {"speaker_name": "Bob",   "start_s": 5.5,  "end_s": 10.0, "text": "Thanks Alice. I think we need to prioritize the API redesign."},
            {"speaker_name": "Alice", "start_s": 10.5, "end_s": 15.0, "text": "Agreed. We also need to finalize the deployment timeline."},
            {"speaker_name": "Bob",   "start_s": 15.5, "end_s": 20.0, "text": "I'd suggest we target end of July for the first release."},
            {"speaker_name": "Alice", "start_s": 20.5, "end_s": 24.0, "text": "That works. Let's schedule a follow-up next Friday."},
        ],
        # A plausible LLM-generated summary (mocked)
        "generated_summary": (
            "Alice and Bob held a morning meeting to discuss the Q3 roadmap. "
            "Bob proposed prioritizing the API redesign as the main deliverable. "
            "The team agreed to target an end-of-July release for the first milestone. "
            "Alice emphasized finalizing the deployment timeline and scheduled a follow-up meeting for the next Friday. "
            "The overall tone was collaborative and action-oriented."
        ),
        # Human-written reference for metric comparison
        "reference_summary": (
            "Alice and Bob discussed the Q3 roadmap in a productive morning meeting. "
            "Bob recommended focusing on the API redesign, and the team agreed to target a July release. "
            "Alice wrapped up by arranging a follow-up meeting for the following Friday."
        ),
        # Minimum acceptable thresholds for this fixture
        "thresholds": {
            "rouge1_f": 0.40,
            "rouge2_f": 0.15,
            "rougeL_f": 0.30,
            "bleu1":    0.30,
            "bleu2":    0.15,
        },
    },
    {
        "id": "lecture_excerpt",
        "transcript": (
            "Instructor: Today we cover gradient descent. "
            "It's the backbone of training neural networks. "
            "You compute the gradient of the loss with respect to each parameter. "
            "Then update the weights in the opposite direction. "
            "The learning rate controls the step size. "
            "Too large and you overshoot the minimum. Too small and training is very slow."
        ),
        "speaker_turns": [
            {"speaker_name": "Instructor", "start_s": 0.0,  "end_s": 30.0,
             "text": "Today we cover gradient descent. It's the backbone of training neural networks. "
                     "You compute the gradient of the loss with respect to each parameter. "
                     "Then update the weights in the opposite direction. "
                     "The learning rate controls the step size. "
                     "Too large and you overshoot the minimum. Too small and training is very slow."},
        ],
        "generated_summary": (
            "The instructor introduced gradient descent as the foundational optimization algorithm for neural network training. "
            "It involves computing the gradient of the loss with respect to each parameter and updating weights in the opposite direction. "
            "A key hyperparameter is the learning rate, which governs step size — too large causes overshooting, while too small leads to slow convergence. "
            "The lecture was educational in tone, delivered by a single speaker."
        ),
        "reference_summary": (
            "The lecture covered gradient descent, explaining how gradients are computed and weights are updated to minimize the loss. "
            "The instructor highlighted the learning rate as a critical hyperparameter that balances convergence speed and stability."
        ),
        "thresholds": {
            "rouge1_f": 0.35,
            "rouge2_f": 0.10,
            "rougeL_f": 0.20,   # longer generated vs compact ref → LCS penalised
            "bleu1":    0.25,
            "bleu2":    0.10,
        },
    },
    {
        "id": "empty_transcript",
        "transcript": "",
        "speaker_turns": [],
        "generated_summary": "",
        "reference_summary": "",
        "thresholds": {},  # empty input → empty output, metrics not applied
    },
]


# ─── Helper: call _summarize_asr_with_llm with stub ─────────────────────────

def _run_summarize(transcript: str, speaker_turns: list[dict], stub_response: str) -> str:
    from src.orchestration.workflow_runtime import _summarize_asr_with_llm
    client = _make_stub_client(stub_response)
    return _summarize_asr_with_llm(transcript, speaker_turns, client)


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestASRSummaryEmpty:
    """Edge case: empty transcript should return empty string."""

    def test_empty_transcript_returns_empty(self):
        result = _run_summarize("", [], stub_response="Should not be called")
        assert result == "", f"Expected empty string, got: {result!r}"

    def test_whitespace_only_transcript_returns_empty(self):
        result = _run_summarize("   \n\t  ", [], stub_response="Should not be called")
        assert result == "", f"Expected empty string for whitespace, got: {result!r}"


class TestASRSummaryStubOutput:
    """Verify _summarize_asr_with_llm correctly passes through the LLM response."""

    def test_returns_stub_text(self):
        transcript = "Hello world. This is a test."
        stub = "This is the test summary."
        result = _run_summarize(transcript, [], stub_response=stub)
        assert result == stub

    def test_strips_whitespace_from_response(self):
        transcript = "Hello world."
        stub = "  Summary with padding.  "
        result = _run_summarize(transcript, [], stub_response=stub)
        assert result == stub.strip()

    def test_client_exception_returns_empty(self):
        """If the LLM client raises, _summarize_asr_with_llm must return ''."""
        from src.orchestration.workflow_runtime import _summarize_asr_with_llm

        def _raising_chat(**kw):
            raise RuntimeError("connection refused")

        client = types.SimpleNamespace(chat=_raising_chat)
        result = _summarize_asr_with_llm("Some transcript text.", [], client)
        assert result == ""


class TestASRSummaryROUGE:
    """ROUGE-score evaluations against reference summaries."""

    @pytest.mark.parametrize("fixture", [f for f in FIXTURES if f["transcript"]], ids=[f["id"] for f in FIXTURES if f["transcript"]])
    def test_rouge1_fmeasure(self, fixture):
        result = _run_summarize(fixture["transcript"], fixture["speaker_turns"], fixture["generated_summary"])
        scores = _rouge_scores(result, fixture["reference_summary"])
        threshold = fixture["thresholds"].get("rouge1_f", 0.0)
        actual = scores["rouge1"]["fmeasure"]
        assert actual >= threshold, (
            f"[{fixture['id']}] ROUGE-1 F1={actual:.4f} below threshold {threshold:.4f}\n"
            f"  Generated: {result}\n"
            f"  Reference: {fixture['reference_summary']}"
        )

    @pytest.mark.parametrize("fixture", [f for f in FIXTURES if f["transcript"]], ids=[f["id"] for f in FIXTURES if f["transcript"]])
    def test_rouge2_fmeasure(self, fixture):
        result = _run_summarize(fixture["transcript"], fixture["speaker_turns"], fixture["generated_summary"])
        scores = _rouge_scores(result, fixture["reference_summary"])
        threshold = fixture["thresholds"].get("rouge2_f", 0.0)
        actual = scores["rouge2"]["fmeasure"]
        assert actual >= threshold, (
            f"[{fixture['id']}] ROUGE-2 F1={actual:.4f} below threshold {threshold:.4f}\n"
            f"  Generated: {result}\n"
            f"  Reference: {fixture['reference_summary']}"
        )

    @pytest.mark.parametrize("fixture", [f for f in FIXTURES if f["transcript"]], ids=[f["id"] for f in FIXTURES if f["transcript"]])
    def test_rougeL_fmeasure(self, fixture):
        result = _run_summarize(fixture["transcript"], fixture["speaker_turns"], fixture["generated_summary"])
        scores = _rouge_scores(result, fixture["reference_summary"])
        threshold = fixture["thresholds"].get("rougeL_f", 0.0)
        actual = scores["rougeL"]["fmeasure"]
        assert actual >= threshold, (
            f"[{fixture['id']}] ROUGE-L F1={actual:.4f} below threshold {threshold:.4f}\n"
            f"  Generated: {result}\n"
            f"  Reference: {fixture['reference_summary']}"
        )


class TestASRSummaryBLEU:
    """BLEU-score evaluations against reference summaries."""

    @pytest.mark.parametrize("fixture", [f for f in FIXTURES if f["transcript"]], ids=[f["id"] for f in FIXTURES if f["transcript"]])
    def test_bleu1(self, fixture):
        result = _run_summarize(fixture["transcript"], fixture["speaker_turns"], fixture["generated_summary"])
        scores = _bleu_scores(result, fixture["reference_summary"])
        threshold = fixture["thresholds"].get("bleu1", 0.0)
        assert scores["bleu1"] >= threshold, (
            f"[{fixture['id']}] BLEU-1={scores['bleu1']:.4f} below threshold {threshold:.4f}"
        )

    @pytest.mark.parametrize("fixture", [f for f in FIXTURES if f["transcript"]], ids=[f["id"] for f in FIXTURES if f["transcript"]])
    def test_bleu2(self, fixture):
        result = _run_summarize(fixture["transcript"], fixture["speaker_turns"], fixture["generated_summary"])
        scores = _bleu_scores(result, fixture["reference_summary"])
        threshold = fixture["thresholds"].get("bleu2", 0.0)
        assert scores["bleu2"] >= threshold, (
            f"[{fixture['id']}] BLEU-2={scores['bleu2']:.4f} below threshold {threshold:.4f}"
        )


class TestASRSummaryFullReport:
    """
    Prints a full metric report for every fixture.
    Always passes — intended for metric logging / CI artifacts.
    """

    @pytest.mark.parametrize("fixture", [f for f in FIXTURES if f["transcript"]], ids=[f["id"] for f in FIXTURES if f["transcript"]])
    def test_print_all_metrics(self, fixture, capsys):
        result = _run_summarize(fixture["transcript"], fixture["speaker_turns"], fixture["generated_summary"])
        rouge  = _rouge_scores(result, fixture["reference_summary"])
        bleu   = _bleu_scores(result, fixture["reference_summary"])

        with capsys.disabled():
            print(f"\n-- Eval: {fixture['id']} ----------------------------------")
            print(f"  Generated : {result[:120]}{'...' if len(result) > 120 else ''}")
            print(f"  Reference : {fixture['reference_summary'][:120]}")
            print(f"  ROUGE-1   : P={rouge['rouge1']['precision']:.4f}  R={rouge['rouge1']['recall']:.4f}  F={rouge['rouge1']['fmeasure']:.4f}")
            print(f"  ROUGE-2   : P={rouge['rouge2']['precision']:.4f}  R={rouge['rouge2']['recall']:.4f}  F={rouge['rouge2']['fmeasure']:.4f}")
            print(f"  ROUGE-L   : P={rouge['rougeL']['precision']:.4f}  R={rouge['rougeL']['recall']:.4f}  F={rouge['rougeL']['fmeasure']:.4f}")
            print(f"  BLEU-1/2/3/4: {bleu['bleu1']:.4f} / {bleu['bleu2']:.4f} / {bleu['bleu3']:.4f} / {bleu['bleu4']:.4f}")
