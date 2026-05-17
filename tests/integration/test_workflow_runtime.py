"""Integration tests for orchestration runtime contract (stubbed, CPU-only)."""
from __future__ import annotations

import types

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.orchestration import workflow_runtime as wr


class _DummyFrame:
    def __init__(self, index: int, ts: float) -> None:
        self.bgr = np.zeros((8, 8, 3), dtype=np.uint8)
        self.index = index
        self.timestamp_s = ts


class _DummySampler:
    def __init__(self, *_args, **_kwargs) -> None:
        self.stats = types.SimpleNamespace(skipped_frames=1, skip_rate=1 / 3)

    def stream(self):
        yield _DummyFrame(0, 0.0)
        yield _DummyFrame(1, 0.1)


class _DummyCVAgent:
    def __init__(self, _cfg) -> None:
        pass

    def load(self):
        return self

    def infer(self, _bgr, _idx, _ts):
        return types.SimpleNamespace(
            frame_index=_idx,
            timestamp_s=_ts,
            inference_ms=12.5,
            detections=[
                types.SimpleNamespace(label="vehicle"),
                types.SimpleNamespace(label="person"),
            ],
        )


class _DummyVSRAgent:
    def __init__(self, _cfg) -> None:
        pass

    def load(self):
        return self

    def infer(self, _bgr, _idx, _ts):
        return types.SimpleNamespace(
            frame_index=_idx,
            timestamp_s=_ts,
            inference_ms=8.5,
            mouth_roi=types.SimpleNamespace(mouth_open=True, lip_aspect_ratio=0.4),
            visemes=[
                types.SimpleNamespace(
                    viseme_id=7, viseme_label="open",
                    phoneme_hypothesis="aa", confidence=0.8,
                )
            ],
            text_hypothesis="aa",
            lm_rescored_text="aa",
            language_id=None,
        )


class _DummyASRAgent:
    def __init__(self, _cfg) -> None:
        pass

    def load(self):
        return self


# Full stub return matching _transcribe_audio_safe() contract
_DUMMY_ASR_DATA = {
    "asr_segments": 1,
    "transcript_preview": "hello world",
    "asr_skipped": False,
    "asr_transcript": "hello world",
    "asr_segment_details": [
        {"start_s": 0.0, "end_s": 1.0, "text": "hello world",
         "speaker": "spk_00", "speaker_name": "Speaker 1"},
    ],
    "asr_speaker_turns": [
        {"speaker_id": "spk_00", "speaker_name": "Speaker 1",
         "start_s": 0.0, "end_s": 1.0, "text": "hello world"},
    ],
}


def _base_cfg(crewai_enabled: bool = True):
    return OmegaConf.create(
        {
            "sampling": {"ssim_threshold": 0.92, "strategy": "adaptive"},
            "orchestration": {
                "backend": "harness_crewai",
                "max_retries": 1,
                "crewai": {"enabled": crewai_enabled},
            },
            "cv_agent": {"device": "cpu"},
            "asr_agent": {"device": "cpu"},
            "vsr_agent": {"device": "cpu"},
        }
    )


def _silent_cfg():
    """Config for silent-video tests: stub LLM backend + crewai enabled."""
    return OmegaConf.create(
        {
            "sampling": {"ssim_threshold": 0.92, "strategy": "adaptive"},
            "orchestration": {
                "backend": "harness_crewai",
                "max_retries": 1,
                "crewai": {"enabled": True},
            },
            "cv_agent": {"device": "cpu"},
            "asr_agent": {"device": "cpu"},
            "vsr_agent": {"device": "cpu"},
            "llm": {
                "backend": "stub",
                "ollama": {"model": "stub", "host": "http://localhost:11434"},
                "claude": {"model": "stub"},
            },
        }
    )


def _patch_runtime(monkeypatch, cfg):
    monkeypatch.setattr(wr, "load_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(wr, "AdaptiveSampler", _DummySampler)
    monkeypatch.setattr(wr, "CVAgent", _DummyCVAgent)
    monkeypatch.setattr(wr, "ASRAgent", _DummyASRAgent)
    monkeypatch.setattr(wr, "VSRAgent", _DummyVSRAgent)
    monkeypatch.setattr(
        wr,
        "_transcribe_audio_safe",
        lambda *_args, **_kwargs: _DUMMY_ASR_DATA,
    )
    # Stub out the vision LLM call — _DummyVSRAgent.mouth_roi has no .crop
    monkeypatch.setattr(
        wr,
        "_vsr_lip_reading_with_llm",
        lambda *_args, **_kwargs: "",
    )
    # Stub LLM client creation — avoids probing Ollama/Claude during tests
    monkeypatch.setattr(wr, "_make_llm_client", lambda _metrics: wr.LLMClient.stub())
    # Stub secondary client availability check — avoids 3s Ollama probe
    monkeypatch.setattr(wr.LLMClient, "ollama", classmethod(lambda cls, **_kw: wr.LLMClient.stub()))


def test_runtime_stage_contract(monkeypatch):
    _patch_runtime(monkeypatch, _base_cfg(crewai_enabled=True))
    monkeypatch.setattr(
        wr,
        "_run_crewai_synthesis",
        lambda _metrics, _client=None: {"engine": "crewai", "enabled": True, "report": "ok"},
    )

    result = wr.run_video_workflow("dummy.mp4")

    assert result["frames_processed"] == 2
    assert result["frames_skipped"] == 1
    assert result["total_detections"] == 4
    assert len(result["cv_detections"]) == 2
    assert result["cv_detections"][0]["detections"][0]["label"] == "vehicle"
    assert len(result["vsr_results"]) == 2
    assert result["vsr_results"][0]["visemes"][0]["viseme_label"] == "open"
    assert result["crew_enabled"] is True
    assert result["crew_engine"] == "crewai"

    stages = result["orchestration"]["stages"]
    assert [s["name"] for s in stages] == ["load_agents", "run_cv_vsr", "run_asr"]
    assert all(s["success"] for s in stages)
    assert all(s["attempts"] >= 1 for s in stages)


def test_runtime_local_backend_disables_crewai(monkeypatch):
    _patch_runtime(monkeypatch, _base_cfg(crewai_enabled=True))

    result = wr.run_video_workflow("dummy.mp4", backend="local")

    assert result["workflow_backend"] == "local"
    assert result["crew_enabled"] is False
    assert result["crew_engine"] == "deterministic"


def test_runtime_crewai_config_flag_disables_crewai(monkeypatch):
    _patch_runtime(monkeypatch, _base_cfg(crewai_enabled=False))

    def _should_not_run(_metrics, _client=None):
        raise AssertionError("CrewAI synthesis should not run when disabled in config")

    monkeypatch.setattr(wr, "_run_crewai_synthesis", _should_not_run)

    result = wr.run_video_workflow("dummy.mp4", backend="harness_crewai")

    assert result["workflow_backend"] == "harness_crewai"
    assert result["crew_enabled"] is False
    assert result["crew_engine"] == "deterministic"


def test_runtime_asr_transcript_surfaced(monkeypatch):
    """ASR transcript from stub appears in the result payload."""
    _patch_runtime(monkeypatch, _base_cfg(crewai_enabled=False))

    result = wr.run_video_workflow("dummy.mp4")

    assert result.get("asr_transcript") == "hello world"
    assert result.get("transcript_preview") == "hello world"


def test_runtime_vsr_results_have_mouth_roi(monkeypatch):
    """VSR results include serialized mouth_roi fields."""
    _patch_runtime(monkeypatch, _base_cfg(crewai_enabled=False))

    result = wr.run_video_workflow("dummy.mp4")

    for vsr_frame in result["vsr_results"]:
        assert "mouth_open" in vsr_frame or "mouth_roi" in vsr_frame


def test_runtime_with_text_model_override(monkeypatch):
    """llm_text_model kwarg is accepted without error (model client creation fails gracefully)."""
    _patch_runtime(monkeypatch, _base_cfg(crewai_enabled=False))

    result = wr.run_video_workflow(
        "dummy.mp4",
        llm_ollama_model="llama3.2-vision",
        llm_text_model="llama3.2",
    )
    assert result["frames_processed"] == 2


# ─── Silent-video (audio-absent) VSR transcription tests ─────────────────────

# ASR data that signals no speech was found (silent / muted video)
_SILENT_ASR_DATA = {
    "asr_segments": 0,
    "transcript_preview": "",
    "asr_skipped": True,
    "asr_transcript": "",
    "asr_segment_details": [],
    "asr_speaker_turns": [],
}


class _DummyVSRAgentWithCrops:
    """
    VSR stub whose mouth_roi includes a real numpy .crop array so that
    _vsr_lip_reading_with_llm can encode it as JPEG without crashing.
    """

    def __init__(self, _cfg) -> None:
        pass

    def load(self):
        return self

    def infer(self, _bgr, idx, ts):
        import types as _t
        import numpy as _np
        roi = _t.SimpleNamespace(
            crop=_np.zeros((112, 112, 3), dtype=_np.uint8),
            landmarks=_np.zeros((20, 2), dtype=_np.float32),
            mouth_open=True,
            lip_aspect_ratio=0.4,
            head_pose_deg=None,
        )
        return _t.SimpleNamespace(
            frame_index=idx,
            timestamp_s=ts,
            inference_ms=6.0,
            mouth_roi=roi,
            visemes=[_t.SimpleNamespace(
                viseme_id=7, viseme_label="open",
                phoneme_hypothesis="aa", confidence=0.75,
            )],
            text_hypothesis="",
            lm_rescored_text="",
            language_id=None,
        )


def _patch_runtime_silent(monkeypatch, cfg, vsr_llm_return: str = ""):
    """Patch runtime for a silent-video scenario.

    - ASR is skipped (no audio)
    - VSR agent returns frames with real numpy crops
    - _vsr_lip_reading_with_llm is stubbed to return `vsr_llm_return`
    """
    monkeypatch.setattr(wr, "load_config", lambda *_a, **_kw: cfg)
    monkeypatch.setattr(wr, "AdaptiveSampler", _DummySampler)
    monkeypatch.setattr(wr, "CVAgent", _DummyCVAgent)
    monkeypatch.setattr(wr, "ASRAgent", _DummyASRAgent)
    monkeypatch.setattr(wr, "VSRAgent", _DummyVSRAgentWithCrops)
    monkeypatch.setattr(
        wr, "_transcribe_audio_safe",
        lambda *_a, **_kw: _SILENT_ASR_DATA,
    )
    monkeypatch.setattr(
        wr, "_vsr_lip_reading_with_llm",
        lambda *_a, **_kw: vsr_llm_return,
    )
    monkeypatch.setattr(
        wr, "_run_crewai_synthesis",
        lambda _m, _client=None: {"engine": "crewai", "enabled": True, "report": ""},
    )


def test_silent_video_asr_is_empty(monkeypatch):
    """When ASR is skipped (no audio), asr_transcript must be empty."""
    _patch_runtime_silent(monkeypatch, _silent_cfg())

    result = wr.run_video_workflow("silent.mp4")

    assert result["asr_transcript"] == ""
    assert result.get("asr_segments", 0) == 0


def test_silent_video_vsr_transcript_is_primary(monkeypatch):
    """
    With no ASR audio, the vsr_transcript from lip-reading becomes the only
    speech signal.  The result must surface it in 'vsr_transcript'.
    """
    _patch_runtime_silent(
        monkeypatch,
        _silent_cfg(),
        vsr_llm_return="good morning everyone",
    )

    result = wr.run_video_workflow("silent.mp4")

    assert result["vsr_transcript"] == "good morning everyone"
    assert result["asr_transcript"] == ""


def test_silent_video_vsr_frames_have_mouth_roi(monkeypatch):
    """Each VSR result frame must carry a mouth_roi when a face was detected."""
    _patch_runtime_silent(monkeypatch, _silent_cfg())

    result = wr.run_video_workflow("silent.mp4")

    assert len(result["vsr_results"]) == 2
    for frame in result["vsr_results"]:
        # serialized form includes mouth_open
        assert "mouth_open" in frame or "mouth_roi" in frame


def test_silent_video_vsr_transcript_empty_when_llm_returns_nothing(monkeypatch):
    """If lip-reading LLM returns empty string, vsr_transcript stays empty."""
    _patch_runtime_silent(
        monkeypatch,
        _silent_cfg(),
        vsr_llm_return="",
    )

    result = wr.run_video_workflow("silent.mp4")

    assert result["vsr_transcript"] == ""


def test_silent_video_asr_timestamp_alignment_skipped(monkeypatch):
    """
    With no ASR segments, per-frame text_hypothesis should NOT be back-filled
    from ASR (there is nothing to align to).
    """
    _patch_runtime_silent(monkeypatch, _silent_cfg())

    result = wr.run_video_workflow("silent.mp4")

    for frame in result["vsr_results"]:
        # text_hypothesis may be "" or a heuristic phoneme — never "hello world"
        text = frame.get("text", "") or ""
        assert "hello world" not in text


def test_silent_video_pipeline_completes_without_error(monkeypatch):
    """End-to-end: silent video must not raise an exception or set crew_error."""
    _patch_runtime_silent(monkeypatch, _silent_cfg())

    result = wr.run_video_workflow("silent.mp4")

    assert "error" not in result
    assert result["frames_processed"] == 2
    assert result["workflow_backend"] == "harness_crewai"


def test_silent_video_with_speaking_fixture(monkeypatch, speaking_video_path):
    """
    Run the real AdaptiveSampler on the synthetic speaking_video_path fixture.
    ASR is stubbed as silent; VSR lip-reading returns a phrase.
    Verifies the full sampler → agents → result chain on a real file.
    """
    cfg = _silent_cfg()
    monkeypatch.setattr(wr, "load_config", lambda *_a, **_kw: cfg)
    monkeypatch.setattr(wr, "CVAgent", _DummyCVAgent)
    monkeypatch.setattr(wr, "ASRAgent", _DummyASRAgent)
    monkeypatch.setattr(wr, "VSRAgent", _DummyVSRAgentWithCrops)
    monkeypatch.setattr(
        wr, "_transcribe_audio_safe",
        lambda *_a, **_kw: _SILENT_ASR_DATA,
    )
    monkeypatch.setattr(
        wr, "_vsr_lip_reading_with_llm",
        lambda *_a, **_kw: "hello there",
    )
    monkeypatch.setattr(
        wr, "_run_crewai_synthesis",
        lambda _m, _client=None: {"engine": "crewai", "enabled": True, "report": ""},
    )

    result = wr.run_video_workflow(str(speaking_video_path))

    assert result["frames_processed"] >= 1
    assert result["asr_transcript"] == ""
    assert result["vsr_transcript"] == "hello there"
