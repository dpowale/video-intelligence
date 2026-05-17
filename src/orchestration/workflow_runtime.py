"""Workflow runtime with Harness-style stages and optional CrewAI synthesis."""
from __future__ import annotations

import bisect
import os
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from src.agents.asr_agent import ASRAgent
from src.agents.cv_agent import CVAgent
from src.agents.vsr_agent import VSRAgent
from src.fusion.llm_client import LLMClient
from src.utils.common import get_logger, load_config
from src.utils.sampler import AdaptiveSampler

log = get_logger(__name__)

# Module-level agent cache — models are expensive to load; reuse across requests.
_agent_cache: dict[tuple, tuple["CVAgent", "ASRAgent", "VSRAgent"]] = {}
_agent_cache_lock = threading.Lock()


@dataclass
class StageResult:
    name: str
    success: bool
    attempts: int
    elapsed_s: float
    error: str | None = None


class HarnessLikeOrchestrator:
    """Simple stage orchestrator inspired by Harness pipeline execution."""

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries

    def run_stage(self, name: str, fn: Callable[[], Any]) -> tuple[Any, StageResult]:
        last_error: Exception | None = None
        t0 = time.perf_counter()
        for attempt in range(1, self.max_retries + 2):
            try:
                out = fn()
                elapsed = time.perf_counter() - t0
                return out, StageResult(name=name, success=True, attempts=attempt, elapsed_s=elapsed)
            except Exception as exc:
                last_error = exc
                log.warning("Stage %s failed attempt %d: %s", name, attempt, exc)
                if attempt > self.max_retries:
                    break
        elapsed = time.perf_counter() - t0
        return None, StageResult(
            name=name,
            success=False,
            attempts=self.max_retries + 1,
            elapsed_s=elapsed,
            error=str(last_error),
        )


def _serialize_detection(det: Any) -> dict[str, Any]:
    if hasattr(det, "to_dict"):
        return det.to_dict()
    if hasattr(det, "__dict__"):
        return dict(det.__dict__)
    return {"value": det}


def _serialize_cv_result(result: Any, frame_shape: tuple[int, int] | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {
        "frame_index": getattr(result, "frame_index", None),
        "timestamp_s": getattr(result, "timestamp_s", None),
        "inference_ms": getattr(result, "inference_ms", None),
        "detections": [_serialize_detection(det) for det in getattr(result, "detections", [])],
        "faces": [
            face.to_dict() if hasattr(face, "to_dict") else dict(face.__dict__)
            for face in getattr(result, "faces", [])
        ],
    }
    if frame_shape is not None:
        d["frame_h"], d["frame_w"] = int(frame_shape[0]), int(frame_shape[1])
    return d


def _serialize_vsr_result(result: Any) -> dict[str, Any]:
    mouth_roi = getattr(result, "mouth_roi", None)
    return {
        "frame_index": getattr(result, "frame_index", None),
        "timestamp_s": getattr(result, "timestamp_s", None),
        "inference_ms": getattr(result, "inference_ms", None),
        "text_hypothesis": getattr(result, "text_hypothesis", ""),
        "lm_rescored_text": getattr(result, "lm_rescored_text", ""),
        "language_id": getattr(result, "language_id", None),
        "mouth_open": getattr(mouth_roi, "mouth_open", None),
        "lip_aspect_ratio": getattr(mouth_roi, "lip_aspect_ratio", None),
        "visemes": [
            {
                "viseme_id": getattr(viseme, "viseme_id", None),
                "viseme_label": getattr(viseme, "viseme_label", None),
                "phoneme_hypothesis": getattr(viseme, "phoneme_hypothesis", None),
                "confidence": getattr(viseme, "confidence", None),
            }
            for viseme in getattr(result, "visemes", [])
        ],
    }


def _transcribe_audio_safe(video_path: str, asr_agent: ASRAgent) -> dict[str, Any]:
    def _asr_payload(asr_result: Any) -> dict[str, Any]:
        segment_details = [
            {
                "speaker_id": seg.speaker_id,
                "speaker_name": seg.speaker_name or "Speaker 1",
                "start_s": seg.start_s,
                "end_s": seg.end_s,
                "text": seg.text,
            }
            for seg in asr_result.segments
        ]

        speaker_turns: list[dict[str, Any]] = []
        for seg in segment_details:
            if not seg["text"]:
                continue
            if (
                speaker_turns
                and speaker_turns[-1]["speaker_name"] == seg["speaker_name"]
                and seg["start_s"] - speaker_turns[-1]["end_s"] <= 0.35
            ):
                speaker_turns[-1]["text"] = (
                    f"{speaker_turns[-1]['text']} {seg['text']}".strip()
                )
                speaker_turns[-1]["end_s"] = seg["end_s"]
            else:
                speaker_turns.append(
                    {
                        "speaker_name": seg["speaker_name"],
                        "start_s": seg["start_s"],
                        "end_s": seg["end_s"],
                        "text": seg["text"],
                    }
                )

        full_transcript = "\n".join(seg["text"] for seg in segment_details if seg["text"])
        return {
            "asr_segments": len(asr_result.segments),
            "transcript_preview": asr_result.segments[0].text if asr_result.segments else "",
            "asr_transcript": full_transcript,
            "asr_segment_details": segment_details,
            "asr_speaker_turns": speaker_turns,
            "asr_skipped": False,
        }

    if shutil.which("ffmpeg") is None:
        log.warning("ffmpeg not found in PATH - trying direct ASR on input video")
        try:
            asr_result = asr_agent.transcribe(video_path)
            return _asr_payload(asr_result)
        except Exception as exc:
            log.warning("Direct ASR failed without ffmpeg: %s", exc)
            return {
                "asr_segments": 0,
                "transcript_preview": "",
                "asr_transcript": "",
                "asr_segment_details": [],
                "asr_speaker_turns": [],
                "asr_skipped": True,
                "asr_error": f"ffmpeg missing and direct ASR failed: {exc}",
            }

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_path = f.name

    try:
        proc = subprocess.run(
            ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", "-y", audio_path],
            capture_output=True,
            timeout=60,
        )
        if proc.returncode != 0:
            stderr_tail = (proc.stderr.decode(errors="ignore") if proc.stderr else "")[-300:]
            log.warning("ffmpeg exited with code %d - trying direct ASR fallback", proc.returncode)
            try:
                asr_result = asr_agent.transcribe(video_path)
                return _asr_payload(asr_result)
            except Exception as exc:
                return {
                    "asr_segments": 0,
                    "transcript_preview": "",
                    "asr_transcript": "",
                    "asr_segment_details": [],
                    "asr_speaker_turns": [],
                    "asr_skipped": True,
                    "asr_error": (
                        f"ffmpeg failed (code {proc.returncode}): {stderr_tail.strip()} | "
                        f"direct ASR failed: {exc}"
                    ),
                }

        asr_result = asr_agent.transcribe(audio_path)
        return _asr_payload(asr_result)
    except Exception as exc:
        log.warning("Audio transcription failed - skipping ASR: %s", exc)
        return {
            "asr_segments": 0,
            "transcript_preview": "",
            "asr_transcript": "",
            "asr_segment_details": [],
            "asr_speaker_turns": [],
            "asr_skipped": True,
            "asr_error": str(exc),
        }
    finally:
        try:
            os.unlink(audio_path)
        except Exception:
            pass

def _make_llm_client(metrics: dict[str, Any]) -> LLMClient:
    """Construct an LLM client from the llm_* fields in metrics."""
    backend = str(metrics.get("llm_backend", "auto")).lower()
    ollama_model = str(metrics.get("llm_ollama_model", "llama3.2:latest"))
    ollama_host = metrics.get("llm_ollama_host")
    claude_model = str(metrics.get("llm_claude_model", "claude-sonnet-4-20250514"))
    if backend == "ollama":
        return LLMClient.ollama(model=ollama_model, host=ollama_host)
    if backend == "claude":
        return LLMClient.claude(model=claude_model)
    if backend == "stub":
        return LLMClient.stub()
    return LLMClient.auto(ollama_model=ollama_model)


def _summarize_asr_with_llm(
    transcript: str,
    speaker_turns: list[dict[str, Any]],
    client: LLMClient,
) -> str:
    """Ask the LLM to produce a concise summary of the ASR transcript."""
    if not transcript.strip():
        return ""
    turns_text = "\n".join(
        f"  [{t.get('speaker_name', '?')} {t.get('start_s', 0):.1f}s-{t.get('end_s', 0):.1f}s]: {t.get('text', '')}"
        for t in speaker_turns[:30]
    )
    prompt = (
        "Summarize the following spoken transcription.\n"
        "Identify: (1) main topics discussed, (2) speakers and their roles, "
        "(3) key facts, decisions, or names mentioned, (4) overall tone/sentiment.\n\n"
        f"TRANSCRIPT:\n{transcript[:4000]}\n"
        + (f"\nSPEAKER TURNS:\n{turns_text}\n" if turns_text else "")
        + "\nWrite a concise 3-5 sentence summary."
    )
    try:
        resp = client.chat(prompt=prompt, temperature=0.2)
        return (resp.text or "").strip()
    except Exception as exc:
        log.warning("ASR summary LLM call failed: %s", exc)
        return ""


# Known vision-capable Ollama model name fragments (order = preference)
_VISION_MODEL_FRAGMENTS = ["llama3.2-vision", "llava", "bakllava", "moondream", "minicpm-v", "cogvlm"]


def _resolve_vision_model(ollama_host: str, current_model: str) -> str:
    """
    Return the model name to use for vision inference.

    If `current_model` is vision-capable, returns it unchanged.
    Otherwise queries Ollama's /api/tags and returns the first installed vision model.
    Falls back to `current_model` if none found (Ollama will simply ignore the images).
    """
    # Is the current model already vision-capable?
    low = current_model.lower()
    if any(frag in low for frag in _VISION_MODEL_FRAGMENTS):
        return current_model

    # Query installed models
    try:
        import json as _json
        import urllib.request as _ur
        with _ur.urlopen(f"{ollama_host}/api/tags", timeout=4) as r:
            installed = [m["name"] for m in _json.loads(r.read()).get("models", [])]
    except Exception:
        installed = []

    for model in installed:
        if any(frag in model.lower() for frag in _VISION_MODEL_FRAGMENTS):
            log.info(
                "VSR lip-reading: auto-selected vision model '%s' (requested '%s' is text-only)",
                model, current_model,
            )
            return model

    log.warning(
        "VSR lip-reading: no vision model found in Ollama. "
        "Falling back to '%s' — images will be ignored.", current_model,
    )
    return current_model


def _vsr_lip_reading_with_llm(
    vsr_results: list[Any],  # list[VSRResult] — raw objects with mouth_roi.crop
    client: LLMClient,
    max_frames: int = 3,
) -> str:
    """
    Infer spoken text from mouth-crop images using an Ollama vision model.

    Always runs regardless of audio presence.  Selects up to `max_frames` frames
    spread temporally, upsamples 112×112 crops to 256×256 for better visibility,
    then sends them in a single Ollama vision request with a lip-reading prompt.
    Prefers gpt-oss — the strongest installed vision model — for this task.
    Returns the estimated spoken text, or '' if no speech is apparent.
    """
    import base64

    # All frames where a mouth ROI was detected (not just open ones)
    # Prioritise open-mouth frames but include closed-mouth context too
    all_mouth_frames = [
        r for r in vsr_results
        if getattr(r, "mouth_roi", None) is not None
    ]
    if not all_mouth_frames:
        return ""

    # Prefer open-mouth frames, fall back to all mouth frames
    open_frames = [r for r in all_mouth_frames if getattr(r.mouth_roi, "mouth_open", False)]
    candidate_frames = open_frames if open_frames else all_mouth_frames

    # Temporally-spread sample to capture the full utterance arc
    if len(candidate_frames) > max_frames:
        step = len(candidate_frames) / max_frames
        candidate_frames = [candidate_frames[int(i * step)] for i in range(max_frames)]

    # Encode crops as JPEG base64 — upsample 112×112 → 256×256 for better model visibility
    import cv2 as _cv2
    import numpy as _np
    images_b64: list[str] = []
    timestamps: list[str] = []
    for r in candidate_frames:
        crop = r.mouth_roi.crop  # (112, 112, 3) BGR uint8
        upsampled = _cv2.resize(crop, (256, 256), interpolation=_cv2.INTER_CUBIC)
        ok, buf = _cv2.imencode(".jpg", upsampled, [_cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            images_b64.append(base64.b64encode(bytes(buf)).decode())
            timestamps.append(f"{getattr(r, 'timestamp_s', 0.0):.2f}s")

    if not images_b64:
        return ""

    n = len(images_b64)
    prompt = (
        f"You are an expert lip-reader. You are given {n} sequential close-up images "
        f"of a speaker's mouth (timestamps: {', '.join(timestamps)}), extracted from a video.\n\n"
        "Instructions:\n"
        "1. Examine each image carefully for lip shape, teeth visibility, tongue position, "
        "mouth aperture, and lip rounding.\n"
        "2. Map the sequence of mouth shapes to the most likely spoken phonemes and words.\n"
        "3. Produce the COMPLETE sentence or phrase spoken across all frames.\n"
        "4. Reply with ONLY the final text (e.g. 'good morning everyone').\n"
        "5. If no speech is detectable, reply with exactly: silence\n\n"
        "Do NOT include explanations, timestamps, or frame labels."
    )

    try:
        backend = client._backend
        if not hasattr(backend, "host"):
            # Non-Ollama backend — send first image only via standard interface
            resp = client.chat(prompt=prompt, image_b64=images_b64[0], temperature=0.1)
            text = (resp.text or "").strip().strip("\"'")
            return "" if text.lower() in ("silence", "") else text

        # Use the model that was already selected for the vision client (e.g. llama3.2-vision).
        # Do NOT override with a second Ollama tag query here — the caller already chose
        # the right vision model via _vision_client, which reflects the user's UI selection
        # and the auto-selection logic in run_video_workflow.
        vision_model = backend.model

        log.info("VSR lip-reading: using model '%s' on %d crops", vision_model, n)

        import json as _json, urllib.request as _ur
        messages = [{"role": "user", "content": prompt, "images": images_b64}]
        payload = _json.dumps({
            "model": vision_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9},
        }).encode()
        req = _ur.Request(
            f"{backend.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _ur.urlopen(req, timeout=45) as r:
            data = _json.loads(r.read())
        text = (data.get("message", {}).get("content", "") or "").strip().strip("\"'")
        return "" if text.lower() in ("silence", "") else text
    except Exception as exc:
        log.warning("VSR vision lip-reading LLM call failed: %s", exc)
        return ""


def _build_synthesis_prompt(metrics: dict[str, Any]) -> str:
    """Build a rich synthesis prompt from actual agent output data."""
    parts: list[str] = []

    # CV — top distinct detections across all frames
    cv_frames = metrics.get("cv_detections", [])
    all_dets: list[dict] = []
    best_by_class: dict[str, float] = {}
    for frame in cv_frames:
        for det in frame.get("detections", []):
            all_dets.append(det)
            label = det.get('class', '?')
            conf = det.get('conf', 0)
            if label not in best_by_class or conf > best_by_class[label]:
                best_by_class[label] = conf

    if best_by_class:
        # Sort distinct classes by their highest confidence
        top_classes = sorted(best_by_class.items(), key=lambda x: x[1], reverse=True)[:15]
        det_lines = "; ".join(
            f"{label} ({conf:.2f})"
            for label, conf in top_classes
        )
        parts.append(
            f"CV AGENT ({len(cv_frames)} frame(s), {len(all_dets)} detection(s) total):\n"
            f"  Distinct top detections: {det_lines}"
        )
    else:
        parts.append(f"CV AGENT: {metrics.get('frames_processed', 0)} frame(s) sampled — no detections.")

    # ASR — speaker turns with text
    turns = metrics.get("asr_speaker_turns", [])
    if turns:
        turn_lines = "\n".join(
            f"  [{t.get('speaker_name', '?')} {t.get('start_s', 0):.1f}s–{t.get('end_s', 0):.1f}s]: {t.get('text', '')}"
            for t in turns[:20]
        )
        parts.append(f"ASR AGENT ({len(turns)} speaker turn(s)):\n{turn_lines}")
    elif metrics.get("asr_skipped"):
        parts.append("ASR AGENT: skipped (audio extraction failed).")
    else:
        parts.append("ASR AGENT: no speech detected.")

    # ASR summary (if LLM pre-computed it)
    asr_summary = metrics.get("asr_summary", "")
    if asr_summary:
        parts.append(f"ASR SUMMARY (LLM-generated):\n  {asr_summary}")

    # VSR — mouth activity + lip-reading text
    vsr_frames = metrics.get("vsr_results", [])
    open_count = sum(1 for f in vsr_frames if f.get("mouth_open"))
    vsr_transcript = metrics.get("vsr_transcript", "")
    if vsr_frames:
        vsr_line = (
            f"VSR AGENT ({len(vsr_frames)} frame(s)): mouth open in {open_count} frame(s)"
        )
        if vsr_transcript:
            vsr_line += f"; lip-reading text: '{vsr_transcript}'"
        parts.append(vsr_line)

    body = "\n\n".join(parts)
    return (
        "You are generating an operator-ready multimodal intelligence summary. "
        "Use plain language and keep it concise.\n\n"
        "Return sections exactly in this format:\n"
        "SCENE: ...\n"
        "SPEAKERS: ...\n"
        "OBJECTS: ...\n"
        "CROSS-MODAL: ...\n"
        "CONFIDENCE: ...\n"
        "ACTIONS: ...\n\n"
        f"{body}\n\nSynthesize the above into a structured report."
    )


def _run_crewai_synthesis(
    metrics: dict[str, Any],
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """Run synthesis via configured LLM backend (Ollama-first), with deterministic fallback."""
    deterministic_summary = (
        "Workflow summary: "
        f"processed {metrics['frames_processed']} sampled frames, "
        f"skipped {metrics['frames_skipped']} frames ({metrics['skip_rate_pct']}%), "
        f"found {metrics['total_detections']} detections, "
        f"ASR segments={metrics['asr_segments']}"
        + (
            f", transcript preview='{metrics['transcript_preview'][:120]}'"
            if metrics.get("transcript_preview")
            else ", transcript preview unavailable"
        )
        + "."
    )

    prompt = _build_synthesis_prompt(metrics)

    try:
        if client is None:
            client = _make_llm_client(metrics)

        resp = client.chat(prompt=prompt, temperature=0.2)
        report_text = (resp.text or "").strip() or deterministic_summary
        return {
            "engine": f"llm:{resp.backend}",
            "enabled": True,
            "report": report_text,
            "model": resp.model,
        }
    except Exception as exc:
        log.warning("LLM synthesis failed - deterministic summary used: %s", exc)
        return {
            "engine": "deterministic",
            "enabled": False,
            "report": deterministic_summary,
            "error": str(exc),
        }


def run_video_workflow(
    video_path: str,
    config_path: str = "configs/default.yaml",
    backend: str | None = None,
    llm_backend: str | None = None,
    llm_ollama_model: str | None = None,   # vision model — used for VSR lip-reading
    llm_text_model: str | None = None,     # text model — used for ASR summary + synthesis
    llm_ollama_host: str | None = None,
    llm_claude_model: str | None = None,
    enabled_agents: set[str] | None = None,  # {"cv", "asr", "vsr"} — None means all
    asr_model: str | None = None,            # override ASR model (allow finetune selection)
) -> dict[str, Any]:
    """Run video intelligence pipeline with Harness-style orchestration."""
    cfg = load_config(config_path)
    
    if asr_model:
        cfg["asr_agent"]["model"] = asr_model

    # Resolve which agents to run (default: all)
    _ALL_AGENTS = {"cv", "asr", "vsr"}
    run_agents: set[str] = _ALL_AGENTS if not enabled_agents else (
        {a.strip().lower() for a in enabled_agents} & _ALL_AGENTS
    )
    log.info("Enabled agents: %s", sorted(run_agents))
    selected_backend = backend or cfg.orchestration.get("backend", "harness_crewai")
    crewai_enabled = bool(cfg.orchestration.get("crewai", {}).get("enabled", True))

    orchestrator = HarnessLikeOrchestrator(
        max_retries=int(cfg.orchestration.get("max_retries", 1))
    )

    if selected_backend not in {"harness_crewai", "crewai", "local", "temporal"}:
        log.warning("Unknown orchestration backend '%s' - using harness_crewai", selected_backend)
        selected_backend = "harness_crewai"

    if selected_backend == "temporal":
        log.warning("Temporal backend selected but runtime is configured for local Harness-like mode")

    stages: list[StageResult] = []

    _cache_key = (frozenset(run_agents), str(cfg["asr_agent"].get("model", "base")))

    def _load_agents() -> tuple[CVAgent, ASRAgent, VSRAgent]:
        with _agent_cache_lock:
            cached = _agent_cache.get(_cache_key)
        if cached is not None:
            log.info("Reusing cached agents (key=%s)", _cache_key)
            return cached
        cv  = CVAgent(cfg).load()  if "cv"  in run_agents else CVAgent(cfg)
        asr = ASRAgent(cfg).load() if "asr" in run_agents else ASRAgent(cfg)
        vsr = VSRAgent(cfg).load() if "vsr" in run_agents else VSRAgent(cfg)
        result = (cv, asr, vsr)
        with _agent_cache_lock:
            _agent_cache[_cache_key] = result
        log.info("Agents cached (key=%s)", _cache_key)
        return result

    agents, stage = orchestrator.run_stage("load_agents", _load_agents)
    stages.append(stage)
    if not stage.success:
        raise RuntimeError(f"Stage load_agents failed: {stage.error}")

    cv_agent, asr_agent, vsr_agent = agents

    sampler = AdaptiveSampler(
        video_path,
        ssim_threshold=cfg.sampling.ssim_threshold,
        fixed_fps=cfg.sampling.get("fixed_fps", 2.0),
        strategy=cfg.sampling.strategy,
    )

    cv_results = []
    vsr_results = []
    frame_shapes: dict[int, tuple[int, int]] = {}  # frame_index → (h, w)

    def _run_frame_agents() -> None:
        # Submit all frames before collecting so the pool can process in parallel.
        all_futures: list[tuple[str, Any]] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for frame in sampler.stream():
                frame_shapes[frame.index] = frame.bgr.shape[:2]
                if "cv" in run_agents:
                    all_futures.append(("cv", pool.submit(cv_agent.infer, frame.bgr, frame.index, frame.timestamp_s)))
                if "vsr" in run_agents:
                    all_futures.append(("vsr", pool.submit(vsr_agent.infer, frame.bgr, frame.index, frame.timestamp_s)))
            # pool.__exit__ waits for all submitted futures before continuing
        for name, fut in all_futures:
            if name == "cv":
                cv_results.append(fut.result())
            else:
                vsr_results.append(fut.result())

    with ThreadPoolExecutor(max_workers=4) as pool:
        # Submit ASR only if enabled; otherwise use a no-op future
        if "asr" in run_agents:
            asr_future = pool.submit(_transcribe_audio_safe, video_path, asr_agent)
        else:
            import concurrent.futures as _cf
            _skipped_asr: dict = {
                "asr_segments": 0, "transcript_preview": "",
                "asr_transcript": "", "asr_segment_details": [],
                "asr_speaker_turns": [], "asr_skipped": True,
            }
            asr_future = pool.submit(lambda: _skipped_asr)

        _, stage = orchestrator.run_stage("run_cv_vsr", _run_frame_agents)
        stages.append(stage)
        if not stage.success:
            raise RuntimeError(f"Stage run_cv_vsr failed: {stage.error}")

        asr_data, stage = orchestrator.run_stage(
            "run_asr",
            asr_future.result,
        )
        stages.append(stage)
        if not stage.success:
            asr_data = {
                "asr_segments": 0,
                "transcript_preview": "",
                "asr_transcript": "",
                "asr_segment_details": [],
                "asr_speaker_turns": [],
                "asr_skipped": True,
            }

    llm_cfg = cfg.get("llm", {})

    # ── Resolve effective LLM config (CLI overrides YAML) ─────────────────────
    eff_llm_backend = llm_backend or llm_cfg.get("backend", "auto")
    eff_ollama_model = llm_ollama_model or llm_cfg.get("ollama", {}).get("model", "llama3.2:latest")
    # text_model is for ASR summary + synthesis (no images needed); falls back to vision model
    eff_text_model = llm_text_model or llm_cfg.get("ollama", {}).get("text_model", "") or eff_ollama_model
    eff_ollama_host = llm_ollama_host or llm_cfg.get("ollama", {}).get("host", "http://localhost:11434")
    eff_claude_model = llm_claude_model or llm_cfg.get("claude", {}).get("model", "claude-sonnet-4-20250514")

    _llm_run = selected_backend in {"harness_crewai", "crewai", "temporal"} and crewai_enabled

    # Two separate clients:
    #   _vision_client — VSR lip-reading (needs images; auto-selects vision model from Ollama)
    #   _text_client   — ASR summary + synthesis (text only; uses eff_text_model)
    _vision_client: LLMClient | None = None
    _text_client: LLMClient | None = None
    if _llm_run:
        try:
            _vision_client = _make_llm_client({
                "llm_backend": eff_llm_backend,
                "llm_ollama_model": eff_ollama_model,
                "llm_ollama_host": eff_ollama_host,
                "llm_claude_model": eff_claude_model,
            })
        except Exception as _e:
            log.warning("Could not initialise vision LLM client: %s", _e)
        try:
            _text_client = _make_llm_client({
                "llm_backend": eff_llm_backend,
                "llm_ollama_model": eff_text_model,
                "llm_ollama_host": eff_ollama_host,
                "llm_claude_model": eff_claude_model,
            })
        except Exception as _e:
            log.warning("Could not initialise text LLM client: %s", _e)

    # Alias for backwards compat — vision tasks use vision client
    _llm_client = _vision_client

    # VSR lip-reading via vision model — must run BEFORE serializing vsr_results
    # so we still have raw VSRResult objects with mouth_roi.crop numpy arrays.
    # Single pass over vsr_results to collect all stats.
    _audio_absent = asr_data.get("asr_skipped") or not asr_data.get("asr_transcript", "").strip()
    vsr_transcript = ""
    vsr_vision_model = ""
    _mouth_frame_count = 0
    _mouth_open_count = 0
    for _r in vsr_results:
        if getattr(_r, "mouth_roi", None) is not None:
            _mouth_frame_count += 1
            if getattr(_r.mouth_roi, "mouth_open", False):
                _mouth_open_count += 1
    _has_mouth_frames = _mouth_frame_count > 0
    if _llm_client and _has_mouth_frames and _audio_absent:
        log.info(
            "Running VSR vision lip-reading (%s) on %d mouth frames (%d open)",
            "audio absent" if _audio_absent else "audio present",
            _mouth_frame_count,
            _mouth_open_count,
        )
        # Record which vision model will be used
        _vsr_backend = getattr(_llm_client, "_backend", None)
        if _vsr_backend is not None and hasattr(_vsr_backend, "host"):
            # Model selection mirrors _vsr_lip_reading_with_llm preference order
            _PREFERRED_VSR = ["gpt-oss", "llama3.2-vision", "llava", "bakllava", "moondream"]
            try:
                import json as _jj, urllib.request as _uu
                with _uu.urlopen(f"{_vsr_backend.host}/api/tags", timeout=4) as _rr:
                    _inst = [m["name"] for m in _jj.loads(_rr.read()).get("models", [])]
                for _pref in _PREFERRED_VSR:
                    for _inst_m in _inst:
                        if _pref in _inst_m.lower():
                            vsr_vision_model = _inst_m
                            break
                    else:
                        continue
                    break
            except Exception:
                vsr_vision_model = ""
        vsr_transcript = _vsr_lip_reading_with_llm(vsr_results, _llm_client)

    # Align ASR segment text to VSR frames by timestamp (O(n log n) via bisect).
    _asr_segs = asr_data.get("asr_segment_details", [])
    if vsr_results and _asr_segs:
        _seg_starts = [s.get("start_s", 0.0) for s in _asr_segs]
        for _vr in vsr_results:
            _ts = getattr(_vr, "timestamp_s", 0.0)
            idx = bisect.bisect_right(_seg_starts, _ts) - 1
            if idx >= 0:
                _seg = _asr_segs[idx]
                if _seg.get("end_s", 0.0) >= _ts and _seg.get("text"):
                    _vr.text_hypothesis = _seg["text"]
                    _vr.lm_rescored_text = _seg["text"]

    metrics = {
        "video": video_path,
        "enabled_agents": sorted(run_agents),
        "workflow_backend": selected_backend,
        "frames_processed": len(cv_results),
        "frames_skipped": sampler.stats.skipped_frames,
        "skip_rate_pct": round(sampler.stats.skip_rate * 100, 1),
        "total_detections": sum(len(r.detections) for r in cv_results),
        "cv_detections": [
            _serialize_cv_result(result, frame_shapes.get(result.frame_index))
            for result in cv_results
        ],
        "vsr_results": [_serialize_vsr_result(result) for result in vsr_results],
        "vsr_mode": "heuristic" if vsr_results and getattr(vsr_agent, "_backbone", None) is None else ("full" if vsr_results else "stub"),
        "vsr_transcript": vsr_transcript,
        "vsr_vision_model": vsr_vision_model,
        "runtime_devices": {
            "cv": str(getattr(cv_agent, "device", "unknown")),
            "asr": str(getattr(asr_agent, "device", "unknown")),
            "vsr": str(getattr(vsr_agent, "device", "unknown")),
        },
        "asr_segments": asr_data["asr_segments"],
        "transcript_preview": asr_data["transcript_preview"],
        "asr_transcript": asr_data.get("asr_transcript", ""),
        "asr_segment_details": asr_data.get("asr_segment_details", []),
        "asr_speaker_turns": asr_data.get("asr_speaker_turns", []),
        "asr_skipped": asr_data["asr_skipped"],
        "llm_backend": eff_llm_backend,
        "llm_ollama_model": eff_ollama_model,
        "llm_ollama_host": eff_ollama_host,
        "llm_claude_model": eff_claude_model,
    }
    if "asr_error" in asr_data:
        metrics["asr_error"] = asr_data["asr_error"]

    # ASR summary — run primary model + a different secondary model for comparison.
    # Primary  → asr_summary       (user-chosen model)
    # Secondary→ asr_summary_gpt_oss (always a DIFFERENT model from primary)
    #   If primary == gpt-oss  → secondary = llama3.2:latest
    #   Otherwise              → secondary = gpt-oss:latest
    _PREF_SECONDARY = "gpt-oss:latest"
    _FALLBACK_SECONDARY = "llama3.2:latest"
    metrics["asr_summary"] = ""
    metrics["asr_summary_model"] = ""
    metrics["asr_summary_gpt_oss"] = ""
    metrics["asr_summary_secondary_model"] = ""

    # ── LLM calls: primary summary + secondary summary + synthesis run concurrently ──
    # Timeline (old): primary(T1) → secondary(T2) → synthesis(T3)  = T1+T2+T3
    # Timeline (new): primary(T1) → [secondary(T2) ∥ synthesis(T3)] = T1+max(T2,T3)
    _secondary_client: LLMClient | None = None
    if _text_client and metrics.get("asr_transcript"):
        _secondary_model = _FALLBACK_SECONDARY if eff_text_model == _PREF_SECONDARY else _PREF_SECONDARY
        try:
            _sc = LLMClient.ollama(model=_secondary_model, host=eff_ollama_host)
            if _sc.is_available():
                _secondary_client = _sc
            else:
                metrics["asr_summary_gpt_oss_error"] = f"{_secondary_model} not available in Ollama"
        except Exception as _e:
            log.warning("Secondary client init failed (%s): %s", _secondary_model, _e)
            metrics["asr_summary_gpt_oss_error"] = str(_e)

    crew_data: dict[str, Any] = {
        "engine": "deterministic",
        "enabled": False,
        "report": "LLM synthesis disabled by backend or configuration.",
    }

    if _text_client and metrics.get("asr_transcript"):
        _transcript = metrics["asr_transcript"]
        _turns = metrics.get("asr_speaker_turns", [])

        with ThreadPoolExecutor(max_workers=2) as _llm_pool:
            log.info(
                "Generating ASR summaries concurrently: primary=%s secondary=%s",
                eff_text_model,
                _secondary_model if _secondary_client else "none",
            )
            _primary_fut = _llm_pool.submit(_summarize_asr_with_llm, _transcript, _turns, _text_client)
            _secondary_fut = (
                _llm_pool.submit(_summarize_asr_with_llm, _transcript, _turns, _secondary_client)
                if _secondary_client else None
            )

            # Block on primary first — synthesis needs it
            metrics["asr_summary"] = _primary_fut.result()
            metrics["asr_summary_model"] = eff_text_model

            # Submit synthesis immediately once primary is done (runs while secondary finishes)
            if _llm_run:
                _synthesis_fut = _llm_pool.submit(_run_crewai_synthesis, metrics, _text_client)

            # Collect secondary
            if _secondary_fut is not None:
                try:
                    metrics["asr_summary_gpt_oss"] = _secondary_fut.result()
                    metrics["asr_summary_secondary_model"] = _secondary_model
                except Exception as _e:
                    log.warning("Secondary ASR summary failed (%s): %s", _secondary_model, _e)
                    metrics["asr_summary_gpt_oss_error"] = str(_e)

            # Collect synthesis
            if _llm_run:
                try:
                    crew_data = _synthesis_fut.result()
                except Exception as _e:
                    log.warning("LLM synthesis failed: %s", _e)
    elif _llm_run:
        # No transcript but synthesis still requested — run deterministic
        pass  # crew_data already set to deterministic default above

    # ── Intelligence report synthesis (fallback if not already run above) ─────
    if _llm_run and not metrics.get("asr_transcript"):
        crew_data = _run_crewai_synthesis(metrics, _text_client)
    metrics["crew_report"] = crew_data.get("report", "")
    metrics["crew_engine"] = crew_data.get("engine", "deterministic")
    metrics["crew_enabled"] = bool(crew_data.get("enabled", False))
    if "model" in crew_data:
        metrics["crew_model"] = crew_data["model"]
    if "error" in crew_data:
        metrics["crew_error"] = crew_data["error"]

    metrics["orchestration"] = {
        "engine": "harness-like",
        "stages": [
            {
                "name": s.name,
                "success": s.success,
                "attempts": s.attempts,
                "elapsed_s": round(s.elapsed_s, 3),
                "error": s.error,
            }
            for s in stages
        ],
    }

    return metrics
