"""
Streamlit — Multimodal Video Intelligence Platform
Full-featured UI: upload, configure LLM/SSIM, monitor pipeline,
and inspect per-agent results (CV / ASR / VSR / Fusion).
"""
from __future__ import annotations

import json
import os
import tempfile
import time

import requests
import streamlit as st

try:
    import torch
except Exception:
    torch = None

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None  # type: ignore
    np = None   # type: ignore

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_API = os.getenv("API_URL", "http://127.0.0.1:8000")

OLLAMA_MODELS: dict[str, tuple[str, str, str]] = {
    "llama3.2-vision": ("8 GB",   "✅ Vision",    "Best quality — video frames"),
    "llama3.2":        ("3 GB",   "❌ Text only", "Fast and capable text model"),
    "llava":           ("4 GB",   "✅ Vision",    "Balanced quality & speed"),
    "moondream":       ("1.6 GB", "✅ Vision",    "Low-resource machines"),
    "bakllava":        ("4 GB",   "✅ Vision",    "Instruction following"),
    "gpt-oss":         ("13 GB",  "✅ Vision",    "GPT-class — high quality"),
    "gpt-oss20b":      ("20 GB",  "❌ Text only", "GPT-class — highest quality text reasoning"),
    "mistral":         ("4 GB",   "❌ Text only", "Very fast — ASR/VSR text"),
    "phi3":            ("2 GB",   "❌ Text only", "Lightweight text reasoning"),
    "phi4":            ("9 GB",   "❌ Text only", "Strong reasoning & coding"),
    "qwen2.5":         ("4 GB",   "❌ Text only", "Multilingual ASR fusion"),
    "gemma2":          ("5 GB",   "❌ Text only", "Google Gemma 2"),
}

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Fetch installed Ollama models once per session
if "ollama_installed_models" not in st.session_state:
    try:
        _r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        if _r.ok:
            st.session_state.ollama_installed_models = [
                m["name"] for m in _r.json().get("models", [])
            ]
        else:
            st.session_state.ollama_installed_models = []
    except Exception:
        st.session_state.ollama_installed_models = []

_INSTALLED_MODELS: list[str] = st.session_state.ollama_installed_models


def _local_runtime_summary() -> dict[str, str]:
    if torch is None:
        return {"mode": "unknown", "label": "Unknown", "detail": "PyTorch unavailable"}
    if torch.cuda.is_available():
        return {
            "mode": "cuda",
            "label": "CUDA",
            "detail": torch.cuda.get_device_name(0),
        }
    return {"mode": "cpu", "label": "CPU", "detail": "CUDA unavailable"}


def _format_runtime_device(device_value: str | None) -> str:
    if not device_value:
        return "unknown"
    value = str(device_value).lower()
    if "cuda" in value:
        return "CUDA"
    if "cpu" in value:
        return "CPU"
    return str(device_value)


def _format_elapsed(seconds: float | int | None) -> str:
    if seconds is None:
        return "--"
    total_seconds = max(0, int(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _result_elapsed_seconds(result: dict) -> float:
    elapsed_s = result.get("elapsed_s")
    if elapsed_s is not None:
        return float(elapsed_s)
    return float(sum(s.get("elapsed_s", 0.0) for s in result.get("orchestration", {}).get("stages", [])))


@st.cache_data(show_spinner=False)
def _extract_thumbnails(
    video_bytes: bytes,
    timestamps: tuple[float, ...],
    thumb_w: int = 320,
    thumb_h: int = 180,
) -> dict[float, bytes]:
    """Extract JPEG thumbnails for each timestamp in one video pass."""
    if cv2 is None or not timestamps:
        return {}

    results: dict[float, bytes] = {}
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        tmp_path = f.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return {}
        fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        for ts in sorted(timestamps):
            frame_no = max(0, int(round(ts * fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret:
                continue
            thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            results[ts] = bytes(buf)
        cap.release()
    except Exception:
        pass
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return results


def _draw_obb_on_thumb(
    thumb_bytes: bytes,
    detections: list[dict],
    frame_w: int,
    frame_h: int,
    thumb_w: int = 320,
    thumb_h: int = 180,
) -> bytes:
    """Scale OBB detections to thumbnail size and draw them."""
    if cv2 is None or np is None or not thumb_bytes:
        return thumb_bytes
    buf = np.frombuffer(thumb_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return thumb_bytes

    sx = thumb_w / max(frame_w, 1)
    sy = thumb_h / max(frame_h, 1)
    palette = [
        (0, 200, 0), (0, 120, 255), (255, 60, 0),
        (200, 0, 200), (0, 200, 200), (255, 200, 0),
    ]
    for i, det in enumerate(detections):
        bbox = det.get("bbox_xywhr")
        if not bbox or len(bbox) < 5:
            continue
        cx, cy, w, h, angle = bbox
        cx_s, cy_s = cx * sx, cy * sy
        w_s,  h_s  = w  * sx, h  * sy
        color = palette[i % len(palette)]
        rect = ((float(cx_s), float(cy_s)), (float(w_s), float(h_s)), float(np.degrees(angle)))
        box_pts = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(img, [box_pts], 0, color, 1)
        label = f"{det.get('class', '?')} {det.get('conf', 0):.2f}"
        lx = max(int(cx_s - w_s / 2), 0)
        ly = max(int(cy_s - h_s / 2) - 3, 8)
        cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    _, out = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return bytes(out)

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Video Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    runtime_summary = _local_runtime_summary()
    runtime_icon = "🟢" if runtime_summary["mode"] == "cuda" else ("🟡" if runtime_summary["mode"] == "cpu" else "⚪")
    st.markdown("**Runtime**")
    st.caption(f"{runtime_icon} {runtime_summary['label']} — {runtime_summary['detail']}")

    api_url = st.text_input("API Server URL", value=DEFAULT_API)

    # Live health check
    try:
        r = requests.get(f"{api_url}/health", timeout=3)
        if r.ok:
            st.success(f"✅ API Online — v{r.json().get('version', '?')}")
        else:
            st.warning(f"⚠️ API returned HTTP {r.status_code}")
    except Exception:
        st.error("❌ API Offline — run `python scripts/run_pipeline.py --serve`")

    if "job_id" in st.session_state:
        st.divider()
        st.subheader("⏱️ Processing")
        current_job_id = st.session_state.job_id
        try:
            job_resp = requests.get(f"{api_url}/jobs/{current_job_id}", timeout=3)
            job_resp.raise_for_status()
            sidebar_job = job_resp.json()
            sidebar_status = sidebar_job.get("status", "unknown")
            result = sidebar_job.get("result", {})

            if sidebar_status in {"queued", "running"}:
                started_at = st.session_state.get("job_started_at")
                if started_at is None:
                    started_at = time.time()
                    st.session_state.job_started_at = started_at
                elapsed_label = _format_elapsed(time.time() - started_at)
            elif result:
                elapsed_label = _format_elapsed(_result_elapsed_seconds(result))
            else:
                elapsed_label = "--"

            st.caption(f"Job: `{current_job_id[:8]}` · {sidebar_status}")
            st.metric("Processing Timer", elapsed_label)
        except Exception:
            st.caption(f"Job: `{current_job_id[:8]}`")
            st.metric("Processing Timer", "--")

    st.divider()
    st.subheader("🤖 LLM Backend")
    llm_backend = st.selectbox(
        "Backend",
        ["auto", "ollama", "claude", "stub"],
        index=0,
        help="auto = Ollama → Claude → stub fallback chain",
    )
    ollama_model = "llama3.2:latest"
    text_model = ""
    if llm_backend in ("ollama", "auto"):
        _model_choices = _INSTALLED_MODELS if _INSTALLED_MODELS else list(OLLAMA_MODELS.keys())

        # Vision model — for VSR lip-reading (needs multimodal capability)
        _vision_default = next(
            (i for i, m in enumerate(_model_choices) if "llama3.2-vision" in m), 0
        )
        ollama_model = st.selectbox(
            "👁️ Vision Model (VSR lip-reading)",
            _model_choices,
            index=_vision_default,
            help="Used only for VSR mouth-crop lip-reading. Pick a vision-capable model.",
        )
        if ollama_model in OLLAMA_MODELS:
            size, vision, desc = OLLAMA_MODELS[ollama_model]
            st.caption(f"{vision} · {size} · {desc}")
        elif _INSTALLED_MODELS:
            st.caption("✅ Installed locally")

        # Text model — for ASR summary + intelligence report synthesis
        _text_default = next(
            (i for i, m in enumerate(_model_choices) if "gpt-oss" in m),
            next((i for i, m in enumerate(_model_choices) if "llama3.2" in m and "vision" not in m), 0),
        )
        text_model = st.selectbox(
            "📝 Text Model (ASR summary + report)",
            _model_choices,
            index=_text_default,
            help="Used for ASR transcript summary and intelligence report synthesis. Any model works.",
        )
        if text_model in OLLAMA_MODELS:
            size, vision, desc = OLLAMA_MODELS[text_model]
            st.caption(f"{vision} · {size} · {desc}")
        elif _INSTALLED_MODELS:
            st.caption("✅ Installed locally")

    # ── Ollama connection test ─────────────────────────────────────────────
    if st.button("🔌 Test Ollama Connection", width="stretch"):
        with st.spinner("Connecting to Ollama…"):
            # 1. Ping /api/tags to list available models
            try:
                tags_resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
                tags_resp.raise_for_status()
                available = [m["name"] for m in tags_resp.json().get("models", [])]
                st.success(f"✅ Ollama reachable at `{OLLAMA_HOST}`")
                st.markdown("**Installed models:**")
                for m in available:
                    st.markdown(f"- `{m}`")
                # refresh session cache
                st.session_state.ollama_installed_models = available
            except Exception as exc:
                st.error(f"❌ Cannot reach Ollama at `{OLLAMA_HOST}`: {exc}")
                st.caption("Start Ollama with `ollama serve` and ensure it is on the default port 11434.")
                available = []

            # 2. Quick chat round-trip with the selected model
            test_model = ollama_model if ollama_model in (available or [ollama_model]) else (available[0] if available else None)
            if test_model:
                try:
                    import json as _json, urllib.request as _ur
                    payload = _json.dumps({
                        "model": test_model,
                        "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
                        "stream": False,
                        "options": {"temperature": 0},
                    }).encode()
                    req = _ur.Request(
                        f"{OLLAMA_HOST}/api/chat",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with _ur.urlopen(req, timeout=30) as r:
                        chat_data = _json.loads(r.read())
                    reply = chat_data.get("message", {}).get("content", "").strip()
                    tok_in  = chat_data.get("prompt_eval_count", "?")
                    tok_out = chat_data.get("eval_count", "?")
                    st.success(f"✅ Chat round-trip OK — model: `{test_model}`")
                    st.markdown(f"> **Model reply:** {reply}")
                    st.caption(f"tokens in={tok_in} out={tok_out}")
                except Exception as exc:
                    st.warning(f"⚠️ Chat test failed for `{test_model}`: {exc}")

    st.divider()
    st.subheader("🎞️ Frame Sampling")
    ssim_threshold = st.slider(
        "SSIM Threshold",
        min_value=0.85, max_value=0.98, value=0.92, step=0.01,
        help="Higher = fewer frames processed (skips similar frames)",
    )
    st.caption("Lower → more frames → higher accuracy, slower processing")

    st.divider()
    with st.expander("📐 Pipeline Architecture"):
        st.markdown("""
```
Video → SSIM Adaptive Sampler
              │
      ┌───────┼───────┐
      ▼       ▼       ▼
   CV Agent  ASR   VSR Agent    ← parallel
      └───────┼───────┘
              ▼
        Redis Streams Bus
              ▼
        Fusion Consumer
              ▼
       Ollama / Claude LLM
              ▼
          JSON Report
```
| Agent | Technology |
|---|---|
| **CV** | YOLOv8x (Ultralytics) |
| **ASR** | faster-whisper + X-vectors |
| **VSR** | MediaPipe + Ollama vision |
| **Fusion** | Ollama (local) / Claude |
        """)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎬 Multimodal Video Intelligence")
st.markdown(
    "Three specialized AI agents — **Computer Vision**, **Speech Recognition**, and "
    "**Visual Speech Recognition** — run in parallel on every key frame. A fusion LLM "
    "then synthesises a cross-modal intelligence report."
)

# Agent overview cards
c1, c2, c3, c4 = st.columns(4)
c1.info("**🔍 CV Agent**\nYOLOv8x\nObject detection (COCO 80-class / OBB)")
c2.info("**🎙️ ASR Agent**\nfaster-whisper\nTranscription + speaker diarization")
c3.info("**👄 VSR Agent**\nMediaPipe mouth ROI\nLip crops → Ollama vision")
c4.info("**🧠 Fusion LLM**\nOllama / Claude\nCross-modal intelligence report")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_analyze, tab_results, tab_jobs = st.tabs(["📤 Analyze", "📊 Results", "📋 All Jobs"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: Analyze
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analyze:
    st.subheader("Upload a Video")
    uploaded_file = st.file_uploader(
        "Drag and drop or click to browse (MP4, MOV, MKV, AVI, WebM)",
        type=["mp4", "mov", "mkv", "avi", "webm"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        # Persist video bytes so the Results tab can extract thumbnails
        uploaded_file.seek(0)
        st.session_state.uploaded_video_bytes = uploaded_file.read()
        uploaded_file.seek(0)

        col_preview, col_meta = st.columns([2, 1])
        with col_preview:
            st.video(uploaded_file)
        with col_meta:
            st.markdown("**File**")
            st.write(f"- **Name:** {uploaded_file.name}")
            st.write(f"- **Type:** `{uploaded_file.type}`")
            st.write(f"- **Size:** {uploaded_file.size / (1024 * 1024):.2f} MB")
            st.markdown("**Pipeline**")
            st.write(f"- **LLM backend:** `{llm_backend}`")
            if llm_backend in ("ollama", "auto"):
                st.write(f"- **Ollama model:** `{ollama_model}`")
            st.write(f"- **SSIM threshold:** `{ssim_threshold}`")

            asr_model = st.selectbox(
                "ASR Model",
                ["base", "tiny", "small", "medium", "large-v3", "finetune/models/whisper-lora-ct2"],
                index=0,
                help="Select between base whisper models or the custom finetuned model",
            )

            st.markdown("**Agents**")
            run_cv  = st.checkbox("🔍 Computer Vision (CV)",  value=True, key="agent_cv")
            run_asr = st.checkbox("🎤 Speech Recognition (ASR)", value=True, key="agent_asr")
            run_vsr = st.checkbox("👄 Visual Speech (VSR)",     value=True, key="agent_vsr")
            agents_selected = ",".join(
                a for a, enabled in [("cv", run_cv), ("asr", run_asr), ("vsr", run_vsr)] if enabled
            ) or "cv"  # at least one agent required

        if st.button("🚀 Run Analysis", width="stretch", type="primary"):
            with st.spinner("Uploading video and queuing the pipeline…"):
                try:
                    uploaded_file.seek(0)
                    resp = requests.post(
                        f"{api_url}/analyze",
                        files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)},
                        data={"llm_backend": llm_backend, "ollama_model": ollama_model, "text_model": text_model, "agents": agents_selected, "asr_model": asr_model},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    job_id = resp.json()["job_id"]
                    st.session_state.job_id = job_id
                    st.session_state.job_started_at = time.time()
                    st.success(f"✅ Job queued! **ID:** `{job_id}`")
                    st.info("Switch to the **Results** tab to monitor progress.")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the API. Make sure the server is running on the URL above.")
                except requests.exceptions.RequestException as exc:
                    st.error(f"Failed to start analysis: {exc}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: Results
# ═══════════════════════════════════════════════════════════════════════════════
with tab_results:
    if "job_id" not in st.session_state:
        st.info("Upload a video on the **Analyze** tab to see results here.")
        st.stop()

    job_id = st.session_state.job_id
    col_jid, col_btn = st.columns([4, 1])
    col_jid.markdown(f"**Job ID:** `{job_id}`")
    col_btn.button("🔄 Refresh")  # triggers a rerun when clicked

    try:
        resp = requests.get(f"{api_url}/jobs/{job_id}", timeout=30)
        resp.raise_for_status()
        job_data = resp.json()
        status = job_data.get("status", "unknown")
    except requests.exceptions.Timeout:
        st.warning(
            "⏳ API server is busy — models are loading (PyTorch / CUDA init can take 20–30 s). "
            "Auto-retrying in 5 s…"
        )
        time.sleep(5)
        st.rerun()
    except Exception as exc:
        st.error(f"Could not fetch job: {exc}")
        st.stop()

    # Auto-rerun while in-flight
    if status == "queued":
        st.warning("⏳ Job is queued — waiting for a worker…")
        time.sleep(2)
        st.rerun()
    elif status == "running":
        st.warning("⚙️ Pipeline is running… (refreshing every 2 s)")
        time.sleep(2)
        st.rerun()
    elif status == "error":
        st.error(f"❌ Job failed: {job_data.get('error')}")
        st.stop()
    else:
        st.session_state.pop("job_started_at", None)

    # ── Done ─────────────────────────────────────────────────────────────────
    st.success("✅ Analysis complete!")
    result = job_data.get("result", {})
    st.session_state.runtime_devices = result.get("runtime_devices", {})

    # Summary metrics row
    st.subheader("Summary")
    elapsed_s = result.get("elapsed_s")
    if elapsed_s is None:
        elapsed_s = sum(
            s.get("elapsed_s", 0.0)
            for s in result.get("orchestration", {}).get("stages", [])
        )
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Frames Processed", result.get("frames_processed", "–"))
    m2.metric("Frames Skipped",   result.get("frames_skipped", "–"))
    m3.metric("Skip Rate",        f"{result.get('skip_rate_pct', 0):.1f}%")
    m4.metric("Elapsed",          f"{elapsed_s:.2f} s")
    m5.metric("LLM Backend",      result.get("llm_backend", result.get("workflow_backend", "–")))
    m6.metric("LLM Model",        result.get("crew_model", result.get("llm_ollama_model", "–")))

    st.divider()

    fusion_reports: list[dict] = result.get("fusion_reports", [])
    # Build result tabs dynamically based on which agents were enabled for this job
    _job_agents: set[str] = set(result.get("enabled_agents", ["cv", "asr", "vsr"]))
    _tab_defs: list[tuple[str, str]] = [
        ("report", "🧠 Intelligence Report"),
        ("asr",    "🎤 ASR Transcription"),
        ("cv",     "🔍 CV Results"),
        ("vsr",    "👄 VSR Results"),
        ("orch",   "⚙️ Orchestration"),
        ("json",   "🧧 Raw JSON"),
    ]
    # Always show Report, Orch, JSON; show agent tabs only if that agent ran
    _visible = [
        label for key, label in _tab_defs
        if key in {"report", "orch", "json"} or key in _job_agents
    ]
    _tab_keys = [
        key for key, label in _tab_defs
        if key in {"report", "orch", "json"} or key in _job_agents
    ]
    _tabs = st.tabs(_visible)
    _tab_map = dict(zip(_tab_keys, _tabs))
    r_report = _tab_map["report"]
    r_asr    = _tab_map.get("asr")
    r_cv     = _tab_map.get("cv")
    r_vsr    = _tab_map.get("vsr")
    r_orch   = _tab_map["orch"]
    r_json   = _tab_map["json"]

    # ── Intelligence Report ──────────────────────────────────────────────
    with r_report:
        crew_report = result.get("crew_report", "")
        crew_engine = result.get("crew_engine", "deterministic")
        crew_model  = result.get("crew_model", result.get("llm_ollama_model", ""))

        if crew_report and crew_engine != "deterministic":
            st.caption(
                f"Generated by **{crew_engine}**"
                + (f" · model `{crew_model}`" if crew_model else "")
            )
            st.markdown(crew_report)
        elif crew_report:
            st.info(crew_report)
        else:
            st.warning(
                "No intelligence report was generated. "
                "Make sure **Ollama is running** (`ollama serve`) and the selected model is installed, "
                "then re-run the analysis."
            )
            st.caption(
                "You can verify Ollama from the sidebar → **Test Ollama Connection**. "
                "The LLM backend used for this job was: "
                f"`{result.get('llm_backend', result.get('workflow_backend', '?'))}`"
            )

    # ── ASR ──────────────────────────────────────────────────────────────
    if r_asr is not None:
     with r_asr:
        asr_transcript = result.get("asr_transcript", "")
        asr_summary = result.get("asr_summary", "")
        asr_summary_gpt_oss = result.get("asr_summary_gpt_oss", "")
        asr_summary_model = result.get("asr_summary_model", result.get("crew_model", result.get("llm_ollama_model", "")))
        speaker_turns = result.get("asr_speaker_turns", [])
        segment_details = result.get("asr_segment_details", [])

        # Summary banners — side by side when both are present
        if asr_summary or asr_summary_gpt_oss:
            st.subheader("📝 Transcription Summary")
            has_both = bool(asr_summary and asr_summary_gpt_oss)
            if has_both:
                col_llm1, col_llm2 = st.columns(2)
                with col_llm1:
                    primary_label = asr_summary_model or "Primary model"
                    st.caption(f"**{primary_label}**")
                    st.info(asr_summary)
                with col_llm2:
                    st.caption("**gpt-oss**")
                    st.info(asr_summary_gpt_oss)
            elif asr_summary:
                if asr_summary_model:
                    st.caption(f"Generated by Ollama · `{asr_summary_model}`")
                st.info(asr_summary)
            else:
                st.caption("**gpt-oss**")
                st.info(asr_summary_gpt_oss)
            st.divider()
        elif not asr_transcript:
            st.warning("No speech detected. Ensure the video has audio and Ollama is running for summaries.")

        if asr_transcript:
            st.markdown("**Full transcript**")
            st.text_area("", asr_transcript, height=200, label_visibility="collapsed")
        elif result.get("transcript_preview"):
            st.markdown("**Transcript preview**")
            st.write(result.get("transcript_preview"))

        if speaker_turns:
            st.markdown("**Speaker turns**")
            for i, turn in enumerate(speaker_turns, start=1):
                with st.expander(
                    f"Turn {i} — {turn.get('speaker_name', 'Speaker')} "
                    f"[{turn.get('start_s', 0):.2f}s - {turn.get('end_s', 0):.2f}s]"
                ):
                    st.write(turn.get("text", ""))

        if segment_details:
            with st.expander("Show detailed ASR segments"):
                st.json(segment_details)

        if result.get("asr_error"):
            st.warning(f"ASR note: {result.get('asr_error')}")

    # ── CV / Fusion ──────────────────────────────────────────────────────
    if r_cv is not None:
     with r_cv:
        runtime_devices = result.get("runtime_devices", {})
        cv_runtime = _format_runtime_device(runtime_devices.get("cv"))
        asr_runtime = _format_runtime_device(runtime_devices.get("asr"))
        vsr_runtime = _format_runtime_device(runtime_devices.get("vsr"))

        st.caption(f"Running on CV: {cv_runtime} | ASR: {asr_runtime} | VSR: {vsr_runtime}")

        cv_detections = result.get("cv_detections", [])
        if cv_detections:
            # ── Face analysis aggregate summary ───────────────────────────
            all_faces = [f for fd in cv_detections for f in fd.get("faces", [])]
            if all_faces:
                st.subheader(f"👤 Faces detected: {len(all_faces)} across {len(cv_detections)} frames")
                gender_counts: dict[str, int] = {}
                ages: list[int] = []
                emotion_counts: dict[str, int] = {}
                for face in all_faces:
                    g = face.get("gender", "Unknown")
                    gender_counts[g] = gender_counts.get(g, 0) + 1
                    a = face.get("age")
                    if a:
                        ages.append(int(a))
                    e = face.get("dominant_emotion", "")
                    if e:
                        emotion_counts[e] = emotion_counts.get(e, 0) + 1

                face_col1, face_col2, face_col3, face_col4 = st.columns(4)
                face_col1.metric("Total Faces", len(all_faces))
                if ages:
                    face_col2.metric("Age Range", f"{min(ages)}–{max(ages)} yrs")
                gender_str = "  ·  ".join(f"{k}: {v}" for k, v in sorted(gender_counts.items()))
                face_col3.metric("Genders", gender_str or "–")
                if emotion_counts:
                    top_emotion = max(emotion_counts, key=emotion_counts.__getitem__)
                    face_col4.metric("Top Emotion", top_emotion)
                st.divider()

            st.markdown("**YOLOv8m — YOLO detections  |  Florence-2 — scene caption + open-vocab**")
            frames_with_detections = [
                frame_data for frame_data in cv_detections
                if frame_data.get("detections")
                or frame_data.get("scene_caption")
                or frame_data.get("open_vocab_detections")
            ]

            if not frames_with_detections:
                st.caption("No objects detected in sampled frames.")

            # ── Bulk thumbnail extraction ─────────────────────────────────
            video_bytes: bytes | None = st.session_state.get("uploaded_video_bytes")
            thumbs: dict[float, bytes] = {}
            show_obb = False
            if video_bytes and cv2 is not None and frames_with_detections:
                all_ts = tuple(
                    float(fd.get("timestamp_s", 0.0)) for fd in frames_with_detections
                )
                with st.spinner("Extracting frame thumbnails…"):
                    thumbs = _extract_thumbnails(video_bytes, all_ts)
                show_obb = bool(thumbs)

            if thumbs:
                show_obb = st.toggle("Show OBB overlays on thumbnails", value=True)

            for frame_data in frames_with_detections:
                fi    = frame_data.get("frame_index", "?")
                ts    = float(frame_data.get("timestamp_s", 0.0))
                dets  = frame_data.get("detections", [])
                ov    = frame_data.get("open_vocab_detections", [])
                cap   = frame_data.get("scene_caption", "").strip()
                fw    = frame_data.get("frame_w", 0)
                fh    = frame_data.get("frame_h", 0)

                with st.expander(f"Frame {fi} — {ts:.1f} s — {len(dets)} YOLO  |  {len(ov)} Florence-2"):
                    # ── Florence-2 scene caption (full-width banner) ──────────
                    if cap:
                        st.info(f"🖼️ **Florence-2 scene:** {cap}")
                    else:
                        st.caption("_Florence-2 scene caption: not available (model not loaded or still downloading)_")

                    thumb_bytes = thumbs.get(ts)
                    if thumb_bytes:
                        if show_obb and dets and fw and fh:
                            thumb_bytes = _draw_obb_on_thumb(
                                thumb_bytes, dets, fw, fh
                            )
                        col_thumb, col_dets = st.columns([2, 3])
                        with col_thumb:
                            st.image(thumb_bytes, caption=f"t = {ts:.2f} s", width="stretch")
                        with col_dets:
                            if dets:
                                st.markdown("**YOLOv8m detections:**")
                            for idx, det in enumerate(dets, start=1):
                                class_name = det.get("class", det.get("class_name", "object"))
                                confidence = det.get("conf", det.get("confidence"))
                                bbox = det.get("bbox_xywhr")
                                geo  = det.get("geo")
                                st.markdown(
                                    f"**{idx}.** {class_name}"
                                    + (f" · conf **{confidence:.3f}**" if isinstance(confidence, (int, float)) else "")
                                )
                                if bbox:
                                    st.caption(f"bbox_xywhr: {[round(v,1) for v in bbox]}")
                                if geo:
                                    st.caption(f"geo: {geo}")
                    else:
                        # No video bytes available — text-only layout
                        if dets:
                            st.markdown("**YOLOv8m detections:**")
                        for idx, det in enumerate(dets, start=1):
                            class_name = det.get("class", det.get("class_name", "object"))
                            confidence = det.get("conf", det.get("confidence"))
                            bbox = det.get("bbox_xywhr")
                            geo  = det.get("geo")
                            st.markdown(
                                f"**Detection {idx}:** {class_name}"
                                + (f" · conf {confidence:.3f}" if isinstance(confidence, (int, float)) else "")
                            )
                            if bbox:
                                st.caption(f"bbox_xywhr: {bbox}")
                            if geo:
                                st.caption(f"geo: {geo}")

                    # ── Florence-2 open-vocab detections (full-width) ─────────
                    if ov:
                        st.markdown("**Florence-2 open-vocab detections:**")
                        ov_cols = st.columns(3)
                        for i, o in enumerate(ov):
                            lbl  = o.get("label", "")
                            bbox = o.get("bbox_xyxy", [])
                            with ov_cols[i % 3]:
                                st.caption(f"• {lbl}" + (f"\n`{[round(v,0) for v in bbox]}`" if bbox else ""))

        else:
            st.metric("Total detections", result.get("total_detections", 0))
            st.caption("Per-frame fusion reports are unavailable for this orchestration mode.")

    # ── VSR Results ──────────────────────────────────────────────────────
    if r_vsr is not None:
     with r_vsr:
        vsr_results: list[dict] = result.get("vsr_results", [])
        vsr_mode = result.get("vsr_mode", "")
        vsr_transcript = result.get("vsr_transcript", "")

        # Lip-reading transcript banner
        if vsr_transcript:
            st.subheader("👄 Lip-Reading Transcript")
            vsr_vision_model = result.get("vsr_vision_model", "")
            model_label = vsr_vision_model or result.get("crew_model", result.get("llm_ollama_model", ""))
            _audio_note = "audio absent — " if result.get("asr_skipped") or not result.get("asr_transcript", "").strip() else ""
            st.caption(
                f"Lip-reading via Ollama vision ({_audio_note}gpt-oss preferred)"
                + (f" · `{model_label}`" if model_label else "")
            )
            st.success(vsr_transcript)
            st.divider()

        # Mode status banner
        if not vsr_mode or vsr_mode == "stub":
            st.info(
                "**VSR — mouth activity mode** — no frames were processed.\n\n"
                "Ensure the video contains a visible face and at least one agent is enabled."
            )
        elif vsr_mode == "heuristic":
            st.info(
                "**VSR — mouth activity + Ollama lip-reading**\n\n"
                "MediaPipe extracts 112×112 mouth crops per frame. "
                "The transcript above is produced by an Ollama vision model reading those crops. "
                "Per-frame text is aligned from ASR timestamps when audio is available."
            )
        else:
            st.success(f"VSR in **{vsr_mode}** mode — AV-HuBERT backbone loaded.")

        if vsr_results:
            mouth_open_count = sum(1 for r in vsr_results if r.get("mouth_open"))
            avg_lar = (
                sum(r.get("lip_aspect_ratio", 0.0) for r in vsr_results) / len(vsr_results)
                if vsr_results else 0.0
            )
            vcol1, vcol2, vcol3 = st.columns(3)
            vcol1.metric("Frames Analysed", len(vsr_results))
            vcol2.metric("Mouth Open Frames", mouth_open_count)
            vcol3.metric("Avg Lip Aspect Ratio", f"{avg_lar:.3f}")

            st.markdown("**Per-frame mouth activity**")
            st.caption(
                "👄 = mouth open (speech likely) · 😶 = closed · "
                "LAR = lip aspect ratio (height/width) · "
                "ASR text aligned from faster-whisper timestamps when audio is available"
            )
            for vr in vsr_results:
                fi = vr.get("frame_index", "?")
                ts = vr.get("timestamp_s", 0.0)
                mo = vr.get("mouth_open", False)
                lar = vr.get("lip_aspect_ratio")
                text_hyp = vr.get("text_hypothesis", "")
                lm_text = vr.get("lm_rescored_text", "")
                visemes = vr.get("visemes", [])
                # text_hypothesis is either ASR-aligned text (real words) or a
                # heuristic phoneme label (single letter like "p"). Distinguish them:
                _is_phoneme_only = len(text_hyp.split()) <= 1 and len(text_hyp) <= 3
                mo_icon = "👄" if mo else "😶"
                label = f"{mo_icon} Frame {fi} — {ts:.2f} s"
                if lar is not None:
                    label += f" · LAR {lar:.3f}"
                if text_hyp and not _is_phoneme_only:
                    label += f" · \"{text_hyp[:50]}{'…' if len(text_hyp) > 50 else ''}\""
                elif mo:
                    label += " · speech detected"
                with st.expander(label, expanded=False):
                    cols = st.columns(3)
                    cols[0].metric("Mouth Open", "Yes" if mo else "No")
                    if lar is not None:
                        cols[1].metric("LAR", f"{lar:.3f}")
                    inf_ms = vr.get("inference_ms")
                    if inf_ms is not None:
                        cols[2].metric("Infer ms", f"{inf_ms:.1f}")
                    # Show real text if ASR-aligned, otherwise explain where transcript is
                    if text_hyp and not _is_phoneme_only:
                        st.success(f"**ASR text at this timestamp:** {text_hyp}")
                    elif mo and vsr_transcript:
                        st.info(
                            f"**Lip-reading transcript (full clip, see banner above):** "
                            f"{vsr_transcript[:200]}{'…' if len(vsr_transcript) > 200 else ''}"
                        )
                    elif mo:
                        st.warning(
                            "Mouth open — no per-frame transcript. "
                            "The full lip-reading transcript appears in the **green banner above** (Ollama vision)."
                        )
                    if visemes:
                        st.caption(
                            "Heuristic mouth shape: **" + visemes[0].get("viseme_label", "?") + "**"
                            + (" → " + " → ".join(v.get("viseme_label", "?") for v in visemes[1:12]) if len(visemes) > 1 else "")
                        )
        else:
            st.caption("No VSR frame data available for this job.")

    # ── Orchestration ─────────────────────────────────────────────────────
    with r_orch:
        # LLM Synthesis — refer to dedicated tab
        crew_engine = result.get("crew_engine", "deterministic")
        crew_model  = result.get("crew_model", result.get("llm_ollama_model", ""))
        st.caption(
            f"LLM synthesis: **{crew_engine}**"
            + (f" · `{crew_model}`" if crew_model else "")
            + " — full report in the **🧠 Intelligence Report** tab."
        )

        st.divider()
        st.markdown("**Pipeline stages & agents**")

        # Map stage names to the agents invoked within them
        _STAGE_AGENTS = {
            "load_agents": ["CVAgent (YOLOv8-OBB)", "ASRAgent (faster-whisper)", "VSRAgent (AV-HuBERT)"],
            "run_cv_vsr":  ["CVAgent → object detection", "VSRAgent → lip / viseme analysis"],
            "run_asr":     ["ASRAgent → Whisper transcription", "ASRAgent → speaker diarization"],
        }
        _STATUS_ICON = {True: "✅", False: "❌"}

        stages = result.get("orchestration", {}).get("stages", [])
        if stages:
            for stage in stages:
                name     = stage.get("name", "?")
                success  = stage.get("success", False)
                attempts = stage.get("attempts", 1)
                elapsed  = stage.get("elapsed_s", 0.0)
                error    = stage.get("error")
                icon     = _STATUS_ICON[success]
                agents   = _STAGE_AGENTS.get(name, [])

                with st.expander(
                    f"{icon} **{name}** — {elapsed:.2f} s"
                    + (f"  _(attempt {attempts})_" if attempts > 1 else ""),
                    expanded=not success,
                ):
                    if agents:
                        st.markdown("**Agents called:**")
                        for ag in agents:
                            st.markdown(f"- {ag}")
                    if error:
                        st.error(f"Error: {error}")
                    cols = st.columns(3)
                    cols[0].metric("Status",   "success" if success else "failed")
                    cols[1].metric("Attempts", attempts)
                    cols[2].metric("Elapsed",  f"{elapsed:.3f} s")

            # LLM synthesis as its own pseudo-stage
            crew_engine_s = result.get("crew_engine", "deterministic")
            crew_model_s  = result.get("crew_model", "")
            llm_icon = "✅" if result.get("crew_enabled") else "⚠️"
            with st.expander(
                f"{llm_icon} **llm_synthesis** — via {crew_engine_s}"
                + (f" (`{crew_model_s}`)" if crew_model_s else ""),
                expanded=False,
            ):
                st.markdown("**Agents called:**")
                st.markdown(f"- LLMClient → Ollama (`{crew_model_s or 'n/a'}`)")
                cols2 = st.columns(2)
                cols2[0].metric("Engine", crew_engine_s)
                cols2[1].metric("Model",  crew_model_s or "–")
        else:
            st.caption("No stage data available.")

    # ── Raw JSON ──────────────────────────────────────────────────────────
    with r_json:
        st.markdown("**Full report JSON (inline)**")
        st.json(result)

    st.divider()
    st.download_button(
        label="⬇️ Download Full JSON Report",
        data=json.dumps(result, indent=2),
        file_name=f"mvi_report_{job_id[:8]}.json",
        mime="application/json",
        width="stretch",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: All Jobs
# ═══════════════════════════════════════════════════════════════════════════════
with tab_jobs:
    st.subheader("Job History")
    st.button("🔄 Refresh List")  # triggers rerun
    try:
        resp = requests.get(f"{api_url}/jobs", timeout=5)
        resp.raise_for_status()
        all_jobs: list[dict] = resp.json()
    except Exception as exc:
        st.error(f"Could not fetch job list: {exc}")
        st.stop()

    if not all_jobs:
        st.info("No jobs yet. Upload a video on the **Analyze** tab.")
    else:
        for job in reversed(all_jobs):
            jid = job["job_id"]
            jst = job["status"]
            _JOB_STATUS_ICON = {"done": "✅", "running": "⚙️", "queued": "⏳", "error": "❌"}
            icon = _JOB_STATUS_ICON.get(jst, "❓")
            col_id, col_status, col_load = st.columns([5, 2, 1])
            col_id.code(jid)
            col_status.markdown(f"{icon} `{jst}`")
            if col_load.button("Load", key=f"load_{jid}"):
                st.session_state.job_id = jid
                st.session_state.pop("job_started_at", None)
                st.success(f"Loaded job `{jid}`. Switch to the **Results** tab.")
