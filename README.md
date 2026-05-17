# Multimodal Video Intelligence Platform

[![CI](https://github.com/YOUR_ORG/multimodal-video-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_ORG/multimodal-video-intelligence/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch 2.11+](https://img.shields.io/badge/PyTorch-2.11+-ee4c2c.svg)](https://pytorch.org/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20%7C%20Claude-8B5CF6.svg)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A pipeline that extracts structured intelligence from video by running three AI agents in parallel - **computer vision**, **speech recognition**, and **visual speech reading** - then fusing their outputs into a single operator report via a language model. The LLM runs locally through **Ollama** or falls back to **Claude** when a cloud endpoint is preferred.

Key design choices: SSIM frame sampling only processes frames where the scene has changed, reducing a 30-min video from ~45 000 frames to ~1 800. Speaker diarisation labels who is talking across pauses. A LoRA-fine-tuned Whisper adapter is included for noisy audio.

---

## What it does

| Agent | Technology | Output |
|---|---|---|
| **CV** | YOLOv8m, COCO 80-class, conf = 0.30 | Per-frame bounding boxes with class labels and confidence scores; supports OBB mode for aerial footage |
| **ASR** | faster-whisper (CUDA FP16) | Full transcript with 100 ms word-level timestamps, automatic language detection, speaker-turn labels, and a 3-5 sentence LLM summary |
| **VSR** | MediaPipe 468-point face mesh | 112x112 mouth ROI per frame -> viseme sequence; falls back to Ollama vision lip-reading when audio is absent or failed |
| **Fusion** | CrewAI + Ollama / Claude | Structured intelligence report covering scene, speakers, objects, cross-modal correlations, and confidence |

---

## Architecture

```
Video (local / S3 / RTSP)
         |
         v
  SSIM Adaptive Sampler --> skips redundant frames
         |
    +----+--------------+
    |    |              |
 CV Agent  ASR Agent  VSR Agent     --> run in parallel
    |        |          |
    |        |          +- Lip-reading via Ollama vision        ||
    |        |          |  (only when audio is absent)              |
    |        +- LLM summary of transcription
    +--------------------+
             |
             v
      Fusion Engine (CrewAI)
             |
    +----------------+   +---------------+
    |  Ollama (local)| or| Claude (cloud)|
    +----------------+   +---------------+
             |
             v
       Intelligence Report  -->  JSON + Streamlit UI
```

---

## How It Works - Processing Pipeline

A full walkthrough of the pipeline is documented in [explanation.md](explanation.md). Summary:

| Step | What happens |
|---|---|
| **1 - Upload** | Client POSTs a video to `/analyze`. Extension validated; job UUID written to in-memory store; file saved to temp disk. |
| **2 - Agent cache** | `_load_agents()` checks a frozen-key cache. On a hit, model instances are reused; on a miss, YOLOv8m + Whisper + MediaPipe load (~5-10 s one-time cost). |
| **3 - Adaptive sampling** | `AdaptiveSampler` computes SSIM between consecutive frames. Frames with SSIM > 0.75 are skipped (scene unchanged), reducing a 30-min 25 FPS video from ~45 000 frames to ~200-500. |
| **4 - CV + VSR (parallel)** | Per sampled frame, two thread tasks run concurrently: **YOLOv8m** (COCO 80-class, conf = 0.30) and **MediaPipe** (468-point face mesh -> 112x112 mouth ROI -> viseme). |
| **5 - ASR (parallel with step 4)** | ffmpeg extracts 16 kHz mono WAV -> faster-Whisper transcribes with word-level timestamps. Speaker diarisation uses gap-based heuristics (silence > 350 ms = new turn) with optional SpeechBrain X-vector clustering. |
| **6 - VSR timestamp alignment** | Bisect-search maps each VSR frame timestamp to the overlapping ASR segment so the synthesis prompt can correlate lip activity with spoken words. |
| **7 - Lip-reading fallback** | Only when audio is absent/failed AND a mouth ROI was detected: sends 3 temporally-spread 256x256 mouth crops to `llama3.2-vision` via Ollama for inferred spoken text. |
| **8 - ASR LLM summary (2 models parallel)** | Primary model (`gpt-oss`) and secondary (`llama3.2`) summarise the transcript concurrently in a 2-worker pool. Both results stored for cross-model validation. |
| **9 - Intelligence synthesis** | `_run_crewai_synthesis()` builds a structured prompt from all agent outputs and asks the text LLM to produce a labelled report (`SCENE / SPEAKERS / OBJECTS / CROSS-MODAL / CONFIDENCE / ACTIONS`). Falls back to deterministic string if LLM is unavailable. |
| **10 - Result assembly** | All data merged into `metrics`, stored in the job dict, status -> `done`. Streamlit polls `/jobs/{id}` every 500 ms and renders the report. |

### Threading model

```
Job Executor (max_workers=1)         -- serialises jobs; prevents concurrent CUDA access
+-- run_video_workflow()
    +-- Frame pool (max_workers=4)   -- CV + VSR per frame, concurrent
    +-- Outer pool (max_workers=4)   -- ASR runs parallel to entire frame loop
    +-- LLM pool (max_workers=2)     -- primary + secondary ASR summary concurrent
```

### Evaluation metrics

The `test_asr_summary_evals.py` suite measures summary quality with **ROUGE** (unigram/bigram/LCS overlap) and **BLEU** (modified n-gram precision with brevity penalty). ASR accuracy is measured with **WER/CER** (Levenshtein edit distance). See [explanation.md](explanation.md) for the full metric formulae.

---

## Key features

- **Auto vision-model selection** - VSR lip-reading queries `/api/tags` and swaps in the first installed Ollama vision model if the configured model is text-only
- **Speaker X-vector clustering** - greedy cosine centroid clustering gives consistent speaker labels across pauses, backed by optional Qdrant identity store
- **ASR LLM summary** - Ollama/Claude produces a 3-5 sentence summary covering topics, speakers, key facts, and sentiment
- **SSIM adaptive sampler** - skips visually redundant frames, tunable via `ssim_threshold`

- **Auto-download weights** - `yolov8m` downloads from Ultralytics automatically; no manual weight fetch needed for standard video
- **ROUGE / BLEU eval suite** - offline evaluation of ASR summaries using `rouge-score` + `nltk` BLEU

---

## Prerequisites

- Python 3.11+
- `ffmpeg` (for audio extraction)
- **LLM choice - pick one:**
  - [Ollama](https://ollama.com) (local)
  - Anthropic API key (set `ANTHROPIC_API_KEY`)

---

## Installation

```bash
# 1. Clone
git clone https://github.com/YOUR_ORG/multimodal-video-intelligence.git
cd multimodal-video-intelligence

# 2. Install ffmpeg
#    macOS:   brew install ffmpeg
#    Ubuntu:  sudo apt install ffmpeg
#    Windows: https://ffmpeg.org/download.html

# 3. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 4. Install PyTorch (CPU - works on any machine)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# 5. Install project dependencies
pip install -r requirements.txt
```

### GPU installation (optional, for faster inference)

```bash
# CUDA 12.8
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Then set device: "cuda" in configs/default.yaml
```

---

## LLM Setup - choose one

### Option A - Ollama (recommended, 100% local & free)

```bash
# 1. Install Ollama
#    macOS / Linux: curl -fsSL https://ollama.com/install.sh | sh
#    Windows:       https://ollama.com/download

# 2. Pull the vision model (required for VSR lip-reading)
ollama pull gpt-oss            # used for VSR + secondary ASR summary
ollama pull llama3.2-vision    # fallback if gpt-oss unavailable

# 3. Pull the primary text model
ollama pull llama3.2           # default primary - ASR summary + fusion

# 4. Verify
ollama list
```

> **Note:** If you select a text-only model (e.g. `mistral`, `phi3`) in the UI,
> the pipeline automatically detects this and swaps in the first installed
> vision-capable model for VSR lip-reading. No manual intervention needed.

### Option B - Claude (cloud)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Quickstart

### 1. Stub mode - no GPU, no API key, no weights

Tests the full pipeline architecture with synthetic agent outputs.

```bash
python scripts/run_pipeline.py --video path/to/video.mp4
```

### 2. Ollama fusion (local LLM, free)

```bash
# Make sure Ollama is running: ollama serve
python scripts/run_pipeline.py \
  --video path/to/video.mp4 \
  --llm ollama \
  --ollama-model llama3.2-vision
```

### 3. Claude fusion (cloud)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/run_pipeline.py \
  --video path/to/video.mp4 \
  --llm claude
```

### 4. Auto mode (Ollama -> Claude -> stub)

```bash
python scripts/run_pipeline.py \
  --video path/to/video.mp4 \
  --llm auto
```

### 5. Save results to JSON

```bash
python scripts/run_pipeline.py \
  --video path/to/video.mp4 \
  --llm ollama \
  --output results/my_video.json
```

### 6. SSIM threshold benchmark

Sweep thresholds and find the optimal compute trade-off:

```bash
python scripts/run_pipeline.py --benchmark-ssim path/to/video.mp4
```

### 7. Streamlit UI

```bash
# Terminal 1 - API server (port 8000)
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Streamlit UI (port 8501)
streamlit run scripts/streamlit_app.py
```

Open http://localhost:8501. From the sidebar you can:
- Upload any video file
- Choose LLM backend (`ollama` / `claude` / `auto` / `stub`)
- Select an Ollama model (vision models auto-selected for VSR)
- Monitor pipeline progress in real time

Results are shown across six tabs:

| Tab | Contents |
|---|---|
| Intelligence Report | CrewAI-synthesized cross-modal report |
| ASR Transcription | Full transcript + speaker turns + LLM summary banner |
| CV Results | YOLOv8m per-frame detections with OBB overlays, Florence-2 scene caption, open-vocabulary detections |
| VSR Results | Lip-reading transcript (audio-absent only) + per-frame visemes |
| Orchestration | Pipeline metadata |
| Raw JSON | Full result payload |

### 8. REST API server

```bash
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

# Submit a video
curl -X POST http://localhost:8000/analyze \
  -F "file=@/path/to/video.mp4" \
  -F "llm_backend=ollama" \
  -F "ollama_model=llama3.2-vision"
# -> {"job_id": "abc-123", "status": "queued"}

# Poll status
curl http://localhost:8000/jobs/abc-123
```

---

## Configuration

Edit `configs/default.yaml` to change any setting:

```yaml
# Switch LLM backend
llm:
  backend: "ollama"          # auto | ollama | claude | stub
  ollama:
    model: "llama3.2:latest" # primary model; gpt-oss:latest used as secondary when available
  claude:
    model: "claude-sonnet-4-20250514"

# CV model - auto-downloads if not present locally
cv_agent:
  model: "yolov8m"           # yolov8n/s/m/l/x or yolov8n-obb for aerial
  weights: "weights/yolov8m.pt"
  conf_threshold: 0.30
  device: "cuda"             # cpu | cuda | mps

# ASR model size
asr_agent:
  model: "base"              # tiny | base | small | medium | large-v3
  device: "cuda"
  compute_type: "float16"

# SSIM adaptive sampling
sampling:
  ssim_threshold: 0.75       # lower = more frames processed
```

### Orchestration Runtime

```yaml
orchestration:
  backend: "harness_crewai"      # harness_crewai | crewai | local
  crewai:
    enabled: true                 # set false for deterministic CI runs
```

---

## Ollama model guide

| Model | Size | Vision | Role |
|---|---|---|---|
| `gpt-oss:latest` | varies | yes | VSR lip-reading + secondary ASR summary |
| `llama3.2-vision` | ~8 GB | yes | Fallback vision model |
| `llama3.2:latest` | 2 GB | no  | **Default primary** - ASR summary + fusion |

Vision models are required for VSR lip-reading. If none is configured, the pipeline auto-detects the first installed vision model via the Ollama `/api/tags` endpoint.

---

## Running tests

```bash
# Unit tests - no GPU, no API key, no weights needed
pytest tests/unit/ -v

# ASR summary eval (ROUGE + BLEU metrics, fully offline)
pytest tests/unit/test_asr_summary_evals.py -v

# Integration orchestration tests
pytest tests/integration/ -v

# Lint
ruff check src/ tests/ scripts/

# Type check
mypy src/ --ignore-missing-imports
```

### ASR eval metrics

The eval suite in `tests/unit/test_asr_summary_evals.py` measures LLM summary quality against human-written reference summaries using a stub client (no live LLM call):

| Fixture | ROUGE-1 F | ROUGE-L F | BLEU-1 | BLEU-2 |
|---|---|---|---|---|
| `tech_meeting` | 0.59 | 0.51 | 0.49 | 0.38 |
| `lecture_excerpt` | 0.42 | 0.22 | 0.28 | 0.16 |

Add new fixtures to `FIXTURES` in the test file to benchmark your own domain.

---

## Fine-tuning - Whisper LoRA

Fine-tunes `openai/whisper-large-v3` using LoRA (rank 32, alpha 64) on the `q_proj` and `v_proj` attention layers. Training data is LibriSpeech clean English speech with synthetic Gaussian noise augmentation (SNR 5-15 dB), teaching the adapter to transcribe accurately under noisy conditions. Only ~0.1% of parameters are trainable - the base model stays frozen. The trained adapter is then converted to CTranslate2 format (`whisper-lora-ct2/`) for deployment with `faster-whisper`. The current run (9 steps, 3 epochs) demonstrates the full pipeline: data loading -> noise augmentation -> LoRA training -> checkpoint saving -> CTranslate2 export.

```bash
cd finetune

# Install dependencies
pip install -r requirements.txt

# Run training
python src/train.py --config configs/config.yaml

# Evaluate
python src/evaluate.py --config configs/config.yaml

# Run inference with the trained adapter
python src/inference.py --config configs/config.yaml --audio path/to/audio.wav
```

> To adapt to your own domain, replace `data.dataset_name` in `finetune/configs/config.yaml` with your own audio dataset and provide reference transcripts. Increase `num_epochs` and `max_steps` accordingly.

---

## Docker Compose (full local stack)

```bash
# Start Qdrant + API
docker compose -f docker/docker-compose.yml up

# Services:
#   API:          http://localhost:8000
#   Qdrant:       http://localhost:6333
```

---

## Project structure

```
multimodal-video-intelligence/
+-- src/
|   +-- agents/
|   |   +-- cv_agent.py          # YOLOv8 (standard + OBB) + Florence-2 captioning & open-vocab detection
|   |   +-- asr_agent.py         # faster-whisper + X-vector speaker clustering
|   |   +-- vsr_agent.py         # MediaPipe mouth ROI + AV-HuBERT viseme decoder
|   +-- fusion/
|   |   +-- llm_client.py        # Unified Ollama / Claude / Stub client
|   |   +-- engine.py            # Cross-modal fusion synthesis
|   +-- orchestration/
|   |   +-- workflow_runtime.py  # Harness-style runtime + CrewAI synthesis
|   +-- api/
|   |   +-- server.py            # FastAPI + job queue
|   +-- utils/
|       +-- common.py            # Config loader, logging, device helpers
|       +-- sampler.py           # SSIM adaptive frame sampler
+-- configs/
|   +-- default.yaml             # All settings - edit this first
+-- scripts/
|   +-- run_pipeline.py          # Main CLI entrypoint
|   +-- streamlit_app.py         # Browser UI
+-- tests/
|   +-- unit/
|   |   +-- test_agents.py              # CV / ASR / VSR / sampler unit tests
|   |   +-- test_asr_summary_evals.py   # ROUGE + BLEU eval suite
|   +-- integration/
|       +-- test_workflow_runtime.py    # Orchestration contract tests
+-- finetune/
|   +-- src/
|   |   +-- train.py             # LoRA fine-tuning loop (PEFT + HuggingFace Trainer)
|   |   +-- dataset.py           # LibriSpeech loader, 16 kHz resampling, noise augmentation
|   |   +-- evaluate.py          # WER / CER evaluation against reference transcripts
|   |   +-- inference.py        # CTranslate2 inference with whisper-lora-ct2 weights
|   +-- models/
|   |   +-- whisper-lora/        # Trained LoRA adapter for whisper-large-v3 (9-step run, checkpoints at steps 3/6/9)
|   |   +-- whisper-lora-ct2/    # Same adapter converted to CTranslate2 format for faster-whisper inference
|   +-- configs/
|   |   +-- config.yaml          # LoRA hyperparams (r=32, alpha=64, target q_proj/v_proj)
|   +-- finetune_test.ipynb      # Interactive fine-tuning walkthrough
|   +-- pandas_data_manipulation.ipynb
|   +-- requirements.txt
|   +-- README.md
+-- docker/
|   +-- Dockerfile
|   +-- docker-compose.yml
+-- weights/
|   +-- yolov8n-obb.pt           # Aerial OBB weights (yolov8m auto-downloads)
+-- requirements.txt
+-- README.md
```

---

## License

MIT - see [LICENSE](LICENSE).

All ML components are open-source (Apache-2.0 / MIT / BSD).
