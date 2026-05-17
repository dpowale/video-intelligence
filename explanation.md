# Multimodal Video Intelligence — Step-by-Step Processing Sequence

## Step 1 — Video Upload (API Layer)

**What:** Client POSTs a video file to `/analyze`.  
**Decision:** File extension validated against allowlist `{.mp4, .mov, .mkv, .webm, .avi}`. Job written to in-memory `JOBS` dict with UUID. File saved to temp disk.  
**Why:** Avoids loading the entire file into memory; temp path passed downstream to ffmpeg and YOLO.

---

## Step 2 — Agent Load / Cache Check

**What:** `_load_agents()` checks `_agent_cache` keyed on `(frozenset(agents), asr_model)`.  
**Decision:** If cached → reuse existing model instances. If not → load fresh.  
**Why:** YOLOv8m (~26M params) + Whisper base + MediaPipe take ~5–10s to load. Caching means only the first request pays that cost.

**Models loaded:**

| Agent | Model |
|---|---|
| CV  | `yolov8m.pt` — 80-class COCO, 25.9M params, CUDA FP16 |
| ASR | `faster-whisper base` — CUDA FP16, multilingual |
| VSR | MediaPipe 468-point face mesh (CPU) |

---

## Step 3 — Frame Sampling (Adaptive)

**What:** `AdaptiveSampler` decodes frames and computes SSIM between consecutive frames.  
**Decision:** If SSIM > 0.75 → skip frame (scene hasn't changed). Otherwise emit frame. Fixed fallback: 1–2 FPS.  
**Why:** A 30-minute lecture at 25 FPS = 45,000 frames. Sampling to ~1 FPS with SSIM skipping reduces to ~200–500 frames — 90x less GPU work.

---

## Step 4 — CV + VSR Frame Processing (Parallel)

**What:** For each sampled frame, submit two thread tasks simultaneously into a 4-worker `ThreadPoolExecutor`.

### CV thread — YOLOv8m
- Runs `model.predict()` at conf ≥ 0.30, IoU ≤ 0.45
- Returns bounding boxes with class labels + confidences
- **Decision:** COCO 80-class covers people, vehicles, screens, furniture — sufficient for general intelligence. OBB model swappable for aerial footage.

### VSR thread — MediaPipe + Heuristic Visemes
- Detects 468 face landmarks
- Crops 112×112 mouth ROI
- Computes lip aspect ratio → `mouth_open` flag
- Maps mouth shape to viseme (visual phoneme unit)
- **Decision:** No deep VSR backbone loaded by default (too slow without specialist hardware). Heuristic visemes are fast; LLM handles actual lip-reading interpretation.

**Why parallel:** ASR is I/O bound (audio); CV/VSR are compute bound (GPU/CPU). Running them concurrently on separate threads yields near-linear speedup.

---

## Step 5 — ASR — Whisper Transcription (Parallel with Step 4)

**What:** ffmpeg extracts audio → 16kHz mono WAV → Whisper transcribes.

**Decision path:**
```
ffmpeg available? → extract .wav → Whisper
ffmpeg missing?   → Whisper directly on video file
ffmpeg fails?     → Whisper directly on video file
all fail?         → asr_skipped=True, continue
```

**Output:** Word-level timestamps at 100ms resolution, language detection.

**Speaker diarization:**
- **Gap-based (default):** silence gaps > 350ms → new speaker turn
- **X-Vector (SpeechBrain):** cluster embeddings → assign `spk_01`, `spk_02`...



---

## Step 6 — ASR Timestamp Alignment with VSR

**What:** Bisect-search maps each VSR frame's timestamp to the overlapping ASR segment.  
**Why:** Lets the synthesis prompt correlate "person's mouth was open at 12.3s" with "speaker said 'hello' at 12.1–12.8s".

---

## Step 7 — VSR Lip-Reading via Vision LLM (Conditional)

**What:** Calls `_vsr_lip_reading_with_llm()` — sends mouth crop images to Ollama.

**Decision:** Only runs if:
- Audio is absent/failed (`asr_skipped=True` or empty transcript), **AND**
- At least one mouth ROI was detected

**Why:** If Whisper already transcribed good audio, lip-reading adds nothing. It is a fallback for silent/noisy videos.  
**LLM used:** `llama3.2-vision:latest` via Ollama (preferred for vision tasks).  
**Prompt strategy:** Sends 3 temporally-spread 256×256 mouth crops in a single vision request with a lip-reading prompt. Returns estimated spoken text or empty string if model says "silence".

---

## Step 8 — LLM ASR Summary — Two Models in Parallel

**What:** `_summarize_asr_with_llm()` runs concurrently for two models in a 2-worker `ThreadPoolExecutor`.

| | Model | Purpose |
|---|---|---|
| Primary   | User-selected (default `gpt-oss:latest`) | Main transcript summary |
| Secondary | Opposite model (`llama3.2:latest` ↔ `gpt-oss:latest`) | Cross-model validation |

**Why two models:** Cross-validation. If both summaries agree, confidence is high. Results stored separately as `asr_summary` and `asr_summary_gpt_oss` and shown side-by-side in the UI.

**Prompt asks for:** (1) main topics, (2) speaker roles, (3) key facts/decisions/names, (4) tone/sentiment — condensed to 3–5 sentences.

---

## Step 9 — Intelligence Synthesis — LLM Report

**What:** `_run_crewai_synthesis()` builds a structured prompt from all agent outputs and asks the LLM to produce an operator report.  
**Decision:** Runs only if `crewai.enabled=true` AND a text LLM client is available. Falls back to a deterministic string summary if LLM fails.  
**LLM used:** Text model (same as ASR summary — `gpt-oss` or `llama3.2`).  
**Submitted:** Immediately after primary ASR summary resolves, overlapping with secondary summary.

**Prompt includes:**
- CV: top detected classes by confidence
- ASR: speaker turns with timestamps (up to 20)
- ASR summary (already computed in Step 8)
- VSR: mouth activity + lip-reading text (if any)

**Required output format:**
```
SCENE:        ...
SPEAKERS:     ...
OBJECTS:      ...
CROSS-MODAL:  ...
CONFIDENCE:   ...
ACTIONS:      ...
```

---

## Step 10 — Result Assembly + Response

All data merged into `metrics` dict → stored in `JOBS[job_id]` → status set to `done`. Streamlit polls `/jobs/{id}` every 500ms and renders the report when status flips.

---

## LLM Usage Summary

| LLM Call | Model | When | Purpose |
|---|---|---|---|
| Lip-reading | `llama3.2-vision` | Audio absent + mouth detected | Infer spoken words from mouth crops |
| ASR summary (primary) | user-selected (`gpt-oss`) | Always if transcript exists | Summarize transcript |
| ASR summary (secondary) | opposite model (`llama3.2`) | Always if transcript exists | Cross-model validation |
| Synthesis report | text model | Always (with deterministic fallback) | Structured multimodal intelligence report |

---

## Threading Architecture

```
Job Executor (max_workers=1)         ← serializes jobs; prevents concurrent CUDA access
└── run_video_workflow()
    ├── Frame pool (max_workers=4)   ← CV + VSR per frame, concurrent
    ├── Outer pool (max_workers=4)   ← ASR runs parallel to entire frame loop
    └── LLM pool (max_workers=2)     ← primary + secondary ASR summary concurrent
                                        synthesis submitted after primary resolves
```

## Model / Component Reference

| Component | Library | Device | Notes |
|---|---|---|---|
| YOLOv8m | ultralytics | CUDA FP16 | 80-class COCO, conf=0.30, iou=0.45 |
| Whisper base | faster-whisper | CUDA FP16 | multilingual, word timestamps |
| MediaPipe face mesh | mediapipe | CPU | 468 landmarks, mouth ROI 112×112 |
| SpeechBrain x-vector | speechbrain | CPU | speaker embeddings (random fallback) |
| AdaptiveSampler | opencv / scikit-image | CPU | SSIM threshold=0.75, 0.2–2 FPS |
| Ollama vision | llama3.2-vision | Ollama | lip-reading from mouth crops |
| Ollama text (primary) | gpt-oss / llama3.2 | Ollama | ASR summary + synthesis report |
| Qdrant | qdrant-client | remote | speaker name lookup (offline = disabled) |

---

## Metric Formulas

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE-1 and ROUGE-2 measure unigram and bigram overlap respectively.

$$\text{Precision}_n = \frac{|\text{matched } n\text{-grams}|}{|\text{n-grams in hypothesis}|}$$

$$\text{Recall}_n = \frac{|\text{matched } n\text{-grams}|}{|\text{n-grams in reference}|}$$

$$F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

- **ROUGE-1**: $n = 1$ (unigrams)
- **ROUGE-2**: $n = 2$ (bigrams)

#### ROUGE-L (Longest Common Subsequence)

Let $m$ = reference length, $n$ = hypothesis length, $\text{LCS}(X, Y)$ = length of the longest common subsequence.

$$R_\text{lcs} = \frac{\text{LCS}(X, Y)}{m} \qquad P_\text{lcs} = \frac{\text{LCS}(X, Y)}{n}$$

$$F_\text{lcs} = \frac{(1 + \beta^2)\, R_\text{lcs}\, P_\text{lcs}}{R_\text{lcs} + \beta^2\, P_\text{lcs}} \quad (\beta = 1 \Rightarrow \text{equal weight})$$

---

### BLEU (Bilingual Evaluation Understudy)

#### Modified n-gram Precision

Clips each n-gram count in the hypothesis to the maximum count in the reference:

$$p_n = \frac{\displaystyle\sum_{\text{n-gram} \in \hat{y}} \min\!\bigl(\text{count}(\text{n-gram in }\hat{y}),\; \text{count}(\text{n-gram in ref})\bigr)}{\displaystyle\sum_{\text{n-gram} \in \hat{y}} \text{count}(\text{n-gram in }\hat{y})}$$

#### Brevity Penalty

Penalises hypotheses shorter than the reference:

$$BP = \begin{cases} 1 & \text{if } c > r \\ e^{\,1 - r/c} & \text{if } c \leq r \end{cases}$$

where $c$ = hypothesis length (tokens), $r$ = reference length (tokens).

#### BLEU-N Score

$$\text{BLEU-N} = BP \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Uniform weights used here: $w_n = \dfrac{1}{N}$

| Score | Weights $(w_1, w_2, w_3, w_4)$ |
|---|---|
| BLEU-1 | $(1, 0, 0, 0)$ |
| BLEU-2 | $(0.5, 0.5, 0, 0)$ |
| BLEU-3 | $(\tfrac{1}{3}, \tfrac{1}{3}, \tfrac{1}{3}, 0)$ |
| BLEU-4 | $(0.25, 0.25, 0.25, 0.25)$ |

**Smoothing (method1):** When $p_n = 0$ (no n-gram overlap), adds 1 to both numerator and denominator to avoid $\log(0)$.

---

### WER / CER (Word / Character Error Rate)

Both use the **Levenshtein edit distance** between reference and hypothesis sequences.

$$\text{WER} = \frac{S + D + I}{N_\text{ref}}$$

$$\text{CER} = \frac{S_c + D_c + I_c}{N_{c,\text{ref}}}$$

| Symbol | Meaning |
|---|---|
| $S$ | Substitutions (wrong word/char) |
| $D$ | Deletions (missing word/char) |
| $I$ | Insertions (extra word/char) |
| $N_\text{ref}$ | Total words in reference |
| $N_{c,\text{ref}}$ | Total characters in reference |

- WER = 0.0 → perfect match; WER = 1.0 → completely wrong
- CER is always ≤ WER because character-level edits are finer-grained

---

## Fine-Tuning — Whisper LoRA

### What is being fine-tuned

**Model:** `openai/whisper-large-v3` — OpenAI's large multilingual speech-to-text model (1.5B parameters).  
**Goal:** Adapt it to a specific domain, accent, or noise profile without retraining all weights.

---

### Technique: LoRA (Low-Rank Adaptation)

Instead of updating all weights, LoRA injects small trainable rank-decomposition matrices into specific layers and keeps the base model frozen:

$$W' = W_0 + \Delta W = W_0 + BA$$

where $W_0$ is frozen, $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll d$.

The effective learning rate scale is $\frac{\alpha}{r}$. With the config values below: $\frac{64}{32} = 2.0$.

**Config (`finetune/configs/config.yaml`):**

| Parameter | Value | Meaning |
|---|---|---|
| `r` | 32 | Rank of decomposition matrices |
| `alpha` | 64 | Scaling factor — effective LR multiplier = alpha/r = 2.0 |
| `target_modules` | `q_proj`, `v_proj` | Only attention query + value projections adapted |
| `dropout` | 0.05 | Regularisation on LoRA layers |

Only ~0.1–1% of parameters are trainable. The full model stays frozen.

---

### Data Pipeline (`finetune/src/dataset.py`)

1. **Load:** `hf-internal-testing/librispeech_asr_dummy` (LibriSpeech clean validation split) from HuggingFace
2. **Resample:** all audio cast to 16kHz (required by Whisper's feature extractor)
3. **Noise augmentation** (if `apply_noise: true`): Gaussian noise added at random SNR between 5–15 dB

$$x_{\text{noisy}} = x + n, \quad n \sim \mathcal{N}(0,\, \sigma^2), \quad \sigma^2 = \frac{\|x\|^2 / N}{\text{SNR}}, \quad \text{SNR} = 10^{\,\text{snr\_db}/10}$$

4. **Feature extraction:** Whisper log-mel spectrogram (80 mel bins, 30s window) via `processor.feature_extractor`
5. **Tokenization:** reference text → token IDs as labels; padding positions replaced with `-100` so cross-entropy loss ignores them

---

### Training Loop (`finetune/src/train.py`)

| Setting | Value |
|---|---|
| Optimizer | AdamW (HuggingFace Trainer default) |
| Learning rate | 1e-3 |
| Batch size | 16 (× 2-step gradient accumulation → effective 32) |
| Epochs | 3 |
| Warmup steps | 50 (linear LR warmup) |
| Precision | FP16 disabled (CPU-mode for hardware compat) |
| Evaluation | Disabled during training (`eval_strategy="no"`) |
| Checkpoints | Saved every epoch |
| Logging | TensorBoard → `finetune/src/models/whisper-lora/runs/` |

Uses HuggingFace `Seq2SeqTrainer` with `predict_with_generate=True` so evaluation uses beam search, not teacher-forcing.

**Loss function:** Cross-entropy over token predictions, with `-100` labels masked out:

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(y_t \mid y_{<t},\, x)$$

where $x$ is the log-mel spectrogram input and $y_t$ is the ground-truth token at step $t$.

---

### Output

Saved to `finetune/models/whisper-lora/` — only the LoRA adapter weights (small, ~MB range). The base `whisper-large-v3` weights stay on HuggingFace and are merged at inference time via PEFT's `get_peft_model()`.

---

### Evaluation (`finetune/src/evaluate.py`)

After training, `evaluate.py` scores the fine-tuned model against ground-truth transcripts. It computes all metrics defined in the **Metric Formulas** section above:

| Metric | Measures |
|---|---|
| WER | Word-level transcription accuracy |
| CER | Character-level transcription accuracy |
| BLEU-1 to BLEU-4 | N-gram precision of generated text vs. reference |
| ROUGE-1/2/L | N-gram + LCS overlap of generated text vs. reference |

**Scope:** Corpus-level (`corpus_bleu`) — all samples averaged together, not per-sentence.
