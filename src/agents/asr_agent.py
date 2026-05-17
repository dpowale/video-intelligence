"""
ASR Agent — Speaker-Attributed Speech Recognition (Epic: SAA)

Tasks implemented:
  ASR-IMPL-01  Discrete timestamping (100ms Whisper tokens)
  ASR-IMPL-02  X-Vector persona clustering (Qdrant)
  ASR-RES-01   MSDD diarization research hooks
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.utils.common import get_logger, resolve_device

log = get_logger(__name__)


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class WordToken:
    word: str
    start_s: float
    end_s: float
    confidence: float


@dataclass
class SpeakerSegment:
    speaker_id: str                     # anonymous e.g. "spk_01"
    speaker_name: Optional[str]         # resolved from Qdrant
    start_s: float
    end_s: float
    words: list[WordToken] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words)


@dataclass
class ASRResult:
    frame_index: int
    timestamp_s: float
    segments: list[SpeakerSegment] = field(default_factory=list)
    language: str = "en"
    inference_ms: float = 0.0

    def publish_payload(self) -> str:
        return json.dumps({
            "frame": self.frame_index,
            "ts": round(self.timestamp_s, 3),
            "language": self.language,
            "segments": [
                {
                    "speaker": s.speaker_id,
                    "speaker_name": s.speaker_name,
                    "start": round(s.start_s, 3),
                    "end": round(s.end_s, 3),
                    "text": s.text,
                }
                for s in self.segments
            ],
            "inference_ms": round(self.inference_ms, 1),
        })


# ─── GE2E Loss (speaker contrastive training) ─────────────────────────────────

class GE2ELoss(nn.Module):
    """
    Generalized End-to-End Speaker Embedding Loss.
    Reference: Wan et al., 2018 — Generalized End-to-End Loss for Speaker Verification.
    """

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (N_speakers, M_utterances, D)
        Returns:
            scalar GE2E loss
        """
        N, M, D = embeddings.shape
        centroids = embeddings.mean(dim=1)                      # (N, D)
        centroids_norm = nn.functional.normalize(centroids, dim=-1)
        emb_norm = nn.functional.normalize(embeddings, dim=-1)  # (N, M, D)

        # Similarity matrix: (N, M, N)
        sim = torch.einsum("nmd,kd->nmk", emb_norm, centroids_norm)
        sim = self.w * sim + self.b

        # GE2E softmax loss
        target = torch.arange(N, device=embeddings.device).unsqueeze(1).expand(N, M)
        loss = nn.functional.cross_entropy(
            sim.reshape(N * M, N), target.reshape(N * M)
        )
        return loss


# ─── Whisper Fine-Tuning Helpers (ASR-IMPL-01) ───────────────────────────────

def build_timestamped_whisper(
    base_model: str = "Systran/faster-whisper-large-v3",
    resolution_ms: int = 100,
    lora_r: int = 16,
) -> tuple:
    """
    Extend Whisper tokenizer with <|time_idx_NNN|> tokens and resize embeddings.

    ASR-IMPL-01: Fine-tune Whisper with 100ms timestamp tokens.
    Uses LoRA (peft) for memory-efficient fine-tuning.

    Returns:
        (model, tokenizer, peft_config)
    """
    from transformers import WhisperForConditionalGeneration, WhisperTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    log.info("Building timestamped Whisper from %s", base_model)
    tokenizer = WhisperTokenizer.from_pretrained(base_model)

    # Add time tokens at resolution_ms intervals (10 seconds = 100 tokens at 100ms)
    n_tokens = int(10_000 / resolution_ms)
    new_tokens = [f"<|time_idx_{i:04d}|>" for i in range(n_tokens)]
    n_added = tokenizer.add_tokens(new_tokens)
    log.info("Added %d time-index tokens at %dms resolution", n_added, resolution_ms)

    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    model.resize_token_embeddings(len(tokenizer))

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer, lora_cfg


# ─── Speaker Embedding (ASR-IMPL-02) ─────────────────────────────────────────

class SpeakerEmbedder:
    """
    Extracts 192-dim ECAPA-TDNN embeddings and resolves identities via Qdrant.

    ASR-IMPL-02: X-Vector Persona Cluster.
    Uses speechbrain/spkrec-ecapa-voxceleb (no k2 dependency).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg.asr_agent
        self._classifier = None
        self._qdrant = None

    def load(self) -> "SpeakerEmbedder":
        try:
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
            except ImportError:
                from speechbrain.pretrained import EncoderClassifier  # speechbrain < 1.0 fallback
            import torch as _torch
            _sb_device = "cuda" if _torch.cuda.is_available() else "cpu"
            self._classifier = EncoderClassifier.from_hparams(
                source=self.cfg.speaker_embedding.model,
                run_opts={"device": _sb_device},
            )
            log.info("X-vector model loaded on %s", _sb_device)
        except Exception as e:
            log.warning("SpeechBrain not available (%s) — using random embeddings", e)

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            self._qdrant = QdrantClient(
                host=self.cfg.qdrant.host, port=self.cfg.qdrant.port
            )
            collections = [c.name for c in self._qdrant.get_collections().collections]
            if self.cfg.qdrant.collection not in collections:
                self._qdrant.create_collection(
                    self.cfg.qdrant.collection,
                    vectors_config=VectorParams(
                        size=self.cfg.speaker_embedding.dim,
                        distance=Distance.COSINE,
                    ),
                )
                log.info("Created Qdrant collection '%s'", self.cfg.qdrant.collection)
        except Exception as e:
            log.warning("Qdrant not available (%s) — speaker lookup disabled", e)

        return self

    def embed(self, waveform: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Return (dim,) unit-norm speaker embedding for a waveform at 16 kHz."""
        if self._classifier is None:
            # Stub: unseeded RNG so different clips always get distinct embeddings.
            rng = np.random.default_rng()
            v = rng.standard_normal(self.cfg.speaker_embedding.dim).astype(np.float32)
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else v
        import torch
        # SpeechBrain EncoderClassifier expects 16 kHz; resample if needed
        if sample_rate != 16000:
            try:
                import torchaudio
                t = torch.from_numpy(waveform).unsqueeze(0)
                t = torchaudio.functional.resample(t, sample_rate, 16000)
                waveform = t.squeeze(0).numpy()
            except Exception:
                pass  # proceed with original rate
        tensor = torch.from_numpy(waveform).unsqueeze(0)
        with torch.no_grad():
            emb = self._classifier.encode_batch(tensor)
        v = emb.squeeze().numpy().astype(np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def register_speaker(self, speaker_id: int, embedding: np.ndarray, meta: dict) -> None:
        """Upsert speaker embedding into Qdrant."""
        if self._qdrant is None:
            return
        from qdrant_client.models import PointStruct
        self._qdrant.upsert(
            self.cfg.qdrant.collection,
            points=[PointStruct(id=speaker_id, vector=embedding.tolist(), payload=meta)],
        )

    def resolve_identity(self, embedding: np.ndarray, threshold: float = 0.75) -> Optional[str]:
        """Return speaker name if a close match exists in Qdrant."""
        if self._qdrant is None:
            return None
        hits = self._qdrant.search(
            self.cfg.qdrant.collection,
            query_vector=embedding.tolist(),
            limit=1,
        )
        if hits and hits[0].score >= threshold:
            return hits[0].payload.get("name")
        return None


# ─── ASR Agent ───────────────────────────────────────────────────────────────

class ASRAgent:
    """
    Transcribes audio with sub-100ms timestamps and speaker attribution.

    Uses faster-whisper for production inference; falls back to stub.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg.asr_agent
        self.device = resolve_device(cfg.asr_agent.device)
        self._whisper = None
        self._embedder = SpeakerEmbedder(cfg)
        self._diarizer = None
        self._whisper_model_name = str(self.cfg.model)

    def load(self) -> "ASRAgent":
        try:
            from faster_whisper import WhisperModel

            compute_type = str(self.cfg.compute_type)
            if self.device.type != "cuda" and compute_type == "float16":
                compute_type = "int8"

            model_candidates = [str(self.cfg.model)]
            if str(self.cfg.model) != "base":
                model_candidates.append("base")

            last_error: Exception | None = None
            for model_name in model_candidates:
                try:
                    self._whisper = WhisperModel(
                        model_name,
                        device=self.device.type,
                        compute_type=compute_type,
                        cpu_threads=4,
                        num_workers=2,
                    )
                    self._whisper_model_name = model_name
                    log.info("Whisper loaded: %s on %s (%s)", model_name, self.device, compute_type)
                    break
                except Exception as e:
                    last_error = e

            if self._whisper is None and last_error is not None:
                raise last_error
        except Exception as e:
            log.warning("faster-whisper not available (%s) — stub mode", e)

        self._embedder.load()
        self._load_diarizer()
        return self

    def _load_diarizer(self) -> None:
        backend = self.cfg.diarization.backend
        if backend == "pyannote":
            try:
                from pyannote.audio import Pipeline
                # token kwarg replaced use_auth_token in pyannote >= 3.2
                try:
                    self._diarizer = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        token=None,
                    )
                except TypeError:
                    self._diarizer = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                    )
                log.info("pyannote diarization loaded")
            except Exception as e:
                log.warning("pyannote not available (%s)", e)
        elif backend == "gap":
            log.info("Gap-based speaker segmentation enabled")
        elif backend == "nemo_msdd":
            log.info("NeMo MSDD diarization backend selected (ASR-RES-01)")

    @staticmethod
    def _build_words(seg: object) -> list[WordToken]:
        return [
            WordToken(
                word=w.word.strip(),
                start_s=w.start,
                end_s=w.end,
                confidence=w.probability,
            )
            for w in (getattr(seg, "words", None) or [])
        ]

    @staticmethod
    def _overlap_duration(start_s: float, end_s: float, other_start_s: float, other_end_s: float) -> float:
        return max(0.0, min(end_s, other_end_s) - max(start_s, other_start_s))

    def _assign_gap_speakers(self, raw_segments: list[object]) -> list[SpeakerSegment]:
        speaker_segments: list[SpeakerSegment] = []
        speaker_counter = 0
        prev_end: float | None = None
        speaker_change_gap_s = 1.2

        for seg in raw_segments:
            if prev_end is not None and (seg.start - prev_end) > speaker_change_gap_s:
                speaker_counter += 1

            speaker_segments.append(
                SpeakerSegment(
                    speaker_id=f"spk_{speaker_counter:02d}",
                    speaker_name=f"Speaker {speaker_counter + 1}",
                    start_s=seg.start,
                    end_s=seg.end,
                    words=self._build_words(seg),
                )
            )
            prev_end = seg.end

        return speaker_segments

    # ── X-vector clustering ───────────────────────────────────────────────────

    @staticmethod
    def _load_audio_mono_16k(audio_path: str) -> tuple[np.ndarray, int]:
        """Load audio as mono float32 at 16 kHz via torchaudio."""
        import torchaudio  # declared dep
        waveform, sr = torchaudio.load(str(audio_path))   # (channels, T)
        waveform = waveform.mean(dim=0)                    # mono
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        return waveform.numpy().astype(np.float32), 16000

    def _cluster_by_embeddings(
        self,
        audio_path: str,
        segments: list[SpeakerSegment],
        sim_threshold: float = 0.72,
    ) -> list[SpeakerSegment]:
        """
        Re-label segments using x-vector cosine clustering so that the same
        physical speaker always gets the same label across pauses and gaps.

        Algorithm:
          1. Load audio; slice per-segment waveform.
          2. Embed each slice with SpeakerEmbedder (192-dim x-vector).
          3. Greedy nearest-centroid clustering (cosine distance).
          4. Resolve cluster centroids against Qdrant → named speaker or
             fallback to alphabetical label (Speaker A, Speaker B …).
        """
        if not segments:
            return segments

        # ── 1. Load audio ─────────────────────────────────────────────────
        try:
            data, sr = self._load_audio_mono_16k(audio_path)
        except Exception as exc:
            log.warning("Speaker clustering: audio load failed (%s) — keeping initial labels", exc)
            return segments

        # ── 2. Embed each segment ─────────────────────────────────────────
        embeddings: list[np.ndarray | None] = []
        min_samples = int(sr * 0.4)   # skip segments shorter than 400 ms

        for seg in segments:
            start = max(0, int(seg.start_s * sr))
            end   = min(len(data), int(seg.end_s * sr))
            chunk = data[start:end]
            if len(chunk) < min_samples:
                embeddings.append(None)
            else:
                try:
                    embeddings.append(self._embedder.embed(chunk, sample_rate=sr))
                except Exception:
                    embeddings.append(None)

        # ── 3. Greedy cosine clustering ───────────────────────────────────
        cluster_ids: list[int]           = []
        centroids:   list[np.ndarray]    = []
        cluster_counts: list[int]        = []

        for emb in embeddings:
            if emb is None:
                # Inherit previous cluster (same speaker most likely)
                cluster_ids.append(cluster_ids[-1] if cluster_ids else 0)
                continue

            best_cluster = -1
            best_sim     = sim_threshold
            if centroids:
                # Vectorized cosine similarity against all current centroids
                centroid_matrix = np.stack(centroids)          # (K, dim)
                sims = centroid_matrix @ emb                   # (K,) — both unit-norm
                max_idx = int(np.argmax(sims))
                if sims[max_idx] > best_sim:
                    best_sim     = float(sims[max_idx])
                    best_cluster = max_idx

            if best_cluster == -1:
                # New speaker cluster
                best_cluster = len(centroids)
                centroids.append(emb.copy())
                cluster_counts.append(1)
            else:
                # Running mean centroid update
                n = cluster_counts[best_cluster]
                updated = (centroids[best_cluster] * n + emb) / (n + 1)
                norm = np.linalg.norm(updated)
                centroids[best_cluster] = updated / norm if norm > 0 else updated
                cluster_counts[best_cluster] += 1

            cluster_ids.append(best_cluster)

        # ── 4. Resolve names ──────────────────────────────────────────────
        cluster_names: dict[int, str] = {}
        for idx, centroid in enumerate(centroids):
            name = self._embedder.resolve_identity(centroid)
            cluster_names[idx] = name if name else f"Speaker {idx + 1}"

        # If we ended up with zero centroids (all embeddings were None)
        if not centroids:
            return segments

        # ── 5. Rebuild segments ───────────────────────────────────────────
        result: list[SpeakerSegment] = []
        for seg, cid in zip(segments, cluster_ids):
            result.append(SpeakerSegment(
                speaker_id=f"spk_{cid:02d}",
                speaker_name=cluster_names.get(cid, f"Speaker {cid + 1}"),
                start_s=seg.start_s,
                end_s=seg.end_s,
                words=seg.words,
            ))

        n_speakers = len(centroids)
        log.info("Speaker clustering: %d segment(s) → %d unique speaker(s)", len(segments), n_speakers)
        return result

    def _assign_diarized_speakers(self, raw_segments: list[object], diarization: object) -> list[SpeakerSegment]:
        diarized_turns = [
            (float(turn.start), float(turn.end), str(label))
            for turn, _, label in diarization.itertracks(yield_label=True)
        ]
        if not diarized_turns:
            return self._assign_gap_speakers(raw_segments)

        label_to_index: dict[str, int] = {}
        speaker_segments: list[SpeakerSegment] = []

        for seg in raw_segments:
            overlaps: dict[str, float] = {}
            for turn_start, turn_end, label in diarized_turns:
                overlap_s = self._overlap_duration(seg.start, seg.end, turn_start, turn_end)
                if overlap_s > 0:
                    overlaps[label] = overlaps.get(label, 0.0) + overlap_s

            if overlaps:
                speaker_label = max(overlaps.items(), key=lambda item: item[1])[0]
            else:
                speaker_label = speaker_segments[-1].speaker_id if speaker_segments else "spk_00"

            if speaker_label not in label_to_index:
                label_to_index[speaker_label] = len(label_to_index)

            speaker_index = label_to_index[speaker_label]
            speaker_segments.append(
                SpeakerSegment(
                    speaker_id=f"spk_{speaker_index:02d}",
                    speaker_name=f"Speaker {speaker_index + 1}",
                    start_s=seg.start,
                    end_s=seg.end,
                    words=self._build_words(seg),
                )
            )

        return speaker_segments

    def transcribe(
        self, audio_path: str | Path, frame_index: int = 0, timestamp_s: float = 0.0
    ) -> ASRResult:
        """Transcribe audio file; return ASRResult with speaker-attributed segments."""
        t0 = time.perf_counter()
        segments: list[SpeakerSegment] = []

        if self._whisper is not None:
            raw_segments, info = self._whisper.transcribe(
                str(audio_path),
                word_timestamps=True,
                language=self.cfg.get("language"),
            )
            language = info.language

            # Collect raw segments first so we can apply gap-based speaker
            # change detection when diarization is unavailable.
            raw_list = list(raw_segments)  # materialise the generator

            if self._diarizer is not None:
                try:
                    diarization = self._diarizer(
                        str(audio_path),
                        min_speakers=int(self.cfg.diarization.min_speakers),
                        max_speakers=int(self.cfg.diarization.max_speakers),
                    )
                    segments = self._assign_diarized_speakers(raw_list, diarization)
                except Exception as e:
                    log.warning("pyannote diarization failed (%s) - falling back to gap mode", e)
                    segments = self._assign_gap_speakers(raw_list)
            else:
                segments = self._assign_gap_speakers(raw_list)

            # X-vector clustering: correct speaker labels so that the same
            # person across multiple pauses/gaps gets a consistent identity.
            segments = self._cluster_by_embeddings(str(audio_path), segments)
        else:
            language = "en"
            segments = [SpeakerSegment(
                speaker_id="spk_00", speaker_name="Speaker 1",
                start_s=0.0, end_s=1.0,
                words=[WordToken("(stub)", 0.0, 1.0, 1.0)],
            )]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.debug("ASR done | frame=%d segs=%d latency=%.1fms", frame_index, len(segments), elapsed_ms)
        return ASRResult(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            segments=segments,
            language=language,
            inference_ms=elapsed_ms,
        )
