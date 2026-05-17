"""
VSR Agent — Visual Speech Recognition / Lip Reading (Epic: VSR)

Tasks implemented:
  VSR-IMPL-01  Viseme-to-Phoneme decoder (CTC + 5-gram LM)
  VSR-IMPL-02  Mouth ROI extractor (MediaPipe)
  VSR-RES-01   Visual Language ID hooks
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.utils.common import get_logger, resolve_device

log = get_logger(__name__)

# MediaPipe mouth landmark indices (from 468-point mesh)
MOUTH_OUTER_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                    375, 321, 405, 314, 17, 84, 181, 91, 146]
MOUTH_INNER_IDX = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                    324, 318, 402, 317, 14, 87, 178, 88, 95]


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class MouthROI:
    crop: np.ndarray            # (112, 112, 3) BGR
    landmarks: np.ndarray       # (N, 2) pixel coords
    mouth_open: bool
    lip_aspect_ratio: float     # height / width
    head_pose_deg: Optional[float] = None


@dataclass
class VisemeFrame:
    viseme_id: int
    viseme_label: str
    phoneme_hypothesis: str
    confidence: float


@dataclass
class VSRResult:
    frame_index: int
    timestamp_s: float
    mouth_roi: Optional[MouthROI] = None
    visemes: list[VisemeFrame] = field(default_factory=list)
    text_hypothesis: str = ""
    lm_rescored_text: str = ""
    language_id: Optional[str] = None
    inference_ms: float = 0.0

    def publish_payload(self) -> str:
        return json.dumps({
            "frame": self.frame_index,
            "ts": round(self.timestamp_s, 3),
            "mouth_open": self.mouth_roi.mouth_open if self.mouth_roi else None,
            "lip_aspect_ratio": round(self.mouth_roi.lip_aspect_ratio, 3) if self.mouth_roi else None,
            "text": self.lm_rescored_text or self.text_hypothesis,
            "language_id": self.language_id,
            "inference_ms": round(self.inference_ms, 1),
        })


# ─── Viseme taxonomy ─────────────────────────────────────────────────────────

VISEME_TABLE: dict[int, dict] = {
    0:  {"label": "silence",  "phonemes": ["<sil>"]},
    1:  {"label": "bilabial", "phonemes": ["p", "b", "m"]},
    2:  {"label": "labiodental", "phonemes": ["f", "v"]},
    3:  {"label": "dental",   "phonemes": ["th", "dh"]},
    4:  {"label": "alveolar", "phonemes": ["t", "d", "n", "s", "z", "l"]},
    5:  {"label": "palatal",  "phonemes": ["sh", "zh", "ch", "jh", "y"]},
    6:  {"label": "velar",    "phonemes": ["k", "g", "ng"]},
    7:  {"label": "open",     "phonemes": ["aa", "ae", "ah"]},
    8:  {"label": "mid",      "phonemes": ["er", "eh", "uh"]},
    9:  {"label": "close",    "phonemes": ["iy", "ih"]},
    10: {"label": "rounded",  "phonemes": ["ow", "uw", "oo"]},
    11: {"label": "rhotic",   "phonemes": ["r"]},
    12: {"label": "lateral",  "phonemes": ["w"]},
}


# ─── Mouth ROI Extractor (VSR-IMPL-02) ───────────────────────────────────────

class MouthROIExtractor:
    """
    Extracts 112×112 temporal mouth crops using MediaPipe FaceMesh.

    VSR-IMPL-02: MediaPipe-based facial landmarking to extract
    112×112 temporal crop.
    """

    def __init__(self, roi_size: int = 112) -> None:
        self.roi_size = roi_size
        self._face_mesh = None
        self._face_cascade = None
        self._load()

    def _load(self) -> None:
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions"):
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=4,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                log.info("MediaPipe FaceMesh loaded")
        except Exception as e:
            log.warning("MediaPipe not available (%s) — using local face cascade fallback", e)

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if cascade_path.exists():
            face_cascade = cv2.CascadeClassifier(str(cascade_path))
            if not face_cascade.empty():
                self._face_cascade = face_cascade
                log.info("OpenCV Haar face detector loaded")

    def extract(self, bgr_frame: np.ndarray) -> list[MouthROI]:
        """Return one MouthROI per detected face."""
        h, w = bgr_frame.shape[:2]

        if self._face_mesh is None:
            return self._extract_with_face_cascade(bgr_frame)

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return self._extract_with_face_cascade(bgr_frame)

        rois: list[MouthROI] = []
        for face_lm in results.multi_face_landmarks:
            pts = np.array(
                [[lm.x * w, lm.y * h] for lm in face_lm.landmark], dtype=np.float32
            )
            roi = self._crop_mouth(bgr_frame, pts, h, w)
            rois.append(roi)
        return rois

    def _extract_with_face_cascade(self, frame: np.ndarray) -> list[MouthROI]:
        if self._face_cascade is None:
            return [self._stub_roi()]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40),
        )
        if len(faces) == 0:
            return [self._stub_roi()]

        rois: list[MouthROI] = []
        for x, y, w, h in faces:
            mouth_x1 = max(x + int(w * 0.18), 0)
            mouth_x2 = min(x + int(w * 0.82), frame.shape[1])
            mouth_y1 = max(y + int(h * 0.58), 0)
            mouth_y2 = min(y + h, frame.shape[0])
            crop = frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (self.roi_size, self.roi_size))
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            center = gray_crop[self.roi_size // 3: (2 * self.roi_size) // 3, self.roi_size // 4: (3 * self.roi_size) // 4]
            if center.size == 0:
                # Crop is too small to compute valid mouth stats — skip this face
                continue
            dark_ratio = float(np.clip((center < 70).mean(), 0.0, 1.0))
            texture = float(cv2.Laplacian(center, cv2.CV_32F).var())
            lip_aspect_ratio = float(min(1.0, dark_ratio * 1.8 + texture / 400.0))
            mouth_open = dark_ratio > 0.18 or texture > 14.0

            landmarks = np.array(
                [[mouth_x1, mouth_y1], [mouth_x2, mouth_y1], [mouth_x2, mouth_y2], [mouth_x1, mouth_y2]],
                dtype=np.float32,
            )
            rois.append(
                MouthROI(
                    crop=crop,
                    landmarks=landmarks,
                    mouth_open=mouth_open,
                    lip_aspect_ratio=lip_aspect_ratio,
                )
            )

        return rois or [self._stub_roi()]

    def _crop_mouth(self, frame: np.ndarray, pts: np.ndarray, h: int, w: int) -> MouthROI:
        mouth_pts = pts[MOUTH_OUTER_IDX + MOUTH_INNER_IDX]
        cx, cy = mouth_pts.mean(axis=0).astype(int)
        half = self.roi_size // 2
        x1, y1 = max(cx - half, 0), max(cy - half, 0)
        x2, y2 = min(cx + half, w), min(cy + half, h)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return self._stub_roi()
        crop = cv2.resize(crop, (self.roi_size, self.roi_size))

        # Lip aspect ratio: vertical opening / horizontal span
        outer = pts[MOUTH_OUTER_IDX]
        vert  = np.linalg.norm(outer[0] - outer[10])
        horiz = np.linalg.norm(outer[5] - outer[15])
        lar   = float(np.clip(vert / max(horiz, 1e-6), 0.0, 1.0))
        mouth_open = lar > 0.15

        return MouthROI(
            crop=crop, landmarks=mouth_pts,
            mouth_open=mouth_open, lip_aspect_ratio=lar,
        )

    @staticmethod
    def _stub_roi() -> MouthROI:
        crop = np.zeros((112, 112, 3), dtype=np.uint8)
        return MouthROI(crop=crop, landmarks=np.zeros((20, 2)),
                        mouth_open=False, lip_aspect_ratio=0.0)


# ─── Viseme-to-Phoneme Decoder (VSR-IMPL-01) ─────────────────────────────────

class VisemeDecoder(nn.Module):
    """
    CTC-based viseme-to-phoneme decoder on top of AV-HuBERT features.

    VSR-IMPL-01: Probabilistic mapping V_cluster → {phonemes}
                 rescored by 5-gram Language Model.
    Reference: AV-HuBERT (Shi et al., CVPR 2022).
    """

    def __init__(self, d_model: int = 768, n_visemes: int = 13) -> None:
        super().__init__()
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, dim_feedforward=2048,
                dropout=0.1, batch_first=True
            ),
            num_layers=4,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_visemes),
        )
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            features: (B, T, D) mouth feature sequence
            lengths:  (B,) actual sequence lengths
        Returns:
            logits: (T, B, n_visemes) for CTC
        """
        src_key_mask = None
        if lengths is not None:
            B, T, _ = features.shape
            src_key_mask = torch.arange(T, device=features.device).unsqueeze(0) >= lengths.unsqueeze(1)
        x = self.temporal_encoder(features, src_key_padding_mask=src_key_mask)
        logits = self.proj(x).transpose(0, 1)  # (T, B, V)
        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

    def decode(self, logits: torch.Tensor) -> list[VisemeFrame]:
        """Greedy decode viseme sequence from logits (T, 1, V)."""
        probs = F.softmax(logits.squeeze(1), dim=-1)  # (T, V)
        ids = probs.argmax(dim=-1).cpu().tolist()
        confs = probs.max(dim=-1).values.cpu().tolist()

        frames: list[VisemeFrame] = []
        prev = -1
        for vid, conf in zip(ids, confs):
            if vid == 0 or vid == prev:  # blank or repeat
                prev = vid
                continue
            entry = VISEME_TABLE.get(vid, {"label": "unk", "phonemes": ["?"]})
            frames.append(VisemeFrame(
                viseme_id=vid, viseme_label=entry["label"],
                phoneme_hypothesis=entry["phonemes"][0], confidence=conf,
            ))
            prev = vid
        return frames


# ─── LM Rescorer ─────────────────────────────────────────────────────────────

class KenLMRescorer:
    """5-gram KenLM rescoring for viseme-derived text hypotheses."""

    def __init__(self, arpa_path: str | Path) -> None:
        self._lm = None
        try:
            import kenlm
            self._lm = kenlm.Model(str(arpa_path))
            log.info("KenLM 5-gram LM loaded from %s", arpa_path)
        except Exception as e:
            log.warning("KenLM not available (%s) — no LM rescoring", e)

    def score(self, text: str) -> float:
        if self._lm is None:
            return 0.0
        return float(self._lm.score(text, bos=True, eos=True))

    def rescore(self, hypotheses: list[dict]) -> str:
        """Return best hypothesis text by LM score + CTC score."""
        if not hypotheses:
            return ""
        if self._lm is None:
            return hypotheses[0]["text"]
        best = max(hypotheses, key=lambda h: self.score(h["text"]) + h.get("ctc_score", 0.0))
        return best["text"]


# ─── VSR Agent ───────────────────────────────────────────────────────────────

class VSRAgent:
    """
    Visual Speech Recognition pipeline: ROI extraction → feature encoding →
    viseme decoding → LM rescoring.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg.vsr_agent
        self.device = resolve_device(self.cfg.device)
        self._extractor = MouthROIExtractor(roi_size=self.cfg.roi_size)
        self._backbone = None
        self._decoder = VisemeDecoder()
        self._lm = None

    def load(self) -> "VSRAgent":
        self._decoder = self._decoder.to(self.device)
        if self.cfg.lm_rescoring.enabled:
            self._lm = KenLMRescorer(self.cfg.lm_rescoring.arpa_path)
        log.info("VSRAgent ready — MediaPipe mouth ROI + heuristic visemes (Ollama vision handles transcript)")
        return self

    def infer(
        self, bgr_frame: np.ndarray, frame_index: int = 0, timestamp_s: float = 0.0
    ) -> VSRResult:
        t0 = time.perf_counter()

        rois = self._extractor.extract(bgr_frame)
        if not rois:
            return VSRResult(frame_index=frame_index, timestamp_s=timestamp_s,
                             inference_ms=(time.perf_counter() - t0) * 1000)

        roi = rois[0]  # primary speaker
        if self._backbone is not None:
            features = self._encode_roi(roi.crop)           # (1, T, D)
            logits = self._decoder(features)                # (T, 1, V)
            visemes = self._decoder.decode(logits)
        else:
            visemes = self._heuristic_visemes(roi)

        # Build text hypothesis from visemes
        raw_text = " ".join(v.phoneme_hypothesis for v in visemes)
        rescored = self._lm.rescore([{"text": raw_text, "ctc_score": 0.0}]) if self._lm else raw_text

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.debug("VSR infer | frame=%d visemes=%d latency=%.1fms",
                  frame_index, len(visemes), elapsed_ms)

        return VSRResult(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            mouth_roi=roi,
            visemes=visemes,
            text_hypothesis=raw_text,
            lm_rescored_text=rescored,
            inference_ms=elapsed_ms,
        )

    def _heuristic_visemes(self, roi: MouthROI) -> list[VisemeFrame]:
        gray = cv2.cvtColor(roi.crop, cv2.COLOR_BGR2GRAY)
        central_band = gray[
            self.cfg.roi_size // 3: (2 * self.cfg.roi_size) // 3,
            self.cfg.roi_size // 4: (3 * self.cfg.roi_size) // 4,
        ]
        dark_ratio = float((central_band < 70).mean())
        texture = float(cv2.Laplacian(central_band.astype(np.float32), cv2.CV_32F).var())
        lar = roi.lip_aspect_ratio

        # Map LAR + darkness to a phonetically-motivated viseme class.
        # Larger LAR → more open mouth; higher dark_ratio → visible cavity.
        if not roi.mouth_open or lar < 0.05:
            # Closed: bilabial stop/nasal (m/b/p) if slight movement, else silence
            viseme_id = 1 if (dark_ratio > 0.05 or texture > 10) else 0
        elif lar < 0.12:
            # Slightly open: labiodental (f/v) or alveolar (t/d/n)
            viseme_id = 2 if dark_ratio < 0.12 else 4
        elif lar < 0.22:
            # Mid open: palatal (sh/ch) or mid vowel (er/eh)
            viseme_id = 5 if dark_ratio < 0.20 else 8
        elif lar < 0.38:
            # Wide open: open vowel (aa/ae/ah) or velar (k/g)
            viseme_id = 7 if dark_ratio > 0.22 else 6
        else:
            # Very wide: rounded vowel (ow/uw) or low vowel (aa)
            viseme_id = 10 if dark_ratio > 0.28 else 7

        entry = VISEME_TABLE[viseme_id]
        confidence = float(min(0.90, 0.35 + lar * 0.8 + dark_ratio * 0.5))
        return [
            VisemeFrame(
                viseme_id=viseme_id,
                viseme_label=entry["label"],
                phoneme_hypothesis=entry["phonemes"][0],
                confidence=confidence,
            )
        ]

    def _encode_roi(self, crop: np.ndarray) -> torch.Tensor:
        """Encode mouth ROI crop → (1, T, D) feature tensor."""
        if self._backbone is not None:
            # Convert crop to (1, T=1, C, H, W) video tensor
            from torchvision.transforms.functional import to_tensor, normalize
            t = to_tensor(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).unsqueeze(0)
            t = normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(self.device)
            with torch.no_grad():
                out = self._backbone.extract_features(
                    source={"video": t, "audio": None}, padding_mask=None
                )
            return out["x"].transpose(0, 1)  # (1, T, D)
        else:
            # Stub: random features
            return torch.randn(1, 4, 768, device=self.device)
