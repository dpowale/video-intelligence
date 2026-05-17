"""Adaptive frame sampler using SSIM-based visual novelty detection (AER-RES-01)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from src.utils.common import get_logger

log = get_logger(__name__)


@dataclass
class SampledFrame:
    index: int
    timestamp_s: float
    bgr: np.ndarray                     # (H, W, 3)
    ssim_score: Optional[float] = None  # None for first frame
    novel: bool = True


@dataclass
class SamplerStats:
    total_frames: int = 0
    sampled_frames: int = 0
    skipped_frames: int = 0
    elapsed_s: float = 0.0

    @property
    def skip_rate(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.skipped_frames / self.total_frames

    @property
    def throughput_fps(self) -> float:
        if self.elapsed_s == 0:
            return 0.0
        return self.sampled_frames / self.elapsed_s


class AdaptiveSampler:
    """
    Samples video frames based on visual novelty (SSIM delta).

    AER-RES-01: Research SSIM thresholds to scale FPS dynamically
    based on visual novelty vs. compute cost.

    Args:
        source:         Path to video file or RTSP URL.
        ssim_threshold: Process frame if SSIM < threshold (lower = more frames).
        fixed_fps:      Fallback fixed sampling rate if strategy='fixed'.
        strategy:       'adaptive' or 'fixed'.
    """

    def __init__(
        self,
        source: str | Path,
        ssim_threshold: float = 0.92,
        fixed_fps: float = 2.0,
        strategy: str = "adaptive",
        gray_for_ssim: bool = True,
        max_frames: int = 0,
    ) -> None:
        self.source = str(source)
        self.ssim_threshold = ssim_threshold
        self.fixed_fps = fixed_fps
        self.strategy = strategy
        self.gray_for_ssim = gray_for_ssim
        self.max_frames = max_frames   # 0 = unlimited
        self.stats = SamplerStats()

    # ── Public API ────────────────────────────────────────────────────────────

    def stream(self) -> Generator[SampledFrame, None, None]:
        """Yield SampledFrame objects for each novel frame."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        native_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval: int = max(1, int(native_fps / self.fixed_fps))

        prev_gray: Optional[np.ndarray] = None  # stores thumbnail between frames
        idx = 0
        t0 = time.perf_counter()

        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break

                self.stats.total_frames += 1
                ts = idx / native_fps

                if self.strategy == "fixed" and idx % frame_interval != 0:
                    idx += 1
                    self.stats.skipped_frames += 1
                    continue

                if self.strategy == "fixed":
                    # In fixed mode, sample by interval only and skip SSIM gating.
                    self.stats.sampled_frames += 1
                    yield SampledFrame(
                        index=idx,
                        timestamp_s=ts,
                        bgr=bgr,
                        ssim_score=None,
                        novel=True,
                    )
                    if self.max_frames > 0 and self.stats.sampled_frames >= self.max_frames:
                        break
                    idx += 1
                    continue

                score, novel = self._is_novel(bgr, prev_gray)

                if novel:
                    prev_gray = self._to_thumb(bgr)
                    self.stats.sampled_frames += 1
                    yield SampledFrame(
                        index=idx,
                        timestamp_s=ts,
                        bgr=bgr,
                        ssim_score=score,
                        novel=True,
                    )
                    if self.max_frames > 0 and self.stats.sampled_frames >= self.max_frames:
                        break
                else:
                    self.stats.skipped_frames += 1

                idx += 1
        finally:
            cap.release()
            self.stats.elapsed_s = time.perf_counter() - t0
            log.info(
                "Sampler done | total=%d sampled=%d skipped=%d skip_rate=%.1f%%",
                self.stats.total_frames,
                self.stats.sampled_frames,
                self.stats.skipped_frames,
                self.stats.skip_rate * 100,
            )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _is_novel(
        self, bgr: np.ndarray, prev_thumb: Optional[np.ndarray]
    ) -> tuple[Optional[float], bool]:
        if prev_thumb is None:
            return None, True  # always process first frame

        curr_thumb = self._to_thumb(bgr)
        score = float(ssim(prev_thumb, curr_thumb, data_range=255))
        return score, score < self.ssim_threshold

    def _to_thumb(self, bgr: np.ndarray, w: int = 256, h: int = 144) -> np.ndarray:
        """Downscale to a thumbnail and convert to grayscale for fast SSIM."""
        small = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        if self.gray_for_ssim:
            return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return small

    def _to_gray(self, bgr: np.ndarray) -> np.ndarray:
        """Kept for API compatibility; use _to_thumb for SSIM comparisons."""
        if self.gray_for_ssim:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return bgr


# ─── Benchmark helper ────────────────────────────────────────────────────────

def benchmark_thresholds(
    video_path: str | Path,
    thresholds: list[float] | None = None,
) -> dict[float, SamplerStats]:
    """
    Sweep SSIM thresholds and return stats per threshold.
    Use this to find the optimal operating point (AER-RES-01 experiment).
    """
    thresholds = thresholds or [0.85, 0.88, 0.90, 0.92, 0.95, 0.98]
    results: dict[float, SamplerStats] = {}
    for t in thresholds:
        sampler = AdaptiveSampler(video_path, ssim_threshold=t)
        # consume generator without processing frames
        for _ in sampler.stream():
            pass
        results[t] = sampler.stats
        log.info(
            "threshold=%.2f  sampled=%d  skip_rate=%.1f%%",
            t, sampler.stats.sampled_frames, sampler.stats.skip_rate * 100,
        )
    return results
