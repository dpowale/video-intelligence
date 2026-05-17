"""
Shared pytest fixtures for the multimodal-video-intelligence test suite.

All fixtures are CPU-only and require no external services, model weights,
or API keys.  They generate purely synthetic data so tests stay fast and
deterministic.
"""
from __future__ import annotations

import types
from pathlib import Path

import cv2
import numpy as np
import pytest


# ─── Synthetic mouth-crop helpers ────────────────────────────────────────────

def _make_mouth_crop(
    roi_size: int = 112,
    *,
    dark_center: bool = False,
    brightness: int = 180,
) -> np.ndarray:
    """
    Return a (roi_size × roi_size × 3) uint8 BGR array that mimics a mouth crop.

    Parameters
    ----------
    dark_center : bool
        If True, paint the central region black to simulate an open-mouth cavity.
    brightness  : int
        Base pixel value for the surrounding skin region (0–255).
    """
    crop = np.full((roi_size, roi_size, 3), brightness, dtype=np.uint8)
    if dark_center:
        h1, h2 = roi_size // 3, (2 * roi_size) // 3
        w1, w2 = roi_size // 4, (3 * roi_size) // 4
        crop[h1:h2, w1:w2] = 0   # dark cavity
    return crop


def _make_mouth_roi(*, open_mouth: bool, lar: float = 0.35, roi_size: int = 112):
    """Return a SimpleNamespace that matches the MouthROI dataclass contract."""
    crop = _make_mouth_crop(roi_size=roi_size, dark_center=open_mouth)
    return types.SimpleNamespace(
        crop=crop,
        landmarks=np.zeros((20, 2), dtype=np.float32),
        mouth_open=open_mouth,
        lip_aspect_ratio=lar,
        head_pose_deg=None,
    )


def _make_vsr_result(
    frame_index: int,
    timestamp_s: float,
    *,
    open_mouth: bool = True,
    lar: float = 0.35,
    viseme_id: int = 7,
    viseme_label: str = "open",
    phoneme: str = "aa",
    text_hypothesis: str = "",
):
    """Return a SimpleNamespace matching the VSRResult dataclass contract."""
    return types.SimpleNamespace(
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        mouth_roi=_make_mouth_roi(open_mouth=open_mouth, lar=lar),
        visemes=[
            types.SimpleNamespace(
                viseme_id=viseme_id,
                viseme_label=viseme_label,
                phoneme_hypothesis=phoneme,
                confidence=0.75,
            )
        ],
        text_hypothesis=text_hypothesis,
        lm_rescored_text=text_hypothesis,
        language_id=None,
        inference_ms=5.0,
    )


# ─── Pytest fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def mouth_crop_open() -> np.ndarray:
    """112×112 BGR crop with a dark central region (open mouth)."""
    return _make_mouth_crop(dark_center=True)


@pytest.fixture()
def mouth_crop_closed() -> np.ndarray:
    """112×112 BGR crop with uniform bright tone (closed/bilabial mouth)."""
    return _make_mouth_crop(dark_center=False, brightness=200)


@pytest.fixture()
def mouth_roi_open():
    """MouthROI-compatible namespace — mouth open, LAR 0.4."""
    return _make_mouth_roi(open_mouth=True, lar=0.4)


@pytest.fixture()
def mouth_roi_closed():
    """MouthROI-compatible namespace — mouth closed, LAR 0.02."""
    return _make_mouth_roi(open_mouth=False, lar=0.02)


@pytest.fixture()
def vsr_sequence_open():
    """
    Six-frame VSR result sequence — all open-mouth, spread over 0–2.5 s.
    Simulates a speaker mid-utterance with visible lip movement.
    """
    return [
        _make_vsr_result(i, round(i * 0.5, 1), open_mouth=True, lar=0.35 + i * 0.02)
        for i in range(6)
    ]


@pytest.fixture()
def vsr_sequence_silent():
    """
    Six-frame VSR result sequence — all closed-mouth, spread over 0–2.5 s.
    Simulates a video where no speech is visible (silence or off-camera speaker).
    """
    return [
        _make_vsr_result(i, round(i * 0.5, 1), open_mouth=False, lar=0.02)
        for i in range(6)
    ]


@pytest.fixture()
def vsr_sequence_mixed():
    """
    Eight-frame VSR sequence with alternating open/closed — mimics natural speech
    rhythm with pauses.
    """
    open_flags = [False, True, True, False, True, True, False, False]
    lars       = [0.02,  0.38, 0.41, 0.03, 0.35, 0.42, 0.04, 0.02]
    return [
        _make_vsr_result(i, round(i * 0.3, 1), open_mouth=o, lar=l)
        for i, (o, l) in enumerate(zip(open_flags, lars))
    ]


@pytest.fixture()
def silent_video_path(tmp_path) -> Path:
    """
    Write a small synthetic MP4 with NO audio stream.
    5 seconds at 10 fps, 320×240, all grey frames.
    Returns a Path object.
    """
    vpath = tmp_path / "silent_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vpath), fourcc, 10, (320, 240))
    for i in range(50):   # 5 s × 10 fps
        # Slightly varying brightness so SSIM detects novelty occasionally
        level = 100 + (i % 10) * 5
        out.write(np.full((240, 320, 3), level, dtype=np.uint8))
    out.release()
    return vpath


@pytest.fixture()
def speaking_video_path(tmp_path) -> Path:
    """
    Synthetic MP4 with a black region in the mouth area on alternating frames
    (simulates visible lip movement for the extractor heuristic).
    No audio stream.
    """
    vpath = tmp_path / "speaking_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vpath), fourcc, 10, (320, 240))
    for i in range(50):
        frame = np.full((240, 320, 3), 160, dtype=np.uint8)
        if i % 2 == 0:
            # Simulate open-mouth dark region in lower third of frame
            frame[160:220, 110:210] = 10
        out.write(frame)
    out.release()
    return vpath
