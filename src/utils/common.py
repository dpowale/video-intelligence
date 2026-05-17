"""Shared utilities: config, logging, device management."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from omegaconf import DictConfig, OmegaConf

# ─── Logging ─────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    return logger


# ─── Config ──────────────────────────────────────────────────────────────────

def load_config(path: str | Path | None = None) -> DictConfig:
    """Load YAML config and merge with env overrides."""
    default = Path(__file__).parents[2] / "configs" / "default.yaml"
    path = Path(path) if path else default
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    cfg = OmegaConf.create(raw)
    # allow env overrides, e.g. MVI_CV_AGENT__DEVICE=cpu
    env_cfg = OmegaConf.from_dotlist(
        [f"{k[4:].lower().replace('__', '.')}={v}"
         for k, v in os.environ.items() if k.startswith("MVI_")]
    )
    return OmegaConf.merge(cfg, env_cfg)


# ─── Device ──────────────────────────────────────────────────────────────────

def resolve_device(requested: str = "cuda") -> torch.device:
    """Resolve device with graceful CPU fallback."""
    if requested == "cuda" and not torch.cuda.is_available():
        logging.getLogger(__name__).warning(
            "CUDA requested but not available — falling back to CPU"
        )
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def get_gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024 ** 3)


# ─── Video helpers ───────────────────────────────────────────────────────────

def probe_video(path: str | Path) -> dict[str, Any]:
    """Return basic video metadata without loading frames."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_s": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
    }
    cap.release()
    return meta
