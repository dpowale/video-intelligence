#!/usr/bin/env python3
"""
CLI entrypoint — Multimodal Video Intelligence Platform.

Usage:
    # Stub mode (no GPU, no model weights, no API key required)
    python scripts/run_pipeline.py --video path/to/video.mp4

    # Auto-detect LLM: Ollama first, then Claude, then stub
    python scripts/run_pipeline.py --video path/to/video.mp4 --llm auto

    # Force Ollama (local, 100% open-source, free)
    python scripts/run_pipeline.py --video path/to/video.mp4 --llm ollama

    # Specific Ollama model
    python scripts/run_pipeline.py --video path/to/video.mp4 --llm ollama --ollama-model llava

    # Force Claude (requires ANTHROPIC_API_KEY env var)
    python scripts/run_pipeline.py --video path/to/video.mp4 --llm claude

    # SSIM threshold benchmark (AER-RES-01 experiment)
    python scripts/run_pipeline.py --benchmark-ssim path/to/video.mp4

    # Start REST API server
    python scripts/run_pipeline.py --serve --port 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.utils.common import get_logger, probe_video
from src.utils.sampler import benchmark_thresholds
from src.fusion.llm_client import LLMClient
from src.orchestration.workflow_runtime import run_video_workflow

log = get_logger("run_pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MVI Pipeline — Multimodal Video Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video", type=str, help="Path to input video file")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--llm",
        choices=["auto", "ollama", "claude", "stub"],
        default="stub",
        help="LLM backend (default: stub — no API key needed)",
    )
    p.add_argument("--ollama-model", default="llama3.2-vision",
                   help="Ollama model (default: llama3.2-vision)")
    p.add_argument("--ollama-host", default=None,
                   help="Ollama server URL (default: http://localhost:11434)")
    p.add_argument("--claude-model", default="claude-sonnet-4-20250514")
    p.add_argument("--output", type=str, default=None,
                   help="Write JSON output to file (default: stdout)")
    p.add_argument("--benchmark-ssim", type=str, metavar="VIDEO",
                   help="Run SSIM threshold sweep and exit")
    p.add_argument("--serve", action="store_true", help="Start FastAPI server")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--orchestration",
        choices=["harness_crewai", "crewai", "local"],
        default=None,
        help="Override orchestration backend from config",
    )
    return p.parse_args()


def build_llm(args: argparse.Namespace) -> LLMClient:
    if args.llm == "ollama":
        log.info("LLM: Ollama  model=%s", args.ollama_model)
        return LLMClient.ollama(model=args.ollama_model, host=args.ollama_host)
    if args.llm == "claude":
        log.info("LLM: Claude  model=%s", args.claude_model)
        return LLMClient.claude(model=args.claude_model)
    if args.llm == "auto":
        return LLMClient.auto(
            ollama_model=args.ollama_model,
            claude_model=args.claude_model,
        )
    log.info("LLM: stub (no external calls)")
    return LLMClient.stub()


def run_pipeline(video_path: str, cfg, llm: LLMClient, args: argparse.Namespace) -> dict:
    _ = cfg, llm
    return run_video_workflow(
        video_path,
        llm_backend=args.llm,
        llm_ollama_model=args.ollama_model,
        llm_ollama_host=args.ollama_host,
        llm_claude_model=args.claude_model,
    )


def main() -> None:
    args = parse_args()

    if args.serve:
        import uvicorn
        uvicorn.run("src.api.server:app", host="0.0.0.0", port=args.port, reload=False)
        return

    if args.benchmark_ssim:
        results = benchmark_thresholds(args.benchmark_ssim)
        print(f"\n{'Threshold':<12} {'Sampled':>8} {'Skip%':>7} {'fps':>8}")
        print("-" * 40)
        for t, s in sorted(results.items()):
            print(f"{t:<12.2f} {s.sampled_frames:>8} {s.skip_rate*100:>6.1f}% {s.throughput_fps:>7.1f}")
        return

    if not args.video:
        print("Error: --video required. Use --help for options.", file=sys.stderr)
        sys.exit(1)

    if not Path(args.video).exists():
        print(f"Error: file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    meta = probe_video(args.video)
    log.info(
        "Video: %s | %.1fs | %dx%d | %.1f fps",
        Path(args.video).name,
        meta["duration_s"], meta["width"], meta["height"], meta["fps"],
    )

    result = run_video_workflow(
        video_path=args.video,
        config_path=args.config,
        backend=args.orchestration,
        llm_backend=args.llm,
        llm_ollama_model=args.ollama_model,
        llm_ollama_host=args.ollama_host,
        llm_claude_model=args.claude_model,
    )
    out = json.dumps(result, indent=2)

    if args.output:
        Path(args.output).write_text(out)
        log.info("Results written to %s", args.output)
    else:
        print(out)


if __name__ == "__main__":
    main()
