"""
Cross-modal fusion engine.

Assembles CV + ASR + VSR payloads and calls the configured LLM
(Ollama or Claude) to produce a unified intelligence report.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.fusion.llm_client import LLMClient, LLMResponse
from src.utils.common import get_logger

log = get_logger(__name__)

FUSION_SYSTEM = """You are a multimodal video intelligence fusion engine.
You receive synchronized outputs from three specialized agents:
  - CV  (Computer Vision): object detections, scene type, geo-coordinates
  - ASR (Speech Recognition): transcription with speaker attribution, timestamps
  - VSR (Visual Speech Recognition): lip state, visemes, mouth-based language ID

Your task: synthesize these streams into a concise, structured intelligence report.

Format your response as:
SCENE: <one-line scene description>
SPEAKERS: <who is talking and what they said>
OBJECTS: <key detected objects / entities>
CROSS-MODAL: <confirmations or contradictions between streams>
CONFIDENCE: <high / medium / low with brief reason>
ACTIONS: <recommended follow-up>"""


@dataclass
class FusionReport:
    frame_index: int
    timestamp_s: float
    report: str
    backend: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "frame": self.frame_index,
            "ts": round(self.timestamp_s, 3),
            "report": self.report,
            "backend": self.backend,
            "model": self.model,
            "tokens": {"in": self.input_tokens, "out": self.output_tokens},
        }


class FusionEngine:
    """
    Accepts a FusedPayload and calls the LLM to synthesize a report.
    Works with any backend: Ollama (local), Claude (cloud), or Stub.
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def synthesize(
        self,
        frame_index: int,
        timestamp_s: float,
        cv_data: dict | None,
        asr_data: dict | None,
        vsr_data: dict | None,
        frame_b64: str | None = None,
    ) -> FusionReport:
        prompt = self._build_prompt(cv_data, asr_data, vsr_data)

        try:
            resp: LLMResponse = self.llm.chat(
                prompt=prompt,
                system=FUSION_SYSTEM,
                image_b64=frame_b64,   # attach frame if model is vision-capable
                temperature=0.2,
            )
            log.debug(
                "Fusion | frame=%d backend=%s tokens_in=%d tokens_out=%d",
                frame_index, resp.backend, resp.input_tokens, resp.output_tokens,
            )
        except Exception as e:
            log.warning("LLM call failed (%s) — returning empty report", e)
            resp = LLMResponse(text=f"[fusion error: {e}]", model="error", backend="error")

        return FusionReport(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            report=resp.text,
            backend=resp.backend,
            model=resp.model,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
        )

    @staticmethod
    def _build_prompt(cv_data, asr_data, vsr_data) -> str:
        parts = []

        if cv_data:
            dets = cv_data.get("detections", [])
            scene = f"{len(dets)} object(s) detected"
            det_str = "; ".join(
                f"{d.get('class','?')} ({d.get('conf', 0):.2f})" for d in dets[:10]
            )
            parts.append(f"CV AGENT:\n  Scene: {scene}\n  Detections: {det_str or 'none'}\n  Latency: {cv_data.get('inference_ms', 0):.0f}ms")

        if asr_data:
            segs = asr_data.get("segments", [])
            transcript = " ".join(s.get("text", "") for s in segs)
            speakers = list({s.get("speaker_name") or s.get("speaker", "?") for s in segs})
            parts.append(f"ASR AGENT:\n  Language: {asr_data.get('language', 'unknown')}\n  Speakers: {', '.join(speakers)}\n  Transcript: {transcript[:500]}")

        if vsr_data:
            parts.append(
                f"VSR AGENT:\n  Mouth open: {vsr_data.get('mouth_open')}\n"
                f"  Lip ratio: {vsr_data.get('lip_aspect_ratio', 0):.3f}\n"
                f"  Text (LM rescored): {vsr_data.get('text', '') or 'none'}\n"
                f"  Language ID: {vsr_data.get('language_id') or 'n/a'}"
            )

        if not parts:
            return "No agent data available for this frame."

        return "\n\n".join(parts) + "\n\nSynthesize the above into a structured report."
