"""FastAPI REST server — submit videos and stream results via SSE."""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.orchestration.workflow_runtime import run_video_workflow
from src.utils.common import get_logger, load_config

log = get_logger(__name__)
app = FastAPI(title="Multimodal Video Intelligence API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

cfg = load_config()
JOBS: dict[str, dict] = {}

# Thread pool — serialised (max_workers=1) so cached model instances are never
# accessed concurrently by two jobs; models are loaded once and reused.
_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)


# ─── Schemas ─────────────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    status: str         # queued | running | done | error
    result: dict | None = None
    error: str | None = None


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/analyze", response_model=JobStatus)
async def analyze_video(
    file: UploadFile = File(...),
    llm_backend: str = Form("auto"),
    ollama_model: str = Form("llama3.2:latest"),
    text_model: str = Form(""),   # text-only model for ASR summary + synthesis; falls back to ollama_model
    agents: str = Form("cv,asr,vsr"),  # comma-separated subset of cv,asr,vsr
    asr_model: str = Form("base"),
):
    """Upload a video file and start async analysis."""
    suffix = Path(file.filename).suffix.lower()
    allowed = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported format: {suffix}. Allowed: {allowed}")

    job_id = str(uuid.uuid4())
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    content = await file.read()
    tmp.write(content)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    enabled_agents = {a.strip().lower() for a in agents.split(",") if a.strip()}
    JOBS[job_id] = {"status": "queued", "path": tmp_path, "llm_backend": llm_backend, "ollama_model": ollama_model, "text_model": text_model, "agents": agents, "asr_model": asr_model}
    asyncio.create_task(_run_job(job_id, tmp_path, llm_backend=llm_backend, ollama_model=ollama_model, text_model=text_model, enabled_agents=enabled_agents, asr_model=asr_model))
    log.info("Job %s queued for %s (llm=%s vision=%s text=%s agents=%s asr_model=%s)", job_id, file.filename, llm_backend, ollama_model, text_model, agents, asr_model)
    return JobStatus(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    job = JOBS[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
        error=job.get("error"),
    )


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    """Server-Sent Events stream for live job progress."""
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")

    async def event_gen():
        while True:
            job = JOBS.get(job_id, {})
            data = json.dumps({"status": job.get("status"), "result": job.get("result")})
            yield f"data: {data}\n\n"
            if job.get("status") in ("done", "error"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/jobs")
async def list_jobs():
    return [{"job_id": k, "status": v["status"]} for k, v in JOBS.items()]


# ─── Background task ─────────────────────────────────────────────────────────

async def _run_job(job_id: str, video_path: str, *, llm_backend: str = "auto", ollama_model: str = "llama3.2:latest", text_model: str = "", enabled_agents: set[str] | None = None, asr_model: str | None = None) -> None:
    JOBS[job_id]["status"] = "running"
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _THREAD_POOL,
            lambda: run_video_workflow(
                video_path,
                llm_backend=llm_backend,
                llm_ollama_model=ollama_model,
                llm_text_model=text_model or ollama_model,
                enabled_agents=enabled_agents,
                asr_model=asr_model,
            ),
        )
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = result
        log.info("Job %s done", job_id)
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)
        log.error("Job %s failed: %s", job_id, e)
    finally:
        try:
            os.unlink(video_path)
        except Exception:
            pass


# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
