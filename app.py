"""FastAPI app: upload, job runner, status, download, dashboard."""

import json
import logging
import os
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse

import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("app")

JOBS_DIR = Path("jobs")
STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
MAX_UPLOAD_BYTES = 4 * 1024**3
CHUNK_SIZE = 1024**2


def read_status(job_id: str) -> dict | None:
    path = JOBS_DIR / job_id / "status.json"
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_status(status: dict) -> None:
    path = JOBS_DIR / status["id"] / "status.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2))
    # On Windows the replace fails while the poller has the file open; retry briefly.
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05)


def update_status(job_id: str, **changes) -> None:
    status = read_status(job_id)
    status.update(changes)
    write_status(status)


def find_input(job_id: str) -> Path:
    return next((JOBS_DIR / job_id).glob("input.*"))


def run_job(job_id: str) -> None:
    try:
        update_status(job_id, state="processing")
        pipeline.process(
            JOBS_DIR / job_id,
            find_input(job_id),
            set_step=lambda step: update_status(job_id, step=step),
        )
        update_status(job_id, state="done", step=None)
    except Exception as e:
        log.exception("job %s failed", job_id)
        update_status(job_id, state="error", error=str(e))


def mark_interrupted_jobs() -> None:
    JOBS_DIR.mkdir(exist_ok=True)
    for status_file in JOBS_DIR.glob("*/status.json"):
        status = read_status(status_file.parent.name)
        if status and status["state"] in ("queued", "processing"):
            status.update(state="error", error="interrupted by server restart")
            write_status(status)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mark_interrupted_jobs()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/upload")
def upload(file: UploadFile) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    job_id = uuid.uuid4().hex[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    size = 0
    with open(job_dir / f"input{ext}", "wb") as out:
        while chunk := file.file.read(CHUNK_SIZE):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                shutil.rmtree(job_dir)
                raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES // 1024**3} GB)")
            out.write(chunk)

    write_status({
        "id": job_id,
        "state": "queued",
        "step": None,
        "kills": [],
        "error": None,
        "filename": file.filename,
        "created": time.time(),
    })
    threading.Thread(target=run_job, args=(job_id,), daemon=True).start()
    return {"id": job_id}


@app.get("/jobs")
def list_jobs() -> list[dict]:
    jobs = [read_status(p.parent.name) for p in JOBS_DIR.glob("*/status.json")]
    return sorted((j for j in jobs if j), key=lambda j: j["created"], reverse=True)


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    status = read_status(job_id)
    if status is None:
        raise HTTPException(404, "Job not found")
    return status


@app.get("/jobs/{job_id}/output")
def get_output(job_id: str) -> FileResponse:
    status = read_status(job_id)
    if status is None or status["state"] != "done":
        raise HTTPException(404, "Output not ready")
    name = Path(status["filename"]).stem + "_vertical.mp4"
    return FileResponse(JOBS_DIR / job_id / "output.mp4", media_type="video/mp4", filename=name)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
