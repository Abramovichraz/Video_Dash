import json
import shutil
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

import app as app_module
import pipeline

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


@pytest.fixture(scope="module")
def sample_clip(tmp_path_factory):
    """2s 1280x720 30fps clip with audio, like a Battlefield recording."""
    path = tmp_path_factory.mktemp("samples") / "sample.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(path),
    ], check=True, capture_output=True)
    return path


def probe_streams(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)["streams"]


def test_reframe_vertical(sample_clip, tmp_path):
    out = tmp_path / "out.mp4"
    pipeline.reframe_vertical(sample_clip, out)
    streams = probe_streams(out)
    video = next(s for s in streams if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)
    assert any(s["codec_type"] == "audio" for s in streams)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "JOBS_DIR", tmp_path / "jobs")
    with TestClient(app_module.app) as c:
        yield c


def test_upload_flow(client, sample_clip):
    with open(sample_clip, "rb") as f:
        res = client.post("/upload", files={"file": ("clip.mp4", f, "video/mp4")})
    assert res.status_code == 200
    job_id = res.json()["id"]

    deadline = time.time() + 60
    while time.time() < deadline:
        status = client.get(f"/jobs/{job_id}").json()
        if status["state"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert status["state"] == "done", status

    assert [j["id"] for j in client.get("/jobs").json()] == [job_id]
    out = client.get(f"/jobs/{job_id}/output")
    assert out.status_code == 200
    assert out.headers["content-type"] == "video/mp4"


def test_upload_rejects_bad_extension(client):
    res = client.post("/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400
