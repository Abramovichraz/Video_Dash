"""Video processing steps. Plain functions, FFmpeg via subprocess."""

import logging
import shlex
import subprocess
from pathlib import Path

log = logging.getLogger("pipeline")


def run_ffmpeg(cmd: list[str]) -> None:
    log.info("running: %s", shlex.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-20:])
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}):\n{tail}")


def reframe_vertical(src: Path, dst: Path) -> None:
    """Center-crop to 9:16 and scale to 1080x1920."""
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        # Even crop width keeps libx264 happy.
        "-vf", "crop=trunc(ih*9/16/2)*2:ih,scale=1080:1920,setsar=1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(dst),
    ]
    run_ffmpeg(cmd)


def process(job_dir: Path, input_path: Path, set_step) -> Path:
    """Run all pipeline steps for one job. Returns the output path."""
    output = job_dir / "output.mp4"
    set_step("reframe")
    reframe_vertical(input_path, output)
    return output
