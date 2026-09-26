# reel-pipeline

Turn raw gameplay recordings into vertical (9:16) highlight reels for Instagram Reels and YouTube Shorts.
The user uploads a video in a local web dashboard, the pipeline finds the kills, cuts them, reframes to vertical, and returns a ready-to-post clip.

**Current goal: a working skeleton, end to end. Not features.** Upload → process → download must work before anything else is added.

## Context

- Game: Battlefield, recorded at 1280x720, 30fps (other resolutions may come later).
- Player name in the kill feed: `Dragonox9976` (shown in **yellow**, top-right kill feed).
- Ground truth for testing: `samples/clip1.mp4` has two kills by the player, at **~0:20** and **~0:26**.
- Runs locally on the user's machine. Single user. No auth, no cloud.

## Stack (keep it this small)

- Python 3.11+
- FastAPI + uvicorn: API and serving the dashboard
- Dashboard: ONE static `index.html` with vanilla JS. No React, no build step, no npm.
- FFmpeg via `subprocess` (must be on PATH). No moviepy.
- Kill detection (phase 2): OpenCV for frame sampling + EasyOCR for reading the kill feed.
- Storage: plain folders on disk. No database.
- Background work: `threading.Thread` per job. No Celery, no Redis.

## Project layout

```
app.py            # FastAPI: routes + job runner
pipeline.py       # detect / cut / reframe / export, plain functions
static/index.html # the dashboard
jobs/<job_id>/    # input.mp4, status.json, output.mp4
samples/          # test videos (gitignored except clip1.mp4)
test_pipeline.py  # one small test file
```

Do not add files beyond this without a clear reason.

## Job model

Each job is a folder `jobs/<job_id>/` with `status.json`:

```json
{ "id": "...", "state": "queued|processing|done|error", "step": "detect", "kills": [20.1, 26.0], "error": null }
```

## API

- `POST /upload`: multipart video upload → creates job, starts thread, returns `{id}`
- `GET /jobs`: list all jobs (read status.json files)
- `GET /jobs/{id}`: one job status
- `GET /jobs/{id}/output`: the finished mp4
- `GET /`: serves `static/index.html`

## Dashboard (MVP)

One page, three things:
1. Upload box (file input + button).
2. Jobs list, polling `/jobs` every 2 seconds, showing state and current step.
3. When done: an inline `<video>` player and a download link.

Plain CSS, dark theme, mobile-friendly. That is all.

## Build phases (do them in order, stop after each and let me test)

1. **Skeleton**: upload → FFmpeg converts the whole video to 9:16 (center crop, 1080x1920) → download. No detection yet.
2. **Detection**: sample ~2 fps, crop the kill-feed region, OCR, find timestamps where `Dragonox9976` appears as the killer. Test must find kills near 20s and 26s in `clip1.mp4` (±1.5s).
3. **Cut**: pad each kill (3s before, 2s after), merge kills closer than 8s into one segment, concat.
4. **Style**: slow-motion around the kill moment, zoom punch, "1/2" "2/2" kill counter via FFmpeg `drawtext`. Styles defined as small dicts in `pipeline.py` first. Move to YAML only when there are 3+ styles.
5. Later (do not build yet): music beat-sync (librosa), captions (faster-whisper), AI-chosen style via Claude API, watch-folder for ShadowPlay recordings.

## Conventions

- Simplest thing that works. No abstractions with one implementation, no config for values that never change.
- Every FFmpeg command is built as a Python list and logged before running.
- Errors in a job are caught and written to `status.json`. They must never crash the server.
- Validate uploads: video extensions only, reasonable size limit.
- Code, comments, and commit messages in English.
- Run with: `uvicorn app:app --reload`, then open http://localhost:8000
- Test with: `python -m pytest -q`

## Workflow rules

- Start each phase in plan mode and show the plan before writing code.
- After each phase: run the test, run the server, tell me exactly what to click to verify.
- Do not start the next phase until I confirm the current one works.
