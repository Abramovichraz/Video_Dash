# avatar_intro

Turns one avatar image plus a voice recording into a short intro video. The avatar waves hello and lip-syncs to the voice.
It runs locally on CPU with no GPU and no paid API. A 10-second clip renders in about 40 seconds.

```bash
pip install -r requirements.txt
python make_intro.py --image avatar.png --audio my_voice.m4a --out intro.mp4
```

- `--arm right|left`: which of the person's arms waves (default: right).
- `--no-trim`: keep the silence at the start and end of the recording. By default it is trimmed.
- `--preview 0.5,1,2`: write PNG stills at those times instead of a video, for quick tuning.

The MediaPipe models (~50 MB) download to `models/` on the first run. On a headless Linux box MediaPipe also needs `libegl1` and `libgles2`.
Output is 1080x1350 (4:5) H.264 + AAC. The frame is widened with a soft background so the raised hand fits.

## How it works

1. **Detect**: MediaPipe face mesh (478 points), body pose, and a hair/skin/clothes segmentation.
2. **Arm cut-out**: the waving arm is cut into two pieces, upper arm and forearm + hand. The hole left behind is filled with mirrored jacket texture on the torso side and inpainted background elsewhere.
3. **Wave**: the upper arm rotates around the shoulder. The forearm comes up towards the camera: it shortens along the bone, flips, and rotates. Then it waves around the elbow.
4. **Lip-sync**: the voice is analysed per video frame for loudness, voicing and LPC formants F1/F2. These become jaw-open, lips-closed, rounded (o/u) and spread (i/s) values. A landmark-based warp moves the lower lip and jaw, and paints the inside of the mouth when it opens.
5. **Life**: blinks, a brow raise on the greeting, head sway and nods that follow the voice, breathing, and a slow push-in.
6. **Encode**: frames are piped to FFmpeg (bundled with `imageio-ffmpeg`). The voice is loudness-normalised and muxed.
