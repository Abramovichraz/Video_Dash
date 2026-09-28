#!/usr/bin/env python3
"""Turn a still avatar image + a voice recording into a short talking intro video.

The avatar waves hello while saying the recorded line. Everything runs locally on CPU:

- MediaPipe finds the face mesh, the body pose and a person/clothes/skin segmentation.
- The waving arm is cut out, the hole behind it is filled, and it is animated as two
  rigid pieces (upper arm around the shoulder, forearm + hand around the elbow).
- Lip-sync: the voice is analysed per video frame (loudness, voicing, LPC formants)
  and turned into jaw-open / lips-closed / rounded / spread values that drive a
  landmark-based warp of the mouth (with a painted mouth interior when it opens).
- Extra life: blinks, a small brow raise on the greeting, head sway and nods that
  follow the voice, breathing and a slow camera push-in.
- FFmpeg (bundled with imageio-ffmpeg) muxes the frames with the loudness-normalised voice.

Usage:
    python make_intro.py --image avatar.png --audio my_voice.m4a --out intro.mp4
"""
import argparse
import subprocess
import urllib.request
from pathlib import Path

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

FPS = 30
SCALE = 2  # render at 2x the source resolution so sub-pixel mouth motion stays smooth
LEAD = 0.6  # seconds before the voice starts (the arm is already going up)
TAIL = 0.8
MODEL_DIR = Path(__file__).parent / "models"
MODEL_URLS = {
    "face_landmarker.task": "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_heavy.task": "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    "selfie_multiclass_256x256.tflite": "image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
}
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# Face-mesh landmark indices.
LIP_UPPER_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LIP_LOWER_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
EYES = [  # (upper lid, lower lid)
    ([33, 246, 161, 160, 159, 158, 157, 173, 133], [33, 7, 163, 144, 145, 153, 154, 155, 133]),
    ([362, 398, 384, 385, 386, 387, 388, 466, 263], [362, 382, 381, 380, 374, 373, 390, 249, 263]),
]
BROWS = [105, 334]


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def curve(idx, lm):
    """Landmark polyline as (xs, ys) sorted by x, for np.interp."""
    pts = lm[idx]
    order = np.argsort(pts[:, 0])
    return pts[order, 0], pts[order, 1]


def rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


# ---------------------------------------------------------------- detection

def ensure_models():
    MODEL_DIR.mkdir(exist_ok=True)
    for name, url in MODEL_URLS.items():
        path = MODEL_DIR / name
        if not path.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/" + url, path)


def detect(img):
    h, w = img.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    opts = lambda name: BaseOptions(model_asset_path=str(MODEL_DIR / name))
    with vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=opts("face_landmarker.task"))) as f:
        faces = f.detect(image).face_landmarks
    with vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=opts("pose_landmarker_heavy.task"))) as p:
        poses = p.detect(image).pose_landmarks
    with vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=opts("selfie_multiclass_256x256.tflite"), output_category_mask=True)) as s:
        cats = np.squeeze(s.segment(image).category_mask.numpy_view()).astype(np.uint8)
    if not faces or not poses:
        raise SystemExit("could not find a face and a body in the image")
    face = np.array([[l.x * w, l.y * h] for l in faces[0]])
    pose = np.array([[l.x * w, l.y * h] for l in poses[0]])
    return face, pose, cats  # cats: 0 bg, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 other


# ---------------------------------------------------------------- arm cut-out

def seg_dist(xx, yy, a, b):
    """Distance to segment a-b, and the unclipped position along it (0 at a, 1 at b)."""
    ab = b - a
    t_raw = ((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / (ab @ ab)
    t = np.clip(t_raw, 0, 1)
    return np.hypot(xx - (a[0] + t * ab[0]), yy - (a[1] + t * ab[1])), t_raw


def limb_radius(person, a, b, side):
    """Distance from the bone midpoint to the outer edge of the sleeve."""
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([-d[1], d[0]])
    if np.sign(n[0]) != side:
        n = -n
    mid = (a + b) / 2
    h, w = person.shape
    for k in range(1, 400):
        x, y = (mid + n * k).astype(int)
        if not (0 <= x < w and 0 <= y < h) or not person[y, x]:
            return max(k, 20)
    return 60


def cut_arm(img, pose, cats, arm):
    """Return the image with the arm removed (hole filled) and masks for the two arm pieces."""
    h, w = img.shape[:2]
    s_i, e_i, w_i, hip_i = (12, 14, 16, 24) if arm == "right" else (11, 13, 15, 23)
    S, E, W = pose[s_i], pose[e_i], pose[w_i]
    center_x = (pose[11, 0] + pose[12, 0]) / 2
    side = np.sign(S[0] - center_x)  # -1: arm is on the image's left
    person = cats > 0
    r_u = limb_radius(person, S, E, side) * 1.05
    r_f = limb_radius(person, E, W, side) * 1.05

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d_up = (E - S) / np.linalg.norm(E - S)
    above_cap = ((xx - S[0]) * d_up[0] + (yy - S[1]) * d_up[1]) < -0.45 * r_u
    upper = (seg_dist(xx, yy, S, E)[0] < r_u) & person & ~above_cap
    d_f, t_f = seg_dist(xx, yy, E, W)
    fore = (d_f < r_f) & (t_f < 1.0) & person
    skin = cv2.morphologyEx((cats == 2).astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n_lab, labels = cv2.connectedComponents(skin)
    near_wrist = np.hypot(xx - W[0], yy - W[1]) < 2.2 * r_f
    for lab in range(1, n_lab):  # the hand (body-skin blob touching the wrist area)
        blob = labels == lab
        if (blob & near_wrist).any():
            fore |= blob
    upper, fore = [cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0) for m in (upper, fore)]

    # Fill the hole: jacket (mirrored from the torso next to it, pixelated like the
    # source) on the torso side of a line from armpit to hip, background elsewhere.
    hole = cv2.dilate(((upper > 0.1) | (fore > 0.1)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    max_c = np.where(hole, xx, 0).max(1)[:, None]
    min_c = np.where(hole, xx, w - 1).min(1)[:, None]
    inner = max_c if side < 0 else min_c
    torso_src = np.clip(2 * inner - xx - side * 2, 0, w - 1).astype(np.float32)
    torso_fill = cv2.remap(img, torso_src, yy, cv2.INTER_NEAREST)
    f = 8
    small = cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA)
    unknown = cv2.resize(cv2.dilate(person.astype(np.uint8), np.ones((25, 25), np.uint8)), (w // f, h // f),
                         interpolation=cv2.INTER_NEAREST)
    bg_fill = cv2.resize(cv2.inpaint(small, unknown, 6, cv2.INPAINT_TELEA), (w, h), interpolation=cv2.INTER_CUBIC)
    bg_fill = cv2.GaussianBlur(bg_fill, (0, 0), 4) + np.random.default_rng(0).normal(0, 2.5, img.shape)

    armpit = S + np.array([-side * 0.55 * r_u, 1.1 * r_u])
    hip_edge = np.array([W[0] - side * 1.1 * r_f, W[1]])
    t = np.clip((yy - armpit[1]) / (hip_edge[1] - armpit[1]), 0, 1)
    line_x = armpit[0] + t * (hip_edge[0] - armpit[0])
    torso_side = (side * (xx - line_x) < 0) & (yy > S[1] - 0.3 * r_u)
    block = 8
    small = cv2.resize(torso_fill, (w // block, h // block), interpolation=cv2.INTER_AREA)
    torso_fill = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    a_t = cv2.GaussianBlur(torso_side.astype(np.float32), (0, 0), 1.5)[..., None]
    fill = torso_fill * a_t + bg_fill * (1 - a_t)
    a_h = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 2)[..., None]
    plate = (img * (1 - a_h) + fill * a_h).astype(np.uint8)
    return plate, upper, fore, (S, E, W), side


# ---------------------------------------------------------------- canvas

def to_canvas(img, ext, blur_ext=False):
    big = cv2.copyMakeBorder(img, 0, 0, ext, ext, cv2.BORDER_REFLECT)
    if blur_ext:  # soft, slightly darker background extension so the frame is 4:5
        soft = cv2.GaussianBlur(big, (0, 0), 30) * 0.85
        w = big.shape[1]
        x = np.arange(w)
        m = np.maximum(smoothstep(ext + 30, ext, x), smoothstep(w - ext - 30, w - ext, x))[None, :, None]
        big = (big * (1 - m) + soft * m).astype(np.uint8)
    return cv2.resize(big, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------- audio -> visemes

def decode_audio(path, sr, start=None, end=None):
    cmd = [FFMPEG, "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}", "-to", f"{end:.3f}"]
    cmd += ["-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, check=True, capture_output=True).stdout, np.float32)


def speech_bounds(path):
    """Start/end (seconds) of the voice, trimming silence around it."""
    sr = 16000
    x = decode_audio(path, sr)
    hop = sr // 100
    rms = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2) + 1e-10) for i in range(0, len(x) - hop, hop)])
    db = 20 * np.log10(rms)
    floor, peak = np.percentile(db, 10), np.percentile(db, 99)
    active = np.where(db > floor + 0.35 * (peak - floor))[0]  # above the room noise
    if len(active) == 0:
        return 0.0, len(x) / sr
    return max(active[0] / 100 - 0.12, 0), min(active[-1] / 100 + 0.25, len(x) / sr)


def levinson(r, order):
    a = np.zeros(order + 1)
    a[0], e = 1.0, r[0]
    for i in range(1, order + 1):
        k = -(r[i] + a[1:i] @ r[i - 1:0:-1]) / e
        a[1:i] = a[1:i] + k * a[i - 1:0:-1]
        a[i] = k
        e *= 1 - k * k
    return a


def formants(frame, sr, order=12):
    x = frame * np.hamming(len(frame))
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    r = np.correlate(x, x, "full")[len(x) - 1:len(x) + order]
    if r[0] <= 1e-9:
        return 0.0, 0.0
    roots = np.roots(levinson(r, order))
    roots = roots[np.imag(roots) > 0.01]
    f = np.angle(roots) * sr / (2 * np.pi)
    bw = -sr / np.pi * np.log(np.abs(roots))
    f = np.sort(f[(f > 200) & (bw < 500)])
    return (f[0] if len(f) > 0 else 0.0), (f[1] if len(f) > 1 else 0.0)


def follow(x, attack, release):
    y, out = 0.0, np.zeros_like(x)
    for i, v in enumerate(x):
        y += (attack if v > y else release) * (v - y)
        out[i] = y
    return out


def visemes(audio, n_frames):
    """Per video frame: v (-1 lips closed .. 0 rest smile .. 1 jaw open), round, spread, loudness."""
    sr = 10000
    hop, win = sr / FPS, int(0.03 * sr)
    lead = int(LEAD * FPS)
    loud_db, zcr, f1, f2 = (np.zeros(n_frames) for _ in range(4))
    for i in range(n_frames):
        c = int((i - lead) * hop)
        seg = audio[max(c - win // 2, 0):max(c + win // 2, 0)]
        if len(seg) < win // 2:
            loud_db[i] = -100
            continue
        loud_db[i] = 10 * np.log10(np.mean(seg ** 2) + 1e-10)
        zcr[i] = np.mean(np.abs(np.diff(np.sign(seg)))) / 2
        f1[i], f2[i] = formants(seg, sr)
    speaking_db = loud_db[loud_db > -99]
    peak, floor = np.percentile(speaking_db, 97), np.percentile(speaking_db, 30)
    loud = np.clip((loud_db - floor) / (peak - floor + 1e-6), 0, 1)
    voiced = (loud > 0.12) & (zcr < 0.28)
    hiss = (loud > 0.12) & ~voiced
    speech = cv2.dilate((loud > 0.08).astype(np.uint8)[None], np.ones((1, 9), np.uint8))[0] > 0

    openness = loud ** 0.7 * np.clip(0.55 + (f1 - 350) / 600, 0.55, 1.0)  # louder + higher F1 = wider jaw
    v = np.where(voiced, -0.5 + 1.6 * openness, -0.8)  # quiet frames inside speech: lips meet
    v = np.where(hiss, -0.35, v)  # s / sh / ts: teeth almost together
    v = np.where(speech, v, 0.0)  # silence: back to the resting smile
    rnd = np.where(voiced, np.clip((1350 - f2) / 500, 0, 1), 0)
    spr = np.where(voiced, np.clip((f2 - 1900) / 500, 0, 1), 0) + 0.4 * hiss
    v = np.clip(follow(v, 0.65, 0.45), -0.95, 1.0)
    rnd, spr = follow(rnd, 0.4, 0.3), follow(spr, 0.4, 0.3)
    shift = lambda a: np.append(a[1:], a[-1])  # mouth shapes lead the sound slightly
    return shift(v), shift(rnd), shift(spr), follow(loud, 0.25, 0.08)


# ---------------------------------------------------------------- face warp

class Face:
    def __init__(self, lm, shape):
        self.lm = lm
        self.upper = curve(LIP_UPPER_INNER, lm)
        self.lower = curve(LIP_LOWER_INNER, lm)
        self.corner_l, self.corner_r = lm[78], lm[308]
        self.mouth_c = (self.corner_l + self.corner_r) / 2
        self.mouth_hw = np.linalg.norm(self.corner_r - self.corner_l) / 2
        self.chin = lm[152]
        self.face_h = self.chin[1] - lm[10][1]
        self.jaw_hw = np.linalg.norm(lm[397] - lm[172]) / 2
        self.pivot = self.chin + np.array([0, 0.35 * self.face_h])  # base of the neck
        xs, ys = lm[:, 0], lm[:, 1]
        fw = xs.max() - xs.min()
        h, w = shape
        self.x0, self.x1 = int(max(xs.min() - 0.5 * fw, 0)), int(min(xs.max() + 0.5 * fw, w))
        self.y0, self.y1 = int(max(ys.min() - 0.4 * self.face_h, 0)), int(min(self.pivot[1] + 0.2 * self.face_h, h))
        yy, xx = np.mgrid[self.y0:self.y1, self.x0:self.x1].astype(np.float32)
        self.grid = xx, yy
        # how much of the head motion each pixel follows (1 on the head, 0 below the neck)
        self.head_w = (smoothstep(self.pivot[1], self.chin[1] + 0.1 * self.face_h, yy)
                       * smoothstep(0.9 * fw, 0.6 * fw, np.abs(xx - lm[1][0])))

    def warp(self, plate, head_deg, head_shift, v, rnd, spr, blink, brow):
        L = self.lm
        xx, yy = self.grid
        # 1) head: rotate about the neck base, weighted so the collar stays put
        a = np.deg2rad(head_deg) * self.head_w
        px, py = xx - head_shift[0] * self.head_w - self.pivot[0], yy - head_shift[1] * self.head_w - self.pivot[1]
        qx = self.pivot[0] + np.cos(a) * px + np.sin(a) * py
        qy = self.pivot[1] - np.sin(a) * px + np.cos(a) * py
        sx, sy = qx.copy(), qy.copy()

        # 2) brows lift
        for b in BROWS:
            wgt = np.exp(-(((qx - L[b][0]) / (0.22 * self.face_h)) ** 2 + ((qy - L[b][1]) / (0.1 * self.face_h)) ** 2))
            sy += brow * wgt

        # 3) blinks: pull the upper lid down over the eye
        if blink > 0.01:
            for up_idx, lo_idx in EYES:
                ux, uy = curve(up_idx, L)
                lx, ly = curve(lo_idx, L)
                top, bot = np.interp(qx, ux, uy), np.interp(qx, lx, ly)
                hgt = np.maximum(bot - top, 0.5)
                inside = (qx > ux[0]) & (qx < ux[-1])
                lens = np.clip(1 - ((qx - (ux[0] + ux[-1]) / 2) / ((ux[-1] - ux[0]) / 2 + 1e-6)) ** 2, 0, 1) ** 0.5
                b = blink * lens
                lid_end = top + b * hgt
                lid = inside & (qy >= top) & (qy < lid_end)
                eye = inside & (qy >= lid_end) & (qy <= bot + 0.5)
                band = 0.7 * hgt
                sy = np.where(lid, top - band * (1 - (qy - top) / np.maximum(b * hgt, 1e-3)), sy)
                sy = np.where(eye, top + (qy - lid_end) * hgt / np.maximum(hgt - b * hgt, 1e-3), sy)

        # 4) mouth
        mid_x = self.mouth_c[0]
        ui = np.interp(qx, *self.upper)
        li = np.interp(qx, *self.lower)
        gap0 = np.maximum(li - ui, 0)
        lens = np.clip(1 - ((qx - mid_x) / (self.mouth_hw * 1.05)) ** 2, 0, 1) ** 0.6
        jaw = smoothstep(self.jaw_hw * 1.25, self.jaw_hw * 0.55, np.abs(qx - mid_x))
        below_chin = smoothstep(self.chin[1] + 0.3 * self.face_h, self.chin[1], qy)
        interior = np.zeros_like(qx)
        if v > 0:  # open: lower lip + jaw drop, mouth interior shows between the teeth
            o = v * 0.3 * self.mouth_hw * 2
            m = (ui + li) / 2
            shape = lens + (jaw - lens) * smoothstep(li, li + 0.25 * self.face_h * 0.3, qy)
            shift = o * shape * below_chin
            gap_end = m + o * lens
            interior = np.clip(np.minimum(qy - m, gap_end - qy) * SCALE * 0.7 + 0.5, 0, 1) * (lens > 0.02)
            sy = np.where(qy > m, sy - shift, sy)
        elif v < 0:  # close: squeeze the teeth band until the lips meet
            c = -v * gap0 * np.maximum(lens, 0.0)
            band = (qy >= ui) & (qy < li - c)
            sy = np.where(band, ui + (qy - ui) * gap0 / np.maximum(gap0 - c, 1e-3), sy)
            cj = -v * gap0.max() * jaw * below_chin
            sy = np.where(qy >= li - c, sy + np.where(qy < li, c, cj), sy)
        # rounded (o/u) lips pull the corners in, spread (i/s) pushes them out
        bump = np.exp(-(((qx - mid_x) / (1.5 * self.mouth_hw)) ** 2 + ((qy - self.mouth_c[1]) / (0.9 * self.mouth_hw)) ** 2))
        sx = sx + (qx - mid_x) * (0.28 * rnd - 0.1 * spr) * bump

        out = cv2.remap(plate, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT)
        if v > 0:  # paint the inside of the mouth: dark, a bit lighter towards the bottom (tongue)
            rel = np.clip((qy - (ui + li) / 2) / (0.3 * self.mouth_hw * 2 + 1e-3), 0, 1)
            col = np.stack([30 + 20 * rel, 22 + 15 * rel, 45 + 55 * rel], -1)
            a = interior[..., None]
            out = (out * (1 - a) + col * a).astype(np.uint8)
        frame = plate.copy()
        frame[self.y0:self.y1, self.x0:self.x1] = out
        return frame


# ---------------------------------------------------------------- arm animation

def arm_pose(t, side):
    """(upper arm target weight, forearm wave angle) at time t."""
    up = smoothstep(0.0, 0.55, t) * (1 - smoothstep(2.3, 2.9, t))
    env = smoothstep(0.45, 0.7, t) * (1 - smoothstep(2.0, 2.4, t))
    wave = 13 * np.sin(2 * np.pi * 2.3 * (t - 0.55)) * env
    return up, wave


def paste(frame, piece, A, b):
    """Warp an arm piece with x -> A x + b and composite it, touching only its bounding box."""
    color, alpha, (x0, y0, x1, y1) = piece
    h, w = frame.shape[:2]
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]]) @ A.T + b
    X0, Y0 = np.maximum(np.floor(corners.min(0)).astype(int) - 2, 0)
    X1, Y1 = np.minimum(np.ceil(corners.max(0)).astype(int) + 2, [w, h])
    if X1 <= X0 or Y1 <= Y0:
        return frame
    M = np.hstack([A, (b - [X0, Y0])[:, None]])
    c = cv2.warpAffine(color, M, (X1 - X0, Y1 - Y0), flags=cv2.INTER_LINEAR)
    a = cv2.warpAffine(alpha, M, (X1 - X0, Y1 - Y0), flags=cv2.INTER_LINEAR)[..., None]
    roi = frame[Y0:Y1, X0:X1].astype(np.float32)
    frame[Y0:Y1, X0:X1] = np.clip(roi * (1 - a) + c, 0, 255).astype(np.uint8)
    return frame


def angle(v):
    return np.degrees(np.arctan2(v[1], v[0]))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="avatar image (one person, full or upper body)")
    ap.add_argument("--audio", required=True, help="voice recording (any format FFmpeg reads)")
    ap.add_argument("--out", default="intro.mp4")
    ap.add_argument("--arm", choices=["right", "left"], default="right", help="which of the person's arms waves")
    ap.add_argument("--no-trim", action="store_true", help="keep silence at the start/end of the recording")
    ap.add_argument("--preview", help="comma-separated times (s): write PNG stills instead of a video")
    args = ap.parse_args()

    ensure_models()
    src = cv2.imread(args.image)
    if src is None:
        raise SystemExit(f"cannot read {args.image}")
    face_lm, pose, cats = detect(src)
    plate, upper_m, fore_m, (S, E, W), side = cut_arm(src, pose, cats, args.arm)

    h0, w0 = src.shape[:2]
    ext = max(int(round((h0 * 0.8 - w0) / 2)), 0)  # widen to 4:5 so the raised hand fits
    to_c = lambda p: (np.asarray(p) + [ext, 0]) * SCALE
    plate_c = to_canvas(plate, ext, blur_ext=True)
    src_c = to_canvas(src, ext).astype(np.float32)
    pieces = []
    for m in (upper_m, fore_m):
        a = to_canvas((m * 255).astype(np.uint8), ext).astype(np.float32) / 255
        if ext:
            a[:, :ext * SCALE] = 0
            a[:, -ext * SCALE:] = 0
        ys, xs = np.nonzero(a > 0.002)
        pieces.append(((src_c * a[..., None]).astype(np.float32), a, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)))
    S, E, W = to_c(S), to_c(E), to_c(W)
    face = Face(to_c(face_lm), plate_c.shape[:2])
    H, Wd = plate_c.shape[:2]

    # target arm directions: upper arm out and down, forearm up (slightly inwards)
    up_target = np.array([side * np.sin(np.radians(34)), np.cos(np.radians(34))])
    fore_target = np.array([-side * np.sin(np.radians(6)), -np.cos(np.radians(6))])
    d_up = (angle(up_target) - angle(E - S) + 180) % 360 - 180
    # The forearm comes up towards the camera: in 2D it shortens along the bone, flips,
    # and grows again pointing up (reflection along the bone + a small extra rotation).
    d0 = (W - E) / np.linalg.norm(W - E)
    d_fore = (angle(fore_target) - angle(-d0) + 180) % 360 - 180

    start, end = (None, None) if args.no_trim else speech_bounds(args.audio)
    audio = decode_audio(args.audio, 10000, start, end)
    dur = LEAD + len(audio) / 10000 + TAIL
    n = int(dur * FPS)
    v, rnd, spr, loud = visemes(audio, n)

    rng = np.random.default_rng(7)
    blinks, t_b = [], 1.9
    while t_b < dur:
        blinks.append(t_b)
        t_b += rng.uniform(2.2, 3.4)

    def render(i):
        t = i / FPS
        blink = max([np.interp(t - b, [0, 0.07, 0.18], [0, 1, 0], left=0, right=0) for b in blinks] + [0])
        brow = SCALE * 3.0 * smoothstep(0.3, 0.6, t) * (1 - smoothstep(1.3, 1.9, t))
        head_deg = 1.3 * np.sin(2 * np.pi * 0.21 * t + 0.5) + 1.2 * (loud[i] - 0.4) * side * -1
        head_shift = SCALE * np.array([1.5 * np.sin(2 * np.pi * 0.13 * t), -2.5 * loud[i] + np.sin(2 * np.pi * 0.37 * t)])
        frame = face.warp(plate_c, head_deg, head_shift, v[i], rnd[i], spr[i], blink, brow)

        up, wave = arm_pose(t, side)
        R_u = rot(d_up * up)
        k = np.cos(np.pi * up)
        k = np.copysign(max(abs(k), 0.12), k)
        R_f = rot(d_fore * up + wave * side) @ (np.eye(2) + (k - 1) * np.outer(d0, d0))
        frame = paste(frame, pieces[0], R_u, S - R_u @ S)
        E_new = S + R_u @ (E - S)
        frame = paste(frame, pieces[1], R_f, E_new - R_f @ E)

        # camera: breathing + slow push-in towards the face
        zoom = 1 + 0.035 * smoothstep(0, dur, t) + 0.004 * np.sin(2 * np.pi * t / 3.6)
        cx, cy = face.pivot[0], face.pivot[1] * 0.9
        M = np.array([[zoom, 0, cx * (1 - zoom)], [0, zoom, cy * (1 - zoom)]])
        return cv2.warpAffine(frame, M, (Wd, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    if args.preview:
        for t in args.preview.split(","):
            i = min(int(float(t) * FPS), n - 1)
            path = Path(args.out).with_suffix("").as_posix() + f"_t{float(t):.2f}.png"
            cv2.imwrite(path, render(i))
            print("wrote", path, f"v={v[i]:.2f}")
        return

    audio_in = ["-i", str(args.audio)] if start is None else ["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(args.audio)]
    cmd = [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{Wd}x{H}", "-r", str(FPS), "-i", "-",
           *audio_in, "-filter_complex",
           f"[0:v]scale=1080:-2:flags=lanczos[v];[1:a]loudnorm=I=-16:TP=-1.5:LRA=11,adelay=delays={int(LEAD * 1000)}:all=1,apad=whole_dur={dur:.3f}[a]",
           "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", args.out]
    print(" ".join(cmd))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        proc.stdin.write(render(i).tobytes())
        if i % FPS == 0:
            print(f"frame {i}/{n}", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
