"""Render "Sent to Kill Him" - a 15s pixel-art horror-romance reel (9:16).

Wolf boy x vampire girl. Everything (art + sound) is generated procedurally.
Run: python render_reel.py  ->  writes reel.mp4 next to this file.
Needs: pillow, numpy, imageio-ffmpeg.
"""
import math
import random
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
W, H = 180, 320            # pixel canvas, upscaled x6 -> 1080x1920
SCALE = 6
FPS = 24
DUR = 15.0
N = int(DUR * FPS)
SR = 44100
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 24)

PAL = {
    "K": (24, 8, 34), "D": (150, 10, 30), "R": (255, 30, 50), "W": (232, 220, 230),
    "H": (40, 14, 56), "B": (92, 52, 28), "S": (236, 188, 150), "E": (40, 30, 30),
    "M": (180, 90, 90), "F": (160, 30, 36), "f": (60, 12, 16), "J": (40, 60, 110),
    "G": (120, 124, 140), "L": (200, 204, 214), "Y": (255, 220, 40), "P": (255, 90, 140),
}

VAMP = [
    "....HHHH....",
    "...HHHHHH...",
    "..HHWWWWHH..",
    "..HWRWWRWH..",
    "..HWWWWWWH..",
    "..HHWWDWHH..",
    "..HH.WW.HH..",
    ".KKKKDDKKKK.",
    "KKKKKDDKKKKK",
    "KDKKKKKKKKDK",
    "KDKKKKKKKKDK",
    "KDKKKKKKKKDK",
    "KDKKKKKKKKDK",
    ".WKKKKKKKKW.",
    "..KKKKKKKK..",
    "..KKKKKKKK..",
    "..KKKKKKKK..",
    ".KKKKKKKKKK.",
    ".KKKKKKKKKK.",
    "KKKKKKKKKKKK",
    "KDDDDDDDDDDK",
    "....K..K....",
    "...KK..KK...",
]
BOY = [
    "...BBBBB....",
    "..BBBBBBB...",
    "..BSSSSSB...",
    "..SESSESS...",
    "..SSSSSSS...",
    "...SSMSS....",
    "....SSS.....",
    "..FFFFFFF...",
    ".FFfFFfFFF..",
    ".FFfFFfFFF..",
    ".SFfFFfFFS..",
    ".SFFFFFFFS..",
    "..FFFFFFF...",
    "..JJJJJJJ...",
    "..JJJ.JJJ...",
    "..JJ...JJ...",
    "..JJ...JJ...",
    "..JJ...JJ...",
    "..JJ...JJ...",
    ".KKK...KKK..",
]
WOLF = [
    "..G.....G...........",
    "..GG...GG...........",
    "..GGGGGGG...........",
    ".GGYGGGYGG..........",
    ".GGGGGGGGG..........",
    ".GGGGLLGGGG.........",
    "..GGLKKLGGG.........",
    "...GLLLLGGGG........",
    "...GWWWWGGGGG.......",
    "...GGGGGGGGGGG......",
    "..GGGGGGGGGGGGG.....",
    "..GGGGGGGGGGGGGG....",
    "..GGGLGGGGGGGGGG..G.",
    "..GGLLGGGGGGGGGGGGG.",
    "..GGLLGGGGGGGGGGGG..",
    "..GGGLGGGGGGGGGGG...",
    "..GGGLGGGGGGGGGG....",
    "..GG.GG.GGGG.GG.....",
    "..GG.GG.GGGG.GG.....",
    ".GGG.GGG.GGG.GGG....",
]
HEART = [".PP.PP.", "PPPPPPP", "PPPPPPP", ".PPPPP.", "..PPP..", "...P..."]


def sprite(rows, px=3, flip=False, tint=None, glow=None):
    """Char-map -> RGBA image, each char = px*px block."""
    h, w = len(rows), len(rows[0])
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for y, row in enumerate(rows):
        for x, c in enumerate(row):
            if c != ".":
                col = glow or PAL[c]
                if tint:
                    col = tuple(int(a * (1 - tint[1]) + b * tint[1]) for a, b in zip(col, tint[0]))
                im.putpixel((x, y), col + (255,))
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return im.resize((w * px, h * px), Image.NEAREST)


rnd = random.Random(7)
STARS = [(rnd.randrange(W), rnd.randrange(0, 170), rnd.random()) for _ in range(70)]
TREES = [(x, rnd.randrange(150, 200), rnd.randrange(8, 14)) for x in range(-10, W + 20, 14)]
EMBERS = [[rnd.randrange(W), rnd.randrange(H), rnd.uniform(0.3, 1.0)] for _ in range(40)]


def lerp(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def draw_sky(img, t, dawn=0.0):
    d = ImageDraw.Draw(img)
    top, bot = (8, 6, 28), (58, 18, 64)
    top, bot = lerp(top, (60, 40, 90), dawn), lerp(bot, (255, 120, 60), dawn)
    for y in range(H):
        d.line([(0, y), (W, y)], fill=lerp(top, bot, y / H))
    for x, y, p in STARS:
        if math.sin(t * 3 + p * 20) > -0.3 and dawn < 0.6:
            d.point((x, y), fill=(220, 220, 255))


def draw_moon(img, t, reveal):
    d = ImageDraw.Draw(img)
    cx, cy, r = 124, 70, 26
    glow = int(40 * reveal)
    for k in range(4, 0, -1):
        d.ellipse([cx - r - k * 5, cy - r - k * 5, cx + r + k * 5, cy + r + k * 5],
                  fill=(60 + glow, 20 + glow // 2, 70 + glow))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(250, 236, 200))
    d.ellipse([cx - 12, cy - 8, cx - 4, cy], fill=(226, 206, 170))
    d.ellipse([cx + 6, cy + 6, cx + 14, cy + 14], fill=(226, 206, 170))
    # clouds slide off as reveal -> 1
    off = int(reveal * 110)
    for i, (dx, dy, w) in enumerate([(-40, -10, 70), (-20, 8, 80), (-50, 22, 60)]):
        x0 = cx + dx - off * (1 if i % 2 else -1)
        d.rounded_rectangle([x0, cy + dy - 8, x0 + w, cy + dy + 8], 8, fill=(30, 18, 50))


def draw_ground(img, dawn=0.0):
    d = ImageDraw.Draw(img)
    tree = lerp((14, 6, 22), (40, 20, 30), dawn)
    for x, top, w in TREES:
        d.polygon([(x, top), (x - w, 250), (x + w, 250)], fill=tree)
        d.rectangle([x - 1, 245, x + 1, 256], fill=tree)
    d.rectangle([0, 250, W, H], fill=lerp((18, 10, 26), (50, 30, 34), dawn))
    for x in range(0, W, 3):
        d.line([(x, 250), (x + 1, 246 - (x * 7) % 4)], fill=lerp((30, 16, 40), (70, 50, 40), dawn))


def draw_embers(img, t, color=(255, 60, 80)):
    d = ImageDraw.Draw(img)
    for e in EMBERS:
        y = (e[1] - t * 20 * e[2]) % H
        x = e[0] + math.sin(t * 2 + e[1]) * 3
        if int(t * 10 + e[1]) % 3:
            d.point((int(x), int(y)), fill=color)


def vignette():
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    return np.clip(1.15 - 0.55 * r ** 2, 0.25, 1.0)[..., None]


VIG = vignette()
CAP = []  # caption layers for the current frame, composited after upscale


def caption(text, color=(255, 255, 255), y=62, shake=0):
    """Chunky caption: aliasing-free text at 2x pixel canvas, black outline."""
    layer = Image.new("RGBA", (W * 2, H * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.fontmode = "1"
    lines = text.split("\n")
    for i, line in enumerate(lines):
        tw = d.textlength(line, font=FONT)
        x = (W * 2 - tw) / 2 + shake
        yy = y * 2 + i * 30
        for ox in (-2, -1, 0, 1, 2):
            for oy in (-2, -1, 0, 1, 2):
                d.text((x + ox, yy + oy), line, font=FONT, fill=(0, 0, 0, 255))
        d.text((x, yy), line, font=FONT, fill=color + (255,))
    return layer


# --------------------------------------------------------------- scenes
def scene_hook(t):
    """0-2s: extreme close-up. Fangs descend to a neck. Red flash. Pattern interrupt."""
    img = Image.new("RGB", (W, H), (10, 4, 16))
    d = ImageDraw.Draw(img)
    # neck + jaw (his)
    d.polygon([(40, 320), (60, 150), (120, 130), (150, 320)], fill=(220, 172, 136))
    d.polygon([(60, 150), (120, 130), (130, 100), (50, 110)], fill=(200, 150, 118))
    d.polygon([(110, 60), (180, 40), (180, 260), (140, 200), (126, 120)], fill=(92, 52, 28))  # his hair
    d.ellipse([112, 110, 130, 140], fill=(200, 150, 118))                                     # ear
    d.polygon([(60, 150), (120, 130), (118, 150), (70, 172)], fill=(170, 120, 96))            # jaw shadow
    d.line([(82, 170), (96, 240)], fill=(160, 90, 110), width=2)  # vein
    d.rectangle([30, 280, 160, 320], fill=(160, 30, 36))          # flannel collar
    # her mouth, descending from top
    k = ease(t / 1.1)
    my = int(-40 + 170 * k)
    d.rectangle([20, my - 70, 160, my], fill=(232, 220, 230))
    d.polygon([(40, my - 6), (140, my - 6), (120, my + 6), (60, my + 6)], fill=(150, 10, 30))
    for fx in (68, 106):
        d.polygon([(fx, my), (fx + 8, my), (fx + 4, my + 16)], fill=(255, 255, 255))
    if t > 1.15:  # the bite + blood
        drip = int((t - 1.15) * 60)
        d.rectangle([90, 178, 93, 178 + drip], fill=(200, 0, 30))
        d.rectangle([89, 178 + drip, 94, 182 + drip], fill=(200, 0, 30))
    img = img.convert("RGBA")
    shake = int(math.sin(t * 60) * 3) if 1.1 < t < 1.5 else 0
    CAP.append(caption("SHE WAS SENT\nTO KILL HIM.", (255, 40, 60), y=196, shake=shake))
    flash = 1.1 < t < 1.25
    return img.convert("RGB"), flash, shake


def scene_meet(t):
    """2-5s: moonlit forest. She stalks him. He turns."""
    img = Image.new("RGB", (W, H))
    draw_sky(img, t)
    draw_moon(img, t, 0.0)
    draw_ground(img)
    img = img.convert("RGBA")
    vx = int(10 + 30 * ease((t - 2) / 2))
    img.alpha_composite(sprite(VAMP), (vx, 250 - 69))
    img.alpha_composite(sprite(BOY, flip=(t > 3.6)), (116, 250 - 60))
    draw_embers(img, t)
    msg = "BUT NOBODY TOLD HER..." if t < 3.6 else "...WHAT HE WAS."
    CAP.append(caption(msg, y=62))
    return img.convert("RGB"), False, 0


def scene_turn(t):
    """5-8s: clouds part, full moon, he transforms. Screen shake."""
    img = Image.new("RGB", (W, H))
    draw_sky(img, t)
    reveal = ease((t - 5) / 1.0)
    draw_moon(img, t, reveal)
    draw_ground(img)
    img = img.convert("RGBA")
    img.alpha_composite(sprite(VAMP, tint=((255, 255, 255), 0.0)), (40, 250 - 69))
    phase = t - 6.0
    if phase < 0:
        img.alpha_composite(sprite(BOY, flip=True), (116, 250 - 60))
    elif phase < 0.8:
        glow = (255, 230, 90)
        s = sprite(BOY if int(phase * 16) % 2 else WOLF, px=3 + int(phase * 2), flip=True, glow=glow)
        img.alpha_composite(s, (150 - s.width // 2 - 10, 250 - s.height))
    else:
        s = sprite(WOLF, flip=True)
        img.alpha_composite(s, (100, 250 - s.height))
    draw_embers(img, t, (255, 220, 80) if phase > 0 else (255, 60, 80))
    shake = int(math.sin(t * 70) * 4) if 0 <= phase < 1.0 else 0
    if phase >= 0.8:
        CAP.append(caption("A WEREWOLF.", (255, 220, 40), y=62, shake=shake))
    flash = 0 <= phase < 0.08
    return img.convert("RGB"), flash, shake


def scene_stay(t):
    """8-12s: she lowers her fangs. He lowers his head. Hearts."""
    img = Image.new("RGB", (W, H))
    draw_sky(img, t)
    draw_moon(img, t, 1.0)
    draw_ground(img)
    img = img.convert("RGBA")
    img.alpha_composite(sprite(VAMP), (56, 250 - 69))
    s = sprite(WOLF, flip=True)
    wx = int(100 - 14 * ease((t - 9.2) / 1.2))
    img.alpha_composite(s, (wx, 250 - s.height))
    if t > 10.2:
        for i in range(3):
            ht = (t - 10.2 - i * 0.35)
            if ht > 0:
                hy = int(170 - ht * 30)
                img.alpha_composite(sprite(HEART, px=2), (84 + int(math.sin(ht * 4 + i) * 8), hy))
    msg = "SHE COULD HAVE RUN." if t < 10 else "SHE STAYED."
    CAP.append(caption(msg, y=62))
    return img.convert("RGB"), False, 0


def scene_dawn(t):
    """12-15s: sunrise (deadly to her). He becomes her shadow. Cliffhanger -> loop."""
    dawn = ease((t - 12) / 1.5)
    img = Image.new("RGB", (W, H))
    draw_sky(img, t, dawn)
    d = ImageDraw.Draw(img)
    d.ellipse([60, 232 - int(20 * dawn), 120, 292 - int(20 * dawn)], fill=(255, 200, 90))
    draw_ground(img, dawn)
    img = img.convert("RGBA")
    img.alpha_composite(sprite(VAMP, tint=((20, 10, 30), 0.5)), (56, 250 - 69))
    s = sprite(WOLF, flip=True)
    img.alpha_composite(s, (70, 250 - s.height))   # wolf stands between her and the sun
    if t < 13.8:
        CAP.append(caption("DAWN KILLS VAMPIRES.", y=62))
    else:
        CAP.append(caption("SO HE BECAME\nHER SHADOW.", (255, 200, 90), y=56))
    # last 0.4s: cut to black + red eyes -> matches the hook's first frame for a loop
    if t > 14.6:
        img = Image.new("RGBA", (W, H), (10, 4, 16, 255))
        d = ImageDraw.Draw(img)
        d.rectangle([72, 150, 78, 153], fill=(255, 30, 50))
        d.rectangle([102, 150, 108, 153], fill=(255, 30, 50))
    return img.convert("RGB"), False, 0


def frame(i):
    t = i / FPS
    CAP.clear()
    if t < 2:
        img, flash, shake = scene_hook(t)
    elif t < 5:
        img, flash, shake = scene_meet(t)
    elif t < 8:
        img, flash, shake = scene_turn(t)
    elif t < 12:
        img, flash, shake = scene_stay(t)
    else:
        img, flash, shake = scene_dawn(t)
    a = np.asarray(img).astype(np.float32) * VIG
    if flash:
        a = a * 0.4 + np.array([255, 20, 40]) * 0.6
    a = a.clip(0, 255).astype(np.uint8)
    if shake:
        a = np.roll(a, shake, axis=1)
    big = Image.fromarray(a).resize((W * SCALE, H * SCALE), Image.NEAREST).convert("RGBA")
    for c in CAP:
        big.alpha_composite(c.resize((W * SCALE, H * SCALE), Image.NEAREST))
    big = big.convert("RGB")
    arr = np.asarray(big).copy()
    arr[::SCALE] = (arr[::SCALE] * 0.82).astype(np.uint8)  # faint CRT scanlines
    return arr


# --------------------------------------------------------------- audio
def audio():
    n = int(DUR * SR)
    tt = np.arange(n) / SR
    out = np.zeros(n)
    # dread drone
    out += 0.10 * np.sin(2 * np.pi * 55 * tt) + 0.06 * np.sin(2 * np.pi * 55.7 * tt)

    def hit(at, dur, f0, f1, amp, noise=0.0):
        s, e = int(at * SR), int(min(DUR, at + dur) * SR)
        x = np.arange(e - s) / SR
        env = np.exp(-x * 6 / dur)
        f = f0 + (f1 - f0) * x / dur
        sig = np.sin(2 * np.pi * np.cumsum(f) / SR) + noise * np.random.uniform(-1, 1, len(x))
        out[s:e] += amp * env * sig

    for b in (0.0, 0.3, 0.75, 1.0):           # heartbeat under the hook
        hit(b, 0.18, 70, 40, 0.6)
    hit(1.12, 0.9, 900, 120, 0.35, noise=0.8)  # the bite sting
    hit(6.0, 1.2, 60, 30, 0.7, noise=0.5)      # transformation boom
    # howl: rising, vibrato
    s, e = int(6.3 * SR), int(7.9 * SR)
    x = np.arange(e - s) / SR
    f = 380 + 260 * np.sin(np.pi * x / 1.6) + 12 * np.sin(2 * np.pi * 6 * x)
    out[s:e] += 0.22 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * x / 1.6)
    # music-box romance (A minor)
    notes = [440, 523, 659, 587, 523, 494, 440, 330, 440, 523, 659, 784]
    for k, fr in enumerate(notes):
        at = 8.2 + k * 0.33
        if at < 14.4:
            hit(at, 0.6, fr, fr, 0.14)
            hit(at, 0.6, fr * 2, fr * 2, 0.04)
    hit(14.6, 0.4, 1200, 200, 0.3, noise=0.6)  # final sting on the red eyes
    out = np.tanh(out * 1.3) * 0.8
    pcm = (out * 32767).astype(np.int16)
    path = HERE / "reel.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


def main():
    wav = audio()
    out = HERE / "reel.mp4"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W * SCALE}x{H * SCALE}", "-r", str(FPS), "-i", "-", "-i", str(wav),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
           "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out)]
    print(" ".join(cmd))
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for i in range(N):
        p.stdin.write(frame(i).tobytes())
    p.stdin.close()
    p.wait()
    wav.unlink()
    print("wrote", out)


if __name__ == "__main__":
    main()
