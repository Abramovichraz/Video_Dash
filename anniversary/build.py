"""Build standalone.html: index.html with every photo and the song embedded, so it works as a single file."""
import base64
import io
import re
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).parent


def data_uri(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((1100, 1100))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


html = (HERE / "index.html").read_text(encoding="utf-8")
names = sorted(set(re.findall(r"photos/[\w.-]+\.(?:jpe?g|png|webp)", html, re.I)))
for name in names:
    html = html.replace(name, data_uri(HERE / name))
    print("embedded", name)

m = re.search(r"audioFile: '([^']+)'", html)
if m:
    audio = (HERE / m.group(1)).read_bytes()
    html = html.replace(m.group(0), "audioFile: 'data:audio/mpeg;base64," + base64.b64encode(audio).decode() + "'")
    print("embedded", m.group(1))

out = HERE / "standalone.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out.name}: {out.stat().st_size / 1e6:.1f} MB, {len(names)} photos")
