"""Building blocks for the documentation media: real dashboard screenshots, code and terminal
renders, captions and video encoding. Used by ``scripts/make_media.py``."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.styles import get_style_by_name

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "src" / "adaptive_scheduler" / "api" / "static" / "dashboard.html"
FONTS = Path("C:/Windows/Fonts")
W, H = 1920, 1080

# Palette shared with the dashboard and the result figures (dark theme).
BG = (18, 18, 17)
PANEL = (26, 26, 25)
INK = (240, 239, 236)
INK2 = (195, 194, 183)
INK3 = (143, 142, 134)
ACCENT = (57, 135, 229)
OK = (47, 191, 79)
FAIL = (230, 103, 103)
ORANGE = (235, 104, 52)
AQUA = (27, 175, 122)
YELLOW = (237, 161, 0)


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    candidates = {
        "sans": ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"],
        "bold": ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"],
        "light": ["segoeuil.ttf", "segoeui.ttf", "DejaVuSans.ttf"],
        "mono": ["consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"],
        "monob": ["consolab.ttf", "courbd.ttf", "DejaVuSansMono-Bold.ttf"],
    }[name]
    for c in candidates:
        for base in (FONTS, Path("/usr/share/fonts/truetype/dejavu")):
            if (base / c).exists():
                return ImageFont.truetype(str(base / c), size)
    return ImageFont.load_default(size)


# ---------------------------------------------------------------------------- browser screenshots
def browser() -> str:
    for p in (
        "C:/Program Files/Google/Chrome/Application/chrome.exe",
        "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
        "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    ):
        if Path(p).exists():
            return p
    for name in ("google-chrome", "chromium", "chromium-browser", "msedge"):
        if found := shutil.which(name):
            return found
    raise RuntimeError("Chrome or Edge is required to capture screenshots")


def screenshot(
    url: str,
    out: Path,
    size: tuple[int, int] = (W, H),
    budget_ms: int = 1500,
    scale: float = 1.0,
) -> Image.Image:
    """Capture ``url``; ``size`` is in CSS pixels and the image is ``size * scale`` pixels."""
    out = out.resolve()  # the browser resolves relative paths against its own working directory
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run(
            [browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
             f"--user-data-dir={profile}", f"--window-size={size[0]},{size[1]}",
             f"--force-device-scale-factor={scale}",
             f"--virtual-time-budget={budget_ms}", f"--screenshot={out}", url],
            check=True, capture_output=True, timeout=120,
        )  # fmt: skip
    return Image.open(out).convert("RGB")


def replay_page(demo: dict, out: Path) -> Path:
    """The real dashboard with a recorded run embedded, so frames can be rendered offline."""
    html = DASHBOARD.read_text(encoding="utf-8")
    inject = f"<script>window.__REPLAY__ = {json.dumps(demo)};</script>\n<script>"
    html = html.replace("<script>", inject, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def page_url(page: Path, **params: object) -> str:
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return page.resolve().as_uri() + (f"?{query}" if query else "")


# ---------------------------------------------------------------------------- windows & renders
def window(content: Image.Image, title: str, pad: int = 28) -> Image.Image:
    """Wrap an image in a desktop-style window frame with a title bar."""
    bar = 46
    w, h = content.width + 2 * pad, content.height + bar + 2 * pad
    img = Image.new("RGB", (w, h), (30, 30, 30))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, bar], fill=(44, 44, 44))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([18 + i * 26, 15, 34 + i * 26, 31], fill=c)
    d.text((w // 2, bar // 2), title, font=font("sans", 20), fill=INK2, anchor="mm")
    img.paste(content, (pad, bar + pad))
    return img


def code_render(path: Path, start: str, end: str | None = None, max_lines: int = 26,
                size: int = 24) -> Image.Image:  # fmt: skip
    """Render a real excerpt of a source file, syntax-highlighted, with true line numbers."""
    lines = path.read_text(encoding="utf-8").splitlines()
    i0 = next(i for i, line in enumerate(lines) if start in line)
    i1 = i0 + max_lines
    if end is not None:
        i1 = next((i for i in range(i0 + 1, len(lines)) if end in lines[i]), i1)
    i1 = min(i1, i0 + max_lines, len(lines))
    excerpt = "\n".join(lines[i0:i1])
    style = get_style_by_name("one-dark")
    f, fb = font("mono", size), font("monob", size)
    lh = int(size * 1.45)
    gutter = int(size * 3.2)
    width = int(max(f.getlength(line) for line in lines[i0:i1])) + gutter + 50
    img = Image.new("RGB", (max(width, 900), lh * (i1 - i0) + 30), (40, 44, 52))
    d = ImageDraw.Draw(img)
    x, y = gutter, 15
    for n in range(i1 - i0):
        d.text(
            (gutter - 20, 15 + n * lh), str(i0 + n + 1), font=f, fill=(99, 109, 131), anchor="ra"
        )
    for ttype, value in lex(excerpt, PythonLexer()):
        st = style.style_for_token(ttype)
        color = (
            tuple(int(st["color"][k : k + 2], 16) for k in (0, 2, 4))
            if st["color"]
            else (171, 178, 191)
        )
        for k, part in enumerate(value.split("\n")):
            if k:
                x, y = gutter, y + lh
            if part:
                d.text((x, y), part, font=fb if st["bold"] else f, fill=color)
                x += d.textlength(part, font=f)
    return img


def terminal_render(text: str, size: int = 22, cols: int | None = None) -> Image.Image:
    lines = text.rstrip("\n").splitlines()
    f = font("mono", size)
    lh = int(size * 1.4)
    cw = int(f.getlength("M"))
    width = (cols or max(len(line) for line in lines)) * cw + 40
    img = Image.new("RGB", (width, lh * len(lines) + 30), (12, 12, 12))
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        color = (120, 200, 255) if line.startswith("$") else (225, 225, 220)
        if line.startswith("#") or line.startswith("  #"):
            color = (140, 140, 135)
        d.text((20, 15 + i * lh), line, font=f, fill=color)
    return img


# ---------------------------------------------------------------------------- frame composition
def fit(img: Image.Image, box: tuple[int, int]) -> Image.Image:
    s = min(box[0] / img.width, box[1] / img.height, 1.0)
    return img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)


def canvas(color: tuple[int, int, int] = BG) -> Image.Image:
    return Image.new("RGB", (W, H), color)


def place(base: Image.Image, img: Image.Image, xy: tuple[int, int] | None = None,
          shadow: bool = True) -> Image.Image:  # fmt: skip
    if xy is None:
        xy = ((base.width - img.width) // 2, (base.height - img.height) // 2)
    if shadow:
        sh = Image.new("RGBA", (img.width + 60, img.height + 60), (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle(
            [30, 30, img.width + 30, img.height + 30], 14, fill=(0, 0, 0, 160)
        )
        sh = sh.filter(ImageFilter.GaussianBlur(18))
        base.paste(sh, (xy[0] - 30, xy[1] - 22), sh)
    base.paste(img, xy)
    return base


def caption(img: Image.Image, text: str, sub: str | None = None) -> Image.Image:
    """Bottom caption band: the video is understandable with the sound off."""
    out = img.copy()
    band_h = 150 if sub else 112
    band = Image.new("RGBA", (W, band_h), (8, 8, 8, 218))
    out.paste(band, (0, H - band_h), band)
    d = ImageDraw.Draw(out)
    d.text((W // 2, H - band_h + (48 if sub else band_h // 2)), text, font=font("bold", 42),
           fill=INK, anchor="mm")  # fmt: skip
    if sub:
        d.text((W // 2, H - 46), sub, font=font("sans", 30), fill=INK2, anchor="mm")
    return out


def text_block(d: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, f: ImageFont.FreeTypeFont,
               fill: tuple[int, int, int], width: int, spacing: float = 1.3) -> int:  # fmt: skip
    """Draw word-wrapped text; returns the y after the block."""
    x, y = xy
    words, line = text.split(), ""
    lh = int(f.size * spacing)
    for w_ in words:
        trial = (line + " " + w_).strip()
        if d.textlength(trial, font=f) > width and line:
            d.text((x, y), line, font=f, fill=fill)
            y += lh
            line = w_
        else:
            line = trial
    if line:
        d.text((x, y), line, font=f, fill=fill)
        y += lh
    return y


# ---------------------------------------------------------------------------- encoding
class Video:
    """Feed RGB frames straight into ffmpeg (H.264, yuv420p, faststart for web/LinkedIn)."""

    def __init__(
        self, out: Path, fps: int = 30, size: tuple[int, int] = (W, H), crf: int = 20
    ) -> None:
        import imageio_ffmpeg

        out.parent.mkdir(parents=True, exist_ok=True)
        self.fps, self.size, self.frames = fps, size, 0
        self.proc = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(out)],
            stdin=subprocess.PIPE,
        )  # fmt: skip
        self._last: Image.Image | None = None

    def write(self, img: Image.Image, seconds: float = 0.0, frames: int | None = None) -> None:
        img = img.convert("RGB").resize(self.size) if img.size != self.size else img.convert("RGB")
        n = frames if frames is not None else max(1, round(seconds * self.fps))
        data = img.tobytes()
        assert self.proc.stdin is not None
        for _ in range(n):
            self.proc.stdin.write(data)
        self.frames += n
        self._last = img

    def crossfade(self, nxt: Image.Image, seconds: float = 0.4) -> None:
        if self._last is None:
            return
        n = max(1, round(seconds * self.fps))
        nxt = nxt.convert("RGB")
        for i in range(1, n + 1):
            self.write(Image.blend(self._last, nxt, i / (n + 1)), frames=1)

    def scene(self, frames: Iterable[tuple[Image.Image, float]], fade: float = 0.4) -> None:
        first = True
        for img, seconds in frames:
            if first and fade:
                self.crossfade(img, fade)
            first = False
            self.write(img, seconds)

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def gif(mp4: Path, out: Path, width: int = 960, fps: int = 8) -> None:
    """High-quality GIF from an MP4 (two-pass palette) for inline display on GitHub."""
    import imageio_ffmpeg

    vf = f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(mp4),
                    "-vf", vf, str(out)], check=True)  # fmt: skip


def frames_between(a: float, b: float, n: int) -> Iterator[float]:
    for i in range(n):
        yield a + (b - a) * i / max(1, n - 1)
