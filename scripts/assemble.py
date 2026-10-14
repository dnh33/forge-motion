#!/usr/bin/env python3
"""Cut rendered shots into a film: title card, shots in spec order, end card.

    assemble.py --spec prompts/films/one-small-step.json --shots DIR --out film.mp4

Shots are found as <item>-s<seed>.mp4 in DIR (what forge_motion.py render writes).
Cuts between shots are hard, as in most films; the picture fades up from black
after the title and down to black before the end card. A missing shot is an error,
not a gap: a film with a hole in it is not the film that was written.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forge_motion as fm  # noqa: E402

CARD_SECONDS = 3.5
FADE_SECONDS = 0.75
FONTS = [  # first that loads wins: runner (DejaVu, Liberation), then Windows
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "C:/Windows/Fonts/georgia.ttf",
]


def font(size: int):
    for path in FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def spaced(text: str) -> str:
    """Letterspace a title the way a title card does."""
    return " ".join(text)


def card(size, lines) -> Image.Image:
    """A black card with centred lines; each line is (text, px, grey)."""
    w, h = size
    img = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    rows = [(t, font(px), g) for t, px, g in lines if t]
    heights = [draw.textbbox((0, 0), t, font=f)[3] for t, f, _ in rows]
    gap = int(h * 0.05)
    y = (h - sum(heights) - gap * (len(rows) - 1)) // 2
    for (t, f, g), lh in zip(rows, heights):
        tw = draw.textlength(t, font=f)
        draw.text(((w - tw) / 2, y), t, font=f, fill=(g, g, g))
        y += lh + gap
    return img


def ffmpeg() -> str:
    """imageio's bundled binary (installed with the render stack), else PATH."""
    try:
        import imageio_ffmpeg  # noqa: PLC0415
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def run(*args):
    subprocess.run([ffmpeg(), "-v", "error", "-y", *args], check=True)


def card_clip(img: Image.Image, out: Path, fps: int):
    png = out.with_suffix(".png")
    img.save(png)
    fade_out = CARD_SECONDS - FADE_SECONDS
    run("-loop", "1", "-i", str(png), "-t", str(CARD_SECONDS), "-r", str(fps),
        "-vf", f"fade=in:st=0:d={FADE_SECONDS},fade=out:st={fade_out}:d={FADE_SECONDS},format=yuv420p",
        "-c:v", "libx264", "-crf", "16", str(out))


def assemble(spec_path: Path, shots_dir: Path, out: Path) -> Path:
    spec = fm.load_spec(spec_path)
    clips = fm.resolve_clips(spec, root=spec_path.resolve().parents[2])
    film = spec.get("film", {})
    size, fps = clips[0].size, clips[0].fps
    missing = [c.out_name for c in clips if not (shots_dir / f"{c.out_name}.mp4").is_file()]
    if missing:
        raise SystemExit(f"missing shots in {shots_dir}: {', '.join(missing)}")

    h = size[1]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        title = tmp / "title.mp4"
        end = tmp / "end.mp4"
        card_clip(card(size, [(spaced(film.get("title", spec.get("name", ""))), int(h * 0.11), 235),
                              (film.get("subtitle", ""), int(h * 0.045), 150)]), title, fps)
        card_clip(card(size, [(spaced(film.get("end", "THE END")), int(h * 0.08), 235),
                              (film.get("credits", ""), int(h * 0.035), 140)]), end, fps)

        parts = [title, *[shots_dir / f"{c.out_name}.mp4" for c in clips], end]
        inputs = [a for p in parts for a in ("-i", str(p))]
        n = len(parts)
        first, last = 1, n - 2
        last_len = clips[-1].frames / fps
        chains = []
        for i in range(n):
            f = f"[{i}:v]scale={size[0]}:{size[1]},setsar=1,fps={fps},format=yuv420p"
            if i == first:
                f += f",fade=in:st=0:d={FADE_SECONDS}"
            if i == last:
                f += f",fade=out:st={last_len - FADE_SECONDS:.3f}:d={FADE_SECONDS}"
            chains.append(f + f"[v{i}]")
        graph = ";".join(chains) + ";" + "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[out]"
        out.parent.mkdir(parents=True, exist_ok=True)
        run(*inputs, "-filter_complex", graph, "-map", "[out]",
            "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-movflags", "+faststart", str(out))

    meta = {"title": film.get("title"), "shots": [c.out_name for c in clips],
            "seconds": round(sum(c.frames / c.fps for c in clips) + 2 * CARD_SECONDS, 2)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="assemble", description="cut shots into a film")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--shots", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    print(f"wrote {assemble(Path(a.spec), Path(a.shots), Path(a.out))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
