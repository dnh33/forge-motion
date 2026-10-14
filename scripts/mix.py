#!/usr/bin/env python3
"""Lay sound under a cut film: each shot's effects on its cut, the score beneath.

    mix.py --spec S --film film.mp4 --sfx DIR --score score.flac --out film-sound.mp4

Effects are found as <item>.flac in DIR (what sfx.py writes) and placed at the
shot's offset in the cut, which is the title card plus every shot before it, the
same arithmetic assemble.py cuts by. The score runs under the whole film and is
side-chain ducked under the effects, so a sound in the picture is never buried by
the music. The mix is normalised to -14 LUFS integrated, -1.5 dBTP.

A missing effect is allowed (that shot plays on score alone) and reported, because
an effect that failed to render should not hold the whole film back.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forge_motion as fm  # noqa: E402
from assemble import ffmpeg  # noqa: E402

CARD_SECONDS = fm.CARD_SECONDS

import subprocess  # noqa: E402

SFX_GAIN = 1.0
SCORE_GAIN = 0.55


def offsets(spec_path: Path):
    """[(item, start_s, duration_s)] for every shot, in cut order."""
    clips = fm.resolve_clips(fm.load_spec(spec_path), root=spec_path.resolve().parents[2])
    t, out = CARD_SECONDS, []
    for c in clips:
        d = c.frames / c.fps
        out.append((c.item, t, d))
        t += d
    return out


def mix(spec_path: Path, film: Path, sfx_dir: Path, score: Path | None, out: Path) -> list:
    shots = offsets(spec_path)
    total = shots[-1][1] + shots[-1][2] + CARD_SECONDS  # through the end card
    inputs, chains, labels, missing = ["-i", str(film)], [], [], []
    idx = 1
    for item, start, dur in shots:
        f = sfx_dir / f"{item}.flac"
        if not f.is_file():
            missing.append(item)
            continue
        inputs += ["-i", str(f)]
        ms = int(round(start * 1000))
        chains.append(
            f"[{idx}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{dur:.3f},"
            f"afade=t=in:d=0.03,afade=t=out:st={max(0, dur - 0.08):.3f}:d=0.08,"
            f"adelay={ms}|{ms},volume={SFX_GAIN}[s{idx}]")
        labels.append(f"[s{idx}]")
        idx += 1
    graph = []
    if labels:
        # Padded to the whole film: sidechaincompress ends with its shortest input, and
        # the effects end with the last shot, which cut the end card off the film.
        graph += chains + ["".join(labels) + f"amix=inputs={len(labels)}:normalize=0,"
                           f"apad=whole_dur={total:.3f},asplit=2[fx][key]"]
    if score is not None:
        inputs += ["-i", str(score)]
        graph.append(f"[{idx}:a]aresample=48000,aformat=channel_layouts=stereo,volume={SCORE_GAIN}[mus0]")
        if labels:
            graph.append("[mus0][key]sidechaincompress=threshold=0.04:ratio=5:attack=15:release=450[mus]")
            graph.append("[mus][fx]amix=inputs=2:normalize=0[pre]")
        else:
            graph.append("[mus0]anull[pre]")
    elif labels:
        graph.append("[key]anullsink;[fx]anull[pre]")
    else:
        raise SystemExit("nothing to mix: no effects and no score")
    graph.append("[pre]loudnorm=I=-14:TP=-1.5:LRA=11[a]")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg(), "-v", "error", "-y", *inputs, "-filter_complex", ";".join(graph),
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-ar", "48000", "-shortest", "-movflags", "+faststart", str(out)], check=True)
    return missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mix", description="lay effects and score under a cut film")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--film", required=True)
    ap.add_argument("--sfx", required=True)
    ap.add_argument("--score")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    missing = mix(Path(a.spec), Path(a.film), Path(a.sfx), Path(a.score) if a.score else None, Path(a.out))
    if missing:
        print(f"warning: no effects for {', '.join(missing)}; those shots play on the score alone")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
