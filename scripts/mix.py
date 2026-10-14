#!/usr/bin/env python3
"""Lay sound under a cut film: each shot's effects on its cut, the score beneath.

    mix.py --spec S --film film.mp4 --sfx DIR --score score.flac --out film-sound.mp4

Effects are found as <item>.flac in DIR (what sfx.py writes) and placed at the
shot's offset in the cut, which is the title card plus every shot before it, the
same arithmetic assemble.py cuts by. The score runs under the whole film and is
side-chain ducked under the effects, so a sound in the picture is never buried by
the music.

Levels are set the way a mix engineer sets them, with static gains, never a
dynamic normaliser: every effect is measured and brought to SFX_LUFS (MMAudio does
not normalise; one film's twelve effects came out between -3.6 and -36 LUFS), the
score to SCORE_LUFS, and the finished mix is measured once more and brought to
-14 LUFS integrated under a -1.5 dBTP limiter.

A missing effect is allowed (that shot plays on score alone) and reported, because
an effect that failed to render should not hold the whole film back.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forge_motion as fm  # noqa: E402
from assemble import ffmpeg  # noqa: E402

CARD_SECONDS = fm.CARD_SECONDS
SFX_LUFS = -24.0      # every shot's effects, before ducking
SCORE_LUFS = -20.0    # the bed: a few dB over the effects, ducked under them when they speak
TARGET_LUFS = -14.0
CEILING_DBTP = -2.0  # AAC overshoots the limiter by ~0.4 dB; this lands near -1.5 after encoding
PEAK_ROOM_DB = -3.0   # no single effect is pushed past this true peak by its gain


def loudness(path: Path) -> tuple[float, float]:
    """(integrated LUFS, true peak dBTP) of a file, measured by ffmpeg's EBU R128 meter."""
    err = subprocess.run([ffmpeg(), "-nostats", "-hide_banner", "-i", str(path),
                          "-af", "ebur128=peak=true", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    summary = err[err.rfind("Summary:"):]
    i = float(re.search(r"I:\s+(-?[\d.]+|-inf) LUFS", summary).group(1).replace("-inf", "-70"))
    tp = float(re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", summary).group(1).replace("-inf", "-70"))
    return i, tp


def gain_to(target: float, measured: tuple[float, float], peak_room: float | None = None) -> float:
    """Static gain in dB that brings `measured` to `target`, held under `peak_room` dBTP."""
    i, tp = measured
    g = target - i
    if peak_room is not None:
        g = min(g, peak_room - tp)
    return round(g, 2)


def offsets(spec_path: Path):
    """[(item, start_s, duration_s)] for every shot, in cut order."""
    clips = fm.resolve_clips(fm.load_spec(spec_path), root=spec_path.resolve().parents[2])
    t, out = CARD_SECONDS, []
    for c in clips:
        d = c.frames / c.fps
        out.append((c.item, t, d))
        t += d
    return out


def mix(spec_path: Path, film: Path, sfx_dir: Path, score: Path | None, out: Path) -> dict:
    shots = offsets(spec_path)
    total = shots[-1][1] + shots[-1][2] + CARD_SECONDS  # through the end card
    inputs, chains, labels, report = [], [], [], {"missing": [], "gains_db": {}}
    idx = 0
    for item, start, dur in shots:
        f = sfx_dir / f"{item}.flac"
        if not f.is_file():
            report["missing"].append(item)
            continue
        g = gain_to(SFX_LUFS, loudness(f), PEAK_ROOM_DB)
        report["gains_db"][item] = g
        inputs += ["-i", str(f)]
        ms = int(round(start * 1000))
        chains.append(
            f"[{idx}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{dur:.3f},"
            f"volume={g}dB,afade=t=in:d=0.03,afade=t=out:st={max(0, dur - 0.08):.3f}:d=0.08,"
            f"adelay={ms}|{ms}[s{idx}]")
        labels.append(f"[s{idx}]")
        idx += 1
    graph = list(chains)
    if labels:
        # Padded to the whole film: sidechaincompress ends with its shortest input, and
        # the effects end with the last shot, which cut the end card off the film.
        graph.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0,"
                     f"apad=whole_dur={total:.3f},asplit=2[fx][key]")
    if score is not None:
        g = gain_to(SCORE_LUFS, loudness(score))
        report["gains_db"]["score"] = g
        inputs += ["-i", str(score)]
        graph.append(f"[{idx}:a]aresample=48000,aformat=channel_layouts=stereo,volume={g}dB,"
                     f"atrim=0:{total:.3f},afade=t=out:st={total - 2.5:.3f}:d=2.5[mus0]")
        if labels:
            graph.append("[mus0][key]sidechaincompress=threshold=0.03:ratio=4:attack=20:release=500[mus]")
            graph.append("[mus][fx]amix=inputs=2:normalize=0[pre]")
        else:
            graph.append("[mus0]anull[pre]")
    elif labels:
        graph.append("[key]anullsink;[fx]anull[pre]")
    else:
        raise SystemExit("nothing to mix: no effects and no score")

    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "pre.wav"
        subprocess.run([ffmpeg(), "-v", "error", "-y", *inputs, "-filter_complex", ";".join(graph),
                        "-map", "[pre]", "-t", f"{total:.3f}", "-c:a", "pcm_f32le", str(pre)], check=True)
        g = gain_to(TARGET_LUFS, loudness(pre))
        report["gains_db"]["master"] = g
        out.parent.mkdir(parents=True, exist_ok=True)
        ceiling = 10 ** (CEILING_DBTP / 20)
        subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", str(film), "-i", str(pre),
                        "-filter_complex", f"[1:a]volume={g}dB,alimiter=limit={ceiling:.4f}:attack=5:release=80:level=false[a]",
                        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                        "-ar", "48000", "-shortest", "-movflags", "+faststart", str(out)], check=True)
    report["final"] = dict(zip(("lufs", "dbtp"), loudness(out)))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mix", description="lay effects and score under a cut film")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--film", required=True)
    ap.add_argument("--sfx", required=True)
    ap.add_argument("--score")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    report = mix(Path(a.spec), Path(a.film), Path(a.sfx), Path(a.score) if a.score else None, Path(a.out))
    if report["missing"]:
        print(f"warning: no effects for {', '.join(report['missing'])}; those shots play on the score alone")
    print(f"gains (dB): {report['gains_db']}")
    print(f"final: {report['final']['lufs']} LUFS, {report['final']['dbtp']} dBTP")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
