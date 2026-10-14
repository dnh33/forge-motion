#!/usr/bin/env python3
"""A score from a written brief: ACE-Step 1.5 turbo, text to music, instrumental.

    music.py --acestep DIR --caption "..." --duration S --out score.flac [--bpm N --key "D minor"]

ACE-Step 1.5 (ace-step/ACE-Step-1.5, MIT) runs here as its DiT alone, with no
5 Hz language model: the brief is already written, so there is nothing for the LM
to plan, and leaving it out drops its weights and its vLLM dependency. That is the
configuration ACE-Step itself recommends for small machines. Turbo settings per its
INFERENCE.md: 8 steps, shift 3.0, guidance handled by distillation.

Writes <out> plus <out>.json with timings and peak memory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def peak_rss_gb():
    try:
        import resource  # noqa: PLC0415
    except ImportError:
        return None
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="music")
    ap.add_argument("--acestep", required=True, help="clone of ace-step/ACE-Step-1.5")
    ap.add_argument("--caption", required=True)
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bpm", type=int)
    ap.add_argument("--key", default="")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=8)
    a = ap.parse_args(argv)

    out = Path(a.out).resolve()
    root = str(Path(a.acestep).resolve())
    sys.path.insert(0, root)
    import torch  # noqa: PLC0415
    from acestep.handler import AceStepHandler  # noqa: PLC0415
    from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: PLC0415

    torch.set_num_threads(os.cpu_count() or 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = {"device": device, "steps": a.steps, "model": "acestep-v15-turbo (DiT only)"}
    t0 = time.perf_counter()
    dit = AceStepHandler()
    status, ok = dit.initialize_service(project_root=root, config_path="acestep-v15-turbo", device=device)
    if not ok:
        print(f"error: ACE-Step init failed: {status}", file=sys.stderr)
        return 1
    metrics["load_s"] = round(time.perf_counter() - t0, 1)

    t = time.perf_counter()
    params = GenerationParams(
        caption=a.caption, lyrics="[Instrumental]", instrumental=True, bpm=a.bpm, keyscale=a.key,
        duration=a.duration, inference_steps=a.steps, shift=3.0, seed=a.seed,
        thinking=False, use_cot_metas=False, use_cot_caption=False, use_cot_language=False,
        fade_in_duration=0.5, fade_out_duration=2.5,
    )
    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[a.seed], audio_format="flac")
    save_dir = out.parent / "_acestep"
    result = generate_music(dit, None, params, config, save_dir=str(save_dir))
    if not result.success or not result.audios:
        print(f"error: generation failed: {result.error}", file=sys.stderr)
        return 1
    metrics["generate_s"] = round(time.perf_counter() - t, 1)

    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(result.audios[0]["path"], out)
    shutil.rmtree(save_dir, ignore_errors=True)
    metrics.update(total_s=round(time.perf_counter() - t0, 1), peak_rss_gb=peak_rss_gb())
    Path(f"{out}.json").write_text(json.dumps({"caption": a.caption, "bpm": a.bpm, "key": a.key,
                                               "seed": a.seed, "duration": a.duration,
                                               "metrics": metrics}, indent=2), encoding="utf-8")
    print(f"wrote {out}: {metrics}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
