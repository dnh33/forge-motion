#!/usr/bin/env python3
"""Sound effects for one shot: MMAudio video-to-audio, on whatever device is here.

    sfx.py --mmaudio DIR --video shot.mp4 --prompt "..." --out shot.flac [--duration S]

MMAudio (hkchengrex/MMAudio, CVPR 2025) watches the frames (CLIP at 8 fps,
Synchformer at 25 fps for sync) and hears the prompt. Code is MIT; the weights are
CC-BY-NC 4.0, so output is for non-commercial use. Run from a clone of the repo
(`--mmaudio`): it resolves its weight paths relative to that directory and
downloads them on first use.

Writes <out> plus <out>.json with timings and peak memory, as render does.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

NEGATIVE = "music, singing, speech, talking, melody"  # the score is a separate track


def peak_rss_gb():
    try:
        import resource  # noqa: PLC0415
    except ImportError:
        return None
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sfx")
    ap.add_argument("--mmaudio", required=True, help="clone of hkchengrex/MMAudio")
    ap.add_argument("--video", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--variant", default="large_44k_v2")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    a = ap.parse_args(argv)

    video, out = Path(a.video).resolve(), Path(a.out).resolve()
    os.chdir(a.mmaudio)  # MMAudio's weight paths are relative to its repo
    sys.path.insert(0, os.getcwd())

    import torch  # noqa: PLC0415
    import soundfile  # noqa: PLC0415
    from mmaudio.eval_utils import all_model_cfg, generate, load_video  # noqa: PLC0415
    from mmaudio.model.flow_matching import FlowMatching  # noqa: PLC0415
    from mmaudio.model.networks import get_my_mmaudio  # noqa: PLC0415
    from mmaudio.model.utils.features_utils import FeaturesUtils  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    dtype = getattr(torch, a.dtype)
    metrics = {"device": device, "dtype": a.dtype, "variant": a.variant, "steps": a.steps}
    t0 = time.perf_counter()

    model = all_model_cfg[a.variant]
    model.download_if_needed()
    seq_cfg = model.seq_cfg
    net = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    features = FeaturesUtils(tod_vae_ckpt=model.vae_path, synchformer_ckpt=model.synchformer_ckpt,
                             enable_conditions=True, mode=model.mode,
                             bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                             need_vae_encoder=False).to(device, dtype).eval()
    metrics["load_s"] = round(time.perf_counter() - t0, 1)

    t = time.perf_counter()
    info = load_video(video, a.duration)
    seq_cfg.duration = info.duration_sec
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
    rng = torch.Generator(device=device).manual_seed(a.seed)
    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=a.steps)
    with torch.inference_mode():
        audio = generate(info.clip_frames.unsqueeze(0), info.sync_frames.unsqueeze(0), [a.prompt],
                         negative_text=[NEGATIVE], feature_utils=features, net=net, fm=fm, rng=rng,
                         cfg_strength=a.cfg)
    metrics["generate_s"] = round(time.perf_counter() - t, 1)

    out.parent.mkdir(parents=True, exist_ok=True)
    # Not torchaudio.save: it now writes through torchcodec, whose PyPI wheel is a CUDA
    # build that dies on a CPU runner looking for libnvrtc, after the audio is made.
    soundfile.write(str(out), audio.float().cpu()[0].numpy().T, seq_cfg.sampling_rate)
    metrics.update(total_s=round(time.perf_counter() - t0, 1), peak_rss_gb=peak_rss_gb(),
                   seconds=round(info.duration_sec, 3), sampling_rate=seq_cfg.sampling_rate)
    Path(f"{out}.json").write_text(json.dumps({"video": video.name, "prompt": a.prompt,
                                               "negative": NEGATIVE, "seed": a.seed,
                                               "metrics": metrics}, indent=2), encoding="utf-8")
    print(f"wrote {out}: {metrics}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
