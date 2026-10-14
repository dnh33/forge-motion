#!/usr/bin/env python3
"""forge-motion: turn a still into a short clip.

This is the local half of a hybrid pipeline, and the split is deliberate. Video
diffusion on a free 4-vCPU runner is not slow, it is impossible: a 2 to 5 second
clip at a usable resolution takes several hours, past the job timeout, and
usually runs out of memory first. So frame generation can happen in CI, but
motion synthesis happens on a real GPU.

Nothing here talks to a network or a database. The GPU work is lazy-imported, so
`plan` and `check` work on any machine, including one with no GPU and no torch.

    forge_motion.py plan  [--spec S] [--item I] [--json]
    forge_motion.py check [--spec S]
    forge_motion.py render --item I [--spec S] [--out DIR]

`plan` and `check` never need a GPU. `render` does.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SPEC = "prompts/motion.json"

# Backends are described rather than hardcoded at the call site, so `check` can
# tell the user what a render would cost before they commit a download.
BACKENDS = {
    "ltx-video-2b": {
        "label": "LTX-Video 2B",
        "repo": "Lightricks/LTX-Video",
        "approx_download_gb": 9.0,
        "min_vram_gb": 8,
        "note": "fastest of the three; its own open-weights licence, not an OSI one",
    },
    "svd-xt": {
        "label": "Stable Video Diffusion XT",
        "repo": "stabilityai/stable-video-diffusion-img2vid-xt",
        "approx_download_gb": 10.0,
        "min_vram_gb": 10,
        "note": "image-to-video only, fixed 25 frames, community licence",
    },
    "wan-2.1-i2v": {
        "label": "Wan 2.1 I2V 14B",
        "repo": "Wan-AI/Wan2.1-I2V-14B-480P",
        "approx_download_gb": 30.0,
        "min_vram_gb": 16,
        "note": "highest quality of the three, and the heaviest",
    },
    # The only backend with a CPU path, and the one the runner workflow uses. The
    # checkpoint ships as one original-format file; its transformer and VAE match
    # the 0.9.5 diffusers configs (read from the file's own header), and 0.9.5 is
    # where the T5 encoder, tokenizer and scheduler come from. The T5 is 17.7 GB of
    # fp32 on disk and ~9 GB once loaded in bf16, so it runs first and is freed
    # before the transformer loads.
    "ltx-video-2b-distilled": {
        "label": "LTX-Video 2B 0.9.8 distilled",
        "repo": "Lightricks/LTX-Video",
        "single_file": "ltxv-2b-0.9.8-distilled.safetensors",
        "config_repo": "Lightricks/LTX-Video-0.9.5",
        "approx_download_gb": 23.6,
        "min_vram_gb": 8,
        "cpu": True,
        "t2v": True,  # `from` is optional: no still means text-to-video
        # Guidance- and timestep-distilled: this schedule, guidance 1.0, no CFG.
        "timesteps": [1000, 993, 987, 981, 975, 909, 725, 0.03],
        "note": "8 fixed steps, one forward pass each; the only backend that runs on a CPU",
    },
}
DEFAULT_BACKEND = "ltx-video-2b"

CLIP_DEFAULTS = {
    "backend": DEFAULT_BACKEND, "fps": 24, "seconds": 3, "size": [512, 768],
    # The LTX base model is not step-distilled; its own guidance is ~50 steps.
    # Step-distilled checkpoints want 4 to 10 and guidance_scale 1.0.
    "steps": 50,
}

# LTX's video VAE compresses time by 8, so the pipeline needs num_frames = 8n+1.
# Anything else fails inside the VAE rather than at the argument.
FRAME_ALIGN = 8
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"


def align_frames(requested: int, step: int = FRAME_ALIGN) -> int:
    """Round a frame count up to the nearest 8n+1 the video VAE can encode."""
    if requested <= 1:
        return 1
    n = -(-(requested - 1) // step)  # ceil((requested - 1) / step)
    return n * step + 1


class SpecError(Exception):
    """A spec that cannot be rendered as written."""


@dataclass
class Clip:
    item: str
    source: Path | None  # None: text-to-video
    prompt: str
    backend: str
    fps: int
    seconds: float
    size: list
    seed: int
    steps: int
    requested_frames: int = field(init=False)
    frames: int = field(init=False)

    def __post_init__(self):
        self.requested_frames = int(round(self.seconds * self.fps))
        self.frames = align_frames(self.requested_frames)

    @property
    def out_name(self) -> str:
        return f"{self.item}-s{self.seed}"

    def manifest(self) -> dict:
        backend = BACKENDS[self.backend]
        schedule = (
            {"timesteps": backend["timesteps"], "guidance_scale": 1.0}
            if "timesteps" in backend else {}
        )
        return {
            **schedule,
            "item": self.item,
            "mode": "image-to-video" if self.source else "text-to-video",
            "source": str(self.source) if self.source else None,
            "prompt": self.prompt,
            "backend": self.backend,
            "backend_label": BACKENDS[self.backend]["label"],
            "fps": self.fps,
            "seconds": self.seconds,
            "seconds_actual": round(self.frames / self.fps, 3),
            "frames_requested": self.requested_frames,
            "frames": self.frames,
            "size": self.size,
            "seed": self.seed,
            "steps": self.steps,
            "negative_prompt": NEGATIVE_PROMPT,
            "output": f"{self.out_name}.mp4",
        }


# ---------------------------------------------------------------- spec loading

def load_spec(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise SpecError(f"no spec at {p}")
    try:
        spec = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SpecError(f"{p} is not valid JSON: {e}") from e
    if not isinstance(spec.get("clips"), dict) or not spec["clips"]:
        raise SpecError(f"{p} has no 'clips' object")
    return spec


def resolve_clips(spec: dict, root: Path | None = None, only: str | None = None) -> list[Clip]:
    """Turn a spec into concrete clips. Raises SpecError on anything renderable-broken."""
    root = root or Path.cwd()
    items = spec["clips"]
    if only is not None:
        if only not in items:
            raise SpecError(f"no clip named {only!r} in the spec (have: {sorted(items)})")
        items = {only: items[only]}

    # A film spec writes its language once: `defaults` for every shot, `cast` for
    # character sheets that a prompt names as {cat}, and a `style` bible appended to
    # every prompt. Repeating the exact words is what holds a character across cuts.
    shared = spec.get("defaults", {})
    cast = spec.get("cast", {})
    style = (spec.get("style") or "").strip()

    clips = []
    for name, raw in items.items():
        if not isinstance(raw, dict):
            raise SpecError(f"clip {name!r} must be an object")
        raw = {**shared, **raw}
        if raw.get("prompt"):
            try:
                text = raw["prompt"].format(**cast)
            except KeyError as e:
                raise SpecError(f"clip {name!r} names {e.args[0]!r}, which is not in the cast") from e
            raw["prompt"] = f"{text} {style}".strip()
        backend_raw = raw.get("backend", CLIP_DEFAULTS["backend"])
        fixed = BACKENDS.get(backend_raw, {}).get("timesteps")
        if fixed is not None and "steps" in raw and int(raw["steps"]) != len(fixed):
            raise SpecError(
                f"clip {name!r} asks for {raw['steps']} steps, but {backend_raw} is "
                f"timestep-distilled to exactly {len(fixed)}"
            )
        merged = {**CLIP_DEFAULTS, **({"steps": len(fixed)} if fixed else {}), **raw}

        prompt = (merged.get("prompt") or "").strip()
        if not prompt:
            raise SpecError(f"clip {name!r} has an empty prompt; motion needs a direction")

        backend = merged["backend"]
        if backend not in BACKENDS:
            raise SpecError(
                f"clip {name!r} names an unknown backend {backend!r} "
                f"(have: {sorted(BACKENDS)})"
            )

        source_raw = merged.get("from")
        source = None
        if not source_raw and not BACKENDS[backend].get("t2v"):
            raise SpecError(f"clip {name!r} has no 'from' image; {backend} is image-to-video only")
        if source_raw:
            source = Path(source_raw)
            if not source.is_absolute():
                source = (root / source).resolve()
            if not source.is_file():
                raise SpecError(f"clip {name!r} points at a missing image: {source}")

        size = merged["size"]
        if not (isinstance(size, list) and len(size) == 2 and all(int(v) > 0 for v in size)):
            raise SpecError(f"clip {name!r} has a bad size {size!r}; expected [width, height]")

        fps = int(merged["fps"])
        if fps < 1 or fps > 60:
            raise SpecError(f"clip {name!r} has fps {fps}; expected 1 to 60")

        steps = int(merged["steps"])
        if steps < 1 or steps > 200:
            raise SpecError(f"clip {name!r} has steps {steps}; expected 1 to 200")

        seconds = float(merged["seconds"])
        if seconds <= 0:
            raise SpecError(f"clip {name!r} has a non-positive duration")
        if seconds * fps > 600:
            raise SpecError(
                f"clip {name!r} asks for {int(seconds * fps)} frames; "
                "over 600 is a different problem, not a short clip"
            )

        clips.append(Clip(
            item=name, source=source, prompt=prompt, backend=backend,
            fps=fps, seconds=seconds, size=[int(v) for v in size],
            seed=int(merged.get("seed", 0)), steps=steps,
        ))
    return clips


# ---------------------------------------------------------------------- check

def gpu_report() -> dict:
    """Report what the local machine can actually do. Never raises."""
    report = {"cuda": False, "device": None, "vram_gb": None, "torch": None, "diffusers": None}
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return report
    report["torch"] = getattr(torch, "__version__", "?")
    try:
        from importlib.metadata import version  # noqa: PLC0415
        report["diffusers"] = version("diffusers")
    except Exception:
        pass
    if torch.cuda.is_available():
        report["cuda"] = True
        report["device"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        report["vram_gb"] = round(props.total_memory / 1024**3, 1)
    return report


# Package name -> the module it installs, where the two differ.
IMPORT_NAMES = {"protobuf": "google.protobuf"}


def importable(names) -> set:
    """Return the subset of package `names` that can actually be imported here."""
    found = set()
    for name in names:
        try:
            __import__(IMPORT_NAMES.get(name, name))
            found.add(name)
        except Exception:
            pass
    return found


def tokenizer_blockers(available) -> list:
    """The T5 tokenizer needs SentencePiece; the tiktoken fallback cannot read it.

    Missing these does not fail at import. It fails after the whole pipeline has
    loaded, which on a 27 GB model is minutes of disk I/O thrown away.
    """
    need = {"sentencepiece", "protobuf"}
    return [
        f"{name} is not installed (the T5 tokenizer needs it)"
        for name in sorted(need - set(available))
    ]


def check_backend(clip: Clip, report: dict, available=None, device: str = "cuda") -> list:
    """Return the list of things standing between this clip and a render."""
    missing = []
    backend = BACKENDS[clip.backend]
    if device == "cpu" and not backend.get("cpu"):
        missing.append(f"{backend['label']} has no CPU path (use ltx-video-2b-distilled)")
    if report["torch"] is None:
        missing.append("torch is not installed" if device == "cpu"
                       else "torch (with CUDA) is not installed")
    elif device != "cpu" and not report["cuda"]:
        missing.append("torch is installed but no CUDA device is visible")
    if report["diffusers"] is None:
        missing.append("diffusers is not installed")

    if available is None:
        available = importable({"sentencepiece", "protobuf", "tiktoken"})
    missing += tokenizer_blockers(available)

    need = backend["min_vram_gb"]
    have = report.get("vram_gb")
    if device != "cpu" and have is not None and have < need:
        missing.append(
            f"{BACKENDS[clip.backend]['label']} wants about {need} GB of VRAM, "
            f"this device reports {have} GB"
        )
    return missing


# --------------------------------------------------------------------- render

def render(clip: Clip, out_dir: Path, report: dict, device: str = "cuda",
           dtype: str = "bfloat16") -> Path:
    """Render one clip. Imports the heavy stack here, not at module import.

    Image-to-video is a different pipeline class from text-to-video: LTXPipeline
    takes a prompt alone, while LTXImageToVideoPipeline takes the frame to animate.
    """
    blockers = check_backend(clip, report, device=device)
    if blockers:
        raise RuntimeError(
            "cannot render: " + "; ".join(blockers)
            + "\nRun `forge_motion.py check` for the full picture."
        )
    if "single_file" in BACKENDS[clip.backend]:
        return render_distilled(clip, out_dir, device, dtype)

    import torch  # noqa: PLC0415
    from diffusers import LTXImageToVideoPipeline  # type: ignore  # noqa: PLC0415
    from diffusers.utils import export_to_video  # type: ignore  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    print(
        "warning: this loads the whole model onto the GPU and will saturate VRAM\n"
        "         that the display also needs. On a desktop machine that can stall\n"
        "         the session. This local path is experimental; the cloud pipeline\n"
        "         (forge-images) is the supported way to produce imagery.",
        file=sys.stderr,
    )

    model_id = BACKENDS[clip.backend]["repo"]
    pipe = LTXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    # The T5-XXL text encoder alone is about 9 GB in bf16, so tiling the decode is
    # what keeps a 16 GB card off the swap file.
    pipe.vae.enable_tiling()
    pipe.to("cuda")

    image = Image.open(clip.source).convert("RGB").resize(tuple(clip.size))
    generator = torch.Generator(device="cuda").manual_seed(clip.seed)
    frames = pipe(
        image=image,
        prompt=clip.prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=clip.size[0],
        height=clip.size[1],
        num_frames=clip.frames,
        num_inference_steps=clip.steps,
        generator=generator,
    ).frames[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{clip.out_name}.mp4"
    export_to_video(frames, str(out), fps=clip.fps)
    return out


def peak_rss_gb() -> float | None:
    """Peak resident memory of this process, or None where getrusage is absent."""
    try:
        import resource  # noqa: PLC0415  (POSIX only)
    except ImportError:
        return None
    # ru_maxrss is KiB on Linux.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2)


def render_distilled(clip: Clip, out_dir: Path, device: str, dtype_name: str) -> Path:
    """Render with the single-file distilled checkpoint, on CPU or GPU.

    Memory is staged so the peak is the larger of the two halves, not their sum:
    the T5 encoder (~9 GB in bf16) encodes the prompt and is dropped before the
    transformer and VAE load. Every phase is timed and written to a sidecar, because
    the point of a runner job is the measurement as much as the clip.
    """
    import gc  # noqa: PLC0415
    import time  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from diffusers import (  # type: ignore  # noqa: PLC0415
        AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler,
        LTXConditionPipeline, LTXVideoTransformer3DModel,
    )
    from diffusers.utils import export_to_video  # type: ignore  # noqa: PLC0415
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import T5EncoderModel, T5TokenizerFast  # noqa: PLC0415

    backend = BACKENDS[clip.backend]
    cfg = backend["config_repo"]
    dtype = getattr(torch, dtype_name)
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    metrics: dict = {"device": device, "dtype": dtype_name, "threads": torch.get_num_threads(),
                     "torch": torch.__version__}
    t0 = time.perf_counter()

    def mark(phase: str, since: float) -> float:
        now = time.perf_counter()
        metrics[f"{phase}_s"] = round(now - since, 1)
        metrics[f"rss_after_{phase}_gb"] = peak_rss_gb()
        print(f"[{phase}] {now - since:.1f}s, peak rss {peak_rss_gb()} GB", flush=True)
        return now

    # 1. Prompt -> embeddings, then free the encoder.
    tokenizer = T5TokenizerFast.from_pretrained(cfg, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        cfg, subfolder="text_encoder", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    enc = LTXConditionPipeline(scheduler=None, vae=None, text_encoder=text_encoder,
                               tokenizer=tokenizer, transformer=None).to(device)
    with torch.no_grad():
        embeds, mask, _, _ = enc.encode_prompt(
            clip.prompt, do_classifier_free_guidance=False, device=device)
    embeds = embeds.to(dtype)
    del enc, text_encoder
    gc.collect()
    t = mark("encode", t0)

    # 2. Transformer + VAE from the one checkpoint file.
    ckpt = hf_hub_download(backend["repo"], backend["single_file"])
    transformer = LTXVideoTransformer3DModel.from_single_file(
        ckpt, config=cfg, subfolder="transformer", torch_dtype=dtype)
    vae = AutoencoderKLLTXVideo.from_single_file(ckpt, config=cfg, subfolder="vae", torch_dtype=dtype)
    vae.enable_tiling()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(cfg, subfolder="scheduler")
    pipe = LTXConditionPipeline(scheduler=scheduler, vae=vae, text_encoder=None,
                                tokenizer=tokenizer, transformer=transformer).to(device)
    t = mark("load", t)

    # 3. Denoise and decode. One line per step so a watched log shows the rate as it
    # happens; the time from the last step to the return is the VAE decode.
    steps: list = []
    last = [0.0]

    def on_step(_pipe, i, _t, kwargs):
        now = time.perf_counter()
        steps.append(round(now - last[0], 1))
        last[0] = now
        print(f"[step {i + 1}/{clip.steps}] {steps[-1]}s, peak rss {peak_rss_gb()} GB", flush=True)
        return kwargs

    image = Image.open(clip.source).convert("RGB").resize(tuple(clip.size)) if clip.source else None
    generator = torch.Generator(device=device).manual_seed(clip.seed)
    t = last[0] = time.perf_counter()
    with torch.no_grad():
        latents = pipe(
            image=image, prompt_embeds=embeds, prompt_attention_mask=mask,
            width=clip.size[0], height=clip.size[1], num_frames=clip.frames,
            frame_rate=clip.fps, timesteps=backend["timesteps"], guidance_scale=1.0,
            image_cond_noise_scale=0.025, generator=generator,
            callback_on_step_end=on_step, output_type="latent",
        ).frames
    metrics["step_s"] = steps
    t = mark("denoise", t)

    # 4. Decode with the transformer gone. Held through the decode, its 7.7 GB of
    # fp32 weights pushed a 384x512x241 run 5.8 GB into swap. The steps below are
    # the pipeline's own decode (diffusers 0.41, LTXConditionPipeline.__call__).
    pipe.transformer = None
    del transformer
    gc.collect()
    with torch.no_grad():
        latents = pipe._denormalize_latents(
            latents, vae.latents_mean, vae.latents_std, vae.config.scaling_factor).to(dtype)
        noise = torch.randn(latents.shape, generator=generator, device=device, dtype=dtype)
        latents = (1 - 0.025) * latents + 0.025 * noise  # decode_noise_scale
        timestep = torch.tensor([0.05], device=device, dtype=dtype)  # decode_timestep
        video = vae.decode(latents, timestep, return_dict=False)[0]
        frames = pipe.video_processor.postprocess_video(video, output_type="pil")[0]
    t = mark("decode", t)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{clip.out_name}.mp4"
    export_to_video(frames, str(out), fps=clip.fps)
    metrics["total_s"] = round(time.perf_counter() - t0, 1)
    metrics["peak_rss_gb"] = peak_rss_gb()
    sidecar = {**clip.manifest(), "metrics": metrics}
    (out_dir / f"{clip.out_name}.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return out


# ----------------------------------------------------------------------- main

def _plan(args) -> int:
    try:
        spec = load_spec(args.spec)
        clips = resolve_clips(spec, root=Path.cwd(), only=args.item)
    except SpecError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps([c.manifest() for c in clips], indent=2))
        return 0
    print(f"{len(clips)} clip(s) from {args.spec}")
    for c in clips:
        b = BACKENDS[c.backend]
        print(f"  {c.item}: {c.frames} frames at {c.fps} fps ({c.seconds}s), "
              f"{c.size[0]}x{c.size[1]}, seed {c.seed}, {c.steps} steps")
        print(f"      backend {b['label']} (~{b['approx_download_gb']} GB, "
              f">= {b['min_vram_gb']} GB VRAM)")
        origin = f"from {c.source.name}" if c.source else "text-to-video"
        print(f"      {origin}: {c.prompt}")
    return 0


def _check(args) -> int:
    report = gpu_report()
    print("device")
    if report["cuda"]:
        print(f"  CUDA: {report['device']} ({report['vram_gb']} GB)")
    else:
        print("  CUDA: not available")
    print(f"  torch:      {report['torch'] or 'not installed'}")
    print(f"  diffusers:  {report['diffusers'] or 'not installed'}")

    try:
        spec = load_spec(args.spec)
        clips = resolve_clips(spec, root=Path.cwd())
    except SpecError as e:
        print(f"\nspec: {e}", file=sys.stderr)
        return 2

    print(f"\nclips ({len(clips)})")
    ready = True
    for c in clips:
        blockers = check_backend(c, report, device=args.device)
        if blockers:
            ready = False
            print(f"  {c.item}: BLOCKED")
            for b in blockers:
                print(f"      - {b}")
        else:
            print(f"  {c.item}: ready")
    if not ready:
        print("\ninstall the GPU stack and try again:")
        print("  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu128")
        print("  python3 -m pip install diffusers transformers accelerate imageio[ffmpeg] pillow")
    return 0 if ready else 1


def _render(args) -> int:
    try:
        spec = load_spec(args.spec)
        clips = resolve_clips(spec, root=Path.cwd(), only=args.item)
    except SpecError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    report = gpu_report()
    failed = 0
    for c in clips:
        try:
            out = render(c, Path(args.out), report, device=args.device, dtype=args.dtype)
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 3
        except Exception as e:  # a render failure should not kill the batch
            print(f"error: {c.item} failed: {e}", file=sys.stderr)
            failed += 1
            continue
        print(f"wrote {out}")
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="forge-motion", description="turn a still into a short clip")
    ap.add_argument("--spec", default=os.environ.get("MOTION_SPEC", DEFAULT_SPEC))
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="resolve the spec into clips (no GPU needed)")
    p.add_argument("--item")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_plan)

    c = sub.add_parser("check", help="report what this machine can render (no GPU needed)")
    c.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    c.set_defaults(func=_check)

    r = sub.add_parser("render", help="render clips (CUDA, or CPU for the distilled backend)")
    r.add_argument("--item", required=True)
    r.add_argument("--out", default="renders")
    r.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    r.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                   help="weights and activations for the distilled backend")
    r.set_defaults(func=_render)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
