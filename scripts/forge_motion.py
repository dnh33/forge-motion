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
    source: Path
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
        return {
            "item": self.item,
            "source": str(self.source),
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

    clips = []
    for name, raw in items.items():
        if not isinstance(raw, dict):
            raise SpecError(f"clip {name!r} must be an object")
        merged = {**CLIP_DEFAULTS, **raw}

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
        if not source_raw:
            raise SpecError(f"clip {name!r} has no 'from' image; motion is image-driven")
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


def importable(names) -> set:
    """Return the subset of `names` that can actually be imported here."""
    found = set()
    for name in names:
        try:
            __import__(name)
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


def check_backend(clip: Clip, report: dict, available=None) -> list:
    """Return the list of things standing between this clip and a render."""
    missing = []
    if report["torch"] is None:
        missing.append("torch (with CUDA) is not installed")
    elif not report["cuda"]:
        missing.append("torch is installed but no CUDA device is visible")
    if report["diffusers"] is None:
        missing.append("diffusers is not installed")

    if available is None:
        available = importable({"sentencepiece", "protobuf", "tiktoken"})
    missing += tokenizer_blockers(available)

    need = BACKENDS[clip.backend]["min_vram_gb"]
    have = report.get("vram_gb")
    if have is not None and have < need:
        missing.append(
            f"{BACKENDS[clip.backend]['label']} wants about {need} GB of VRAM, "
            f"this device reports {have} GB"
        )
    return missing


# --------------------------------------------------------------------- render

def render(clip: Clip, out_dir: Path, report: dict) -> Path:
    """Render one clip. Imports the heavy stack here, not at module import.

    Image-to-video is a different pipeline class from text-to-video: LTXPipeline
    takes a prompt alone, while LTXImageToVideoPipeline takes the frame to animate.
    """
    blockers = check_backend(clip, report)
    if blockers:
        raise RuntimeError(
            "cannot render: " + "; ".join(blockers)
            + "\nRun `forge_motion.py check` for the full picture."
        )

    import torch  # noqa: PLC0415
    from diffusers import LTXImageToVideoPipeline  # type: ignore  # noqa: PLC0415
    from diffusers.utils import export_to_video  # type: ignore  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

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
        print(f"      from {c.source.name}: {c.prompt}")
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
        blockers = check_backend(c, report)
        if blockers:
            ready = False
            print(f"  {c.item}: BLOCKED")
            for b in blockers:
                print(f"      - {b}")
        else:
            print(f"  {c.item}: ready")
    if not ready:
        print("\ninstall the GPU stack and try again:")
        print("  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124")
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
            out = render(c, Path(args.out), report)
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
    c.set_defaults(func=_check)

    r = sub.add_parser("render", help="render clips (needs a CUDA GPU)")
    r.add_argument("--item", required=True)
    r.add_argument("--out", default="renders")
    r.set_defaults(func=_render)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
