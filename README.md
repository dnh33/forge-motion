# forge-motion

> ## Status: the runner path is the default. The local GPU path is EXPERIMENTAL.
>
> **Clips render on GitHub's free CPU runners** with
> [`render.yml`](.github/workflows/render.yml): Actions, "Render on a CPU runner",
> pick a clip. It costs nothing, needs nothing installed, and cannot affect the
> machine you are sitting at. Measured: a **10 second, 384x512 clip in 32 minutes**
> (numbers below).
>
> The local GPU path works too, but its run **saturated the VRAM that the display
> also needs and froze the owner's PC**. It is kept as a record and for
> deliberate experimentation, and no tooling here should reach for it by default.

Turn a still into a short clip.

Still images come from the cloud pipeline ([forge-images](https://github.com/dnh33/forge-images));
motion comes from LTX-Video 2B 0.9.8 distilled, on a runner's CPU. Why that model,
and what an earlier version of this README got wrong, is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

The consult point between the two halves is a file: a PNG on the images repo is a
valid `from` in a motion spec.

## Read this before running it

A clip needs the whole model resident: this run held **15.9 of the card's 16.3 GB**.
On a machine whose GPU also drives the display, that is enough to stall the desktop,
and it did. If you run this anyway:

- Do it on a machine you are not otherwise using, or with the display on a
  different adapter.
- Expect the first steps to crawl (15 to 69 s each) before settling near 4.2 s.
- Expect ~5 minutes of disk-bound loading before the first step, every time.

## What it needs

- A CUDA GPU with 10 GB or more of VRAM.
- `torch` (CUDA build) and `diffusers`.
- One model download, roughly 10 to 30 GB depending on the backend (LTX-Video 2B is
  27 GB as shipped, see below).

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python3 -m pip install -r requirements-gpu.txt
```

`requirements-gpu.txt` is the exact set that produced a clip, with the reasons. Two
points in it are not optional and are easy to get wrong:

- **cu128, not cu124.** Blackwell (compute capability 12.0) has no kernels in the
  cu124 builds. And the default PyPI wheel on Windows is CPU-only, which is how you
  get "torch is installed but no CUDA device is visible".
- **`sentencepiece` and `protobuf`.** The T5 tokenizer is a SentencePiece model.
  Without these, transformers falls back to a tiktoken extractor that cannot parse
  it, and the run dies *after* loading 27 GB. `check` now names them up front.

## Use

`plan` and `check` need no GPU, no torch, and no network. They are the whole
point of the design: know the cost before you pay it.

```bash
python3 scripts/forge_motion.py plan            # what would be rendered, and what it costs
python3 scripts/forge_motion.py check           # what this machine can actually do
python3 scripts/forge_motion.py render --item seed-drift --out renders/
```

`render` refuses before it imports anything heavy if the machine cannot do the
job, so a missing GPU costs you a second, not a download.

## The spec

One file, `prompts/motion.json`, one entry per clip:

```json
{
  "clips": {
    "ember-drift": {
      "from": "renders/portraits/portraits-marshal-s1101.png",
      "prompt": "slow ember drift upward past the subject, camera pushes in",
      "backend": "ltx-video-2b",
      "fps": 24,
      "seconds": 3,
      "size": [512, 768],
      "seed": 7
    }
  }
}
```

| field | default | meaning |
|---|---|---|
| `from` | required | the still to animate. Relative paths resolve against the working directory |
| `prompt` | required | what moves. Motion needs a direction |
| `backend` | `ltx-video-2b` | which model; see the table below |
| `fps` | `24` | frames per second, 1 to 60 |
| `seconds` | `3` | duration; `seconds x fps` frames, capped at 600 |
| `size` | `[512, 768]` | the canvas, `[width, height]` |
| `seed` | `0` | fixed, so a re-run is reproducible |

Specs are validated before anything is loaded: an unknown backend, an empty
prompt, a missing source image, a bad canvas or an absurd frame count all fail
with a sentence naming the clip, not a stack trace.

## Backends

| backend | download | VRAM floor | note |
|---|---|---|---|
| `ltx-video-2b` | **27 GB measured** | 8 GB | fastest. Its own open-weights licence, not an OSI one |
| `svd-xt` | ~10 GB | 10 GB | image-to-video, fixed 25 frames, community licence |
| `wan-2.1-i2v` | ~30 GB | 16 GB | highest quality, heaviest |

That 27 GB is not a guess, and it is not the ~9 GB an earlier version of this file
claimed. The `Lightricks/LTX-Video` repo ships **fp32** weights: a 19 GB T5-XXL text
encoder in four shards, a 7.7 GB transformer, and a 1.7 GB VAE. They load into
roughly half that as bf16, which is why a 16 GB card fits with about 400 MB to
spare. Budget disk, not VRAM.

## Measured on a free runner

`ubuntu-latest` (4 vCPU, 16 GB RAM, no GPU), `ltx-video-2b-distilled`, float32, 8 steps:

| clip | frames | size | encode | load | per step | decode | render total | job |
|---|---|---|---|---|---|---|---|---|
| `forgemaster-smoke` | 49 (2 s) | 288x384 | 173 s | 43 s | 14 s | 16 s | 5 m 48 s | 8 m 30 s |
| `forgemaster-10s` | 241 (10 s) | 384x512 | 239 s | 53 s | 168 s | 294 s | **32 m 17 s** | 38 m 15 s |

- **Memory** peaks at 14.7 GB while the T5 loads, then holds about 14.3 GB through the
  denoise. The workflow adds an 8 GB swapfile as a guard; the 10 s decode used up to 5.8 GB
  of it. A larger canvas will lean on swap harder, so watch `<clip>.mem.log`.
- **Time** scales with tokens, not frames alone: 384x512x241 is 7.9x the smoke's tokens
  and cost 12x per step (attention is quadratic). The 6 hour ceiling leaves room for
  larger canvases or longer clips.
- **What it looks like:** coherent through all 241 frames, identity held, smoke and flame
  move. The camera push was stronger than "very slow" asked for.

Every run publishes `<clip>.mp4`, a JSON sidecar with these timings, the log and the
memory samples to `renders/<clip>/` on the `renders` branch.

## Measured on a local GPU (experimental)

One clip, end to end, on an RTX 5060 Ti 16 GB (compute capability 12.0), 512x768,
73 frames, 50 steps:

| stage | measured |
|---|---|
| downloading the weights | 6 m 13 s (once) |
| loading them from disk (fp32, 27 GB) | **4 m 47 s** |
| denoising, 50 steps | **11 m 30 s** |
| **total wall clock** | **18 m 37 s** |
| output | 73 frames, 512x768, h264, 24 fps, 3.04 s, 34 KB |

Two things worth knowing before you plan around it:

- **Loading is disk-bound and not small.** Nearly five minutes before a single step
  runs, because 27 GB is read from disk. This dominates short clips.
- **The first steps are much slower than the rest** (15 to 69 s each, settling to
  about 4.2 s/step). That is VRAM pressure at the ceiling: the run holds 15.9 of
  16.3 GB. Dropping `size` or `steps` buys headroom and time.

`forge_motion.py check` will tell you your own machine's numbers before you spend
the download.

## Tests

```bash
python3 -m pytest tests -q     # no GPU, no torch, no network
```

23 tests cover the spec contract, the plan arithmetic, the device reporting and
the refusal path. `render` is only ever exercised through its refusal, with a
stubbed device report, so a machine that does own a GPU still cannot start a real
multi-gigabyte render from a test run. CI runs the suite on every push, and
additionally asserts that the committed example spec resolves and that `check`
reports "blocked" honestly on a CPU runner rather than crashing.

## Scope

Image-to-video only. Text-to-video, upscaling, audio and editorial are all
different problems and are not here.
