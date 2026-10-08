# forge-motion

Turn a still into a short clip, on your own GPU.

This is the motion half of a hybrid pipeline. Still images are generated in CI for
free by [`forge-images`](https://github.com/dnh33/forge-images); motion is
synthesised locally, because video diffusion does not fit a free CPU runner. The
reasoning, with the numbers, is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

The consult point between the two halves is a file: a PNG on the images repo is a
valid `from` in a motion spec.

## What it needs

- A CUDA GPU with 10 GB or more of VRAM.
- `torch` (CUDA build) and `diffusers`.
- One model download, 9 to 30 GB depending on the backend.

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install diffusers transformers accelerate imageio[ffmpeg] pillow
```

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
| `ltx-video-2b` | ~9 GB | 8 GB | fastest. Its own open-weights licence, not an OSI one |
| `svd-xt` | ~10 GB | 10 GB | image-to-video, fixed 25 frames, community licence |
| `wan-2.1-i2v` | ~30 GB | 16 GB | highest quality, heaviest |

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
