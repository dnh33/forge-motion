# Why forge-motion is hybrid

This document exists so the split is not mistaken for a shortcut. It is a
constraint, and the constraint is severe.

## The claim

Video diffusion does not fit a free GitHub-hosted runner. Not slowly. Not at a
lower resolution. It does not fit.

A `ubuntu-latest` runner has 4 vCPU and 16 GB of RAM, no GPU. A video model
denoises a whole clip at once: a 2 to 5 second clip at 480p is 25 to 120 frames,
each of which is a full diffusion pass, and the model keeps the latent volume in
memory throughout. The arithmetic works out to hours per clip on four general
cores, against a 6 hour job ceiling, with peak memory well past 16 GB for the
14B class of model.

**How firm is this?** It is an estimate from model size and frame counts, not a
measured run of our own. We did not attempt a multi-hour CPU job to prove it.
The measured fact from this project is narrower and unambiguous: a **single
768x1024 still at 4 steps takes 2702 seconds (45 minutes)** on one of these
runners. A clip is 25 to 120 of those passes, with more memory per pass. Two
orders of magnitude is the conservative reading.

If you want the number rather than the argument, run `forge-motion check` on a
machine that matters, or dispatch a one-clip attempt yourself and watch the
timeout. The estimate is labelled as an estimate on purpose.

## The split

| half | where | what it does |
|---|---|---|
| **frames** | CI, free | generate the still the motion starts from. This is [`forge-images`](https://github.com/dnh33/forge-images), unchanged. |
| **motion** | your GPU | an image-to-video pass over that still. This repository. |

The consult point is a file: a PNG in `renders/<set>/` on the images repo is a
valid `from` in a motion spec. No service, no queue, no account.

## What this repository will not do

- It will not pretend a CPU path exists "for CI". That path would time out, and
  a workflow that always times out is worse than no workflow.
- It will not download a 9 to 30 GB model as a side effect of a test. `render`
  refuses before importing the heavy stack, and the test suite only ever
  exercises that refusal.
- It will not spend your bandwidth silently. `plan` prints the download size and
  the VRAM floor before anything is fetched.

## Hardware this targets

A single consumer card with 10 GB or more of VRAM. The reference machine for
this project is an RTX 5060 Ti 16 GB, which clears every backend listed in
`scripts/forge_motion.py` except the largest at full precision.

## Choosing a backend

| backend | download | VRAM floor | licence note |
|---|---|---|---|
| `ltx-video-2b` | 27 GB (fp32 on disk) | 8 GB | fastest; its own open-weights licence, not an OSI one |
| `svd-xt` | ~10 GB | 10 GB | image-to-video only, fixed 25 frames, community licence |
| `wan-2.1-i2v` | ~30 GB | 16 GB | highest quality, heaviest |

Verified end to end on an RTX 5060 Ti 16 GB: 27 GB download, 4 m 47 s to load,
11 m 30 s for 50 steps, 18 m 37 s total for a 73-frame 512x768 clip. The download
is the smaller half of the cost; loading fp32 weights off disk is the larger half
for a short clip.

Check the licence of whichever you pick against what you intend to do with the
output. Two of the three are not OSI-approved licences.
