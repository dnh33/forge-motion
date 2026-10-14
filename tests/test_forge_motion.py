"""Tests for forge-motion.

These run anywhere: no GPU, no torch, no network. `plan` and `check` are pure
logic, and `render` is exercised only through its refusal path — with a stubbed
device report, so a machine that happens to own a GPU still cannot start a real
multi-gigabyte render from a test run.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "forge_motion.py"


def load_module():
    spec = importlib.util.spec_from_file_location("forge_motion", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules[cls.__module__], so the
    # module has to be registered before it is executed.
    sys.modules["forge_motion"] = module
    spec.loader.exec_module(module)
    return module


fm = load_module()


def make_spec(tmp_path, clips=None, frame=True):
    """Write a spec plus a real source image, and return (spec_path, image_path).

    Each clip gets `from` injected unless it sets its own, so a test never has to
    reference the image path before this call returns it.
    """
    img = tmp_path / "seed.png"
    if frame:
        Image.new("RGB", (64, 96), (90, 40, 20)).save(img)
    if clips is None:
        clips = {"ember": {"prompt": "embers drift upward, slow push"}}
    clips = {k: {"from": str(img), **v} for k, v in clips.items()}
    spec = {"name": "Test motions", "clips": clips}
    path = tmp_path / "motion.json"
    path.write_text(json.dumps(spec))
    return path, img


def run_cli(*args, cwd=None):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd or REPO, capture_output=True, text=True,
    )


# ---------------------------------------------------------------------- spec

def test_a_missing_spec_is_a_readable_error():
    try:
        fm.load_spec("does-not-exist.json")
    except fm.SpecError as e:
        assert "no spec" in str(e)
    else:
        raise AssertionError("expected SpecError")


def test_invalid_json_is_reported_clearly(tmp_path):
    p = tmp_path / "motion.json"
    p.write_text("{ not json")
    try:
        fm.load_spec(p)
    except fm.SpecError as e:
        assert "valid JSON" in str(e)
    else:
        raise AssertionError("expected SpecError")


def test_a_spec_without_clips_is_rejected(tmp_path):
    p = tmp_path / "motion.json"
    p.write_text(json.dumps({"name": "empty"}))
    try:
        fm.load_spec(p)
    except fm.SpecError as e:
        assert "clips" in str(e)
    else:
        raise AssertionError("expected SpecError")


# ------------------------------------------------------------------- resolve

def test_defaults_are_applied(tmp_path):
    spec_path, img = make_spec(tmp_path)
    clips = fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)
    c = clips[0]
    assert c.backend == fm.DEFAULT_BACKEND
    assert c.fps == 24 and c.seconds == 3
    assert c.frames == 73, "3 seconds at 24 fps is 72, snapped up to the 8n+1 grid"
    assert c.requested_frames == 72
    assert c.size == [512, 768]
    assert c.source == img


def test_frames_follow_seconds_and_fps(tmp_path):
    spec_path, img = make_spec(tmp_path, {"a": {"prompt": "p", "fps": 12, "seconds": 2.5}})
    clips = fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)
    assert clips[0].requested_frames == 30
    assert clips[0].frames == 33, "30 is off the 8n+1 grid the video VAE needs"


def test_frames_are_snapped_to_the_vae_grid(tmp_path):
    """LTX's video VAE compresses time by 8, so num_frames must be 8n+1."""
    for requested in (1, 2, 7, 8, 9, 30, 72, 73, 100):
        n = fm.align_frames(requested)
        assert n % 8 == 1, f"{requested} -> {n} is not 8n+1"
        assert n >= requested, f"{requested} -> {n} lost frames"
        assert n - 8 < requested, f"{requested} -> {n} is not the smallest such value"


def test_a_clip_reports_both_the_requested_and_the_rendered_frame_count(tmp_path):
    spec_path, img = make_spec(tmp_path, {"a": {"prompt": "p", "fps": 24, "seconds": 3}})
    m = fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)[0].manifest()
    assert m["frames_requested"] == 72
    assert m["frames"] == 73
    assert m["seconds_actual"] == round(73 / 24, 3), "the real duration, not the asked-for one"


def test_steps_default_to_the_base_model_guidance(tmp_path):
    spec_path, img = make_spec(tmp_path)
    clips = fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)
    assert clips[0].steps == 50


def test_steps_are_settable_per_clip(tmp_path):
    spec_path, img = make_spec(tmp_path, {"a": {"prompt": "p", "steps": 8}})
    clips = fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)
    assert clips[0].steps == 8


def test_an_absurd_step_count_is_rejected(tmp_path):
    spec_path, img = make_spec(tmp_path, {"a": {"prompt": "p", "steps": 5000}})
    try:
        fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)
    except fm.SpecError as e:
        assert "steps" in str(e)
    else:
        raise AssertionError("expected SpecError")


def test_the_manifest_records_the_step_count(tmp_path):
    spec_path, _ = make_spec(tmp_path, {"a": {"prompt": "p", "steps": 12}})
    m = fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)[0].manifest()
    assert m["steps"] == 12, "a sidecar should say how many steps made the clip"


def test_only_selects_one_clip(tmp_path):
    spec_path, img = make_spec(tmp_path, {
        "a": {"prompt": "one"},
        "b": {"prompt": "two"},
    })
    clips = fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path, only="b")
    assert [c.item for c in clips] == ["b"]


def test_asking_for_a_clip_that_is_not_there_lists_what_is(tmp_path):
    spec_path, img = make_spec(tmp_path)
    try:
        fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path, only="nope")
    except fm.SpecError as e:
        assert "ember" in str(e), "the error should name what does exist"
    else:
        raise AssertionError("expected SpecError")


def test_an_empty_prompt_is_rejected(tmp_path):
    spec_path, img = make_spec(tmp_path, {"a": {"prompt": "  "}})
    try:
        fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)
    except fm.SpecError as e:
        assert "empty prompt" in str(e)
    else:
        raise AssertionError("expected SpecError")


def test_a_missing_source_image_is_rejected(tmp_path):
    spec_path, _ = make_spec(tmp_path, {"a": {
        "from": str(tmp_path / "gone.png"), "prompt": "p"}}, frame=False)
    try:
        fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)
    except fm.SpecError as e:
        assert "missing image" in str(e)
    else:
        raise AssertionError("expected SpecError")


def test_an_unknown_backend_is_rejected(tmp_path):
    spec_path, img = make_spec(tmp_path, {"a": {"prompt": "p", "backend": "magic"}})
    try:
        fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)
    except fm.SpecError as e:
        assert "unknown backend" in str(e)
    else:
        raise AssertionError("expected SpecError")


def test_a_bad_canvas_is_rejected(tmp_path):
    spec_path, img = make_spec(tmp_path, {"a": {"prompt": "p", "size": [512]}})
    try:
        fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)
    except fm.SpecError as e:
        assert "size" in str(e)
    else:
        raise AssertionError("expected SpecError")


def test_an_absurd_frame_count_is_rejected(tmp_path):
    """600 frames is the line between a clip and a different problem."""
    spec_path, img = make_spec(tmp_path, {"a": {"prompt": "p", "fps": 60, "seconds": 30}})
    try:
        fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)
    except fm.SpecError as e:
        assert "frames" in str(e)
    else:
        raise AssertionError("expected SpecError")


def test_a_relative_source_is_resolved_against_the_root(tmp_path):
    (tmp_path / "renders").mkdir()
    Image.new("RGB", (32, 32), (1, 2, 3)).save(tmp_path / "renders" / "in.png")
    p = tmp_path / "motion.json"
    p.write_text(json.dumps({"clips": {"a": {"from": "renders/in.png", "prompt": "p"}}}))
    clips = fm.resolve_clips(fm.load_spec(p), root=tmp_path)
    assert clips[0].source == (tmp_path / "renders" / "in.png").resolve()


def test_the_manifest_carries_what_a_sidecar_needs(tmp_path):
    spec_path, _ = make_spec(tmp_path)
    m = fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)[0].manifest()
    for key in ("item", "source", "prompt", "backend", "backend_label", "fps",
                "seconds", "seconds_actual", "frames", "frames_requested",
                "size", "seed", "steps", "negative_prompt", "output"):
        assert key in m, f"manifest is missing {key}"
    assert m["output"].endswith(".mp4")


def test_the_manifest_records_the_negative_prompt(tmp_path):
    spec_path, _ = make_spec(tmp_path)
    m = fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)[0].manifest()
    assert "blurry" in m["negative_prompt"], "provenance should include how it was steered away"


# ----------------------------------------------------------------------- gpu

def test_a_machine_with_nothing_installed_reports_that():
    report = fm.gpu_report()
    assert "cuda" in report and "torch" in report
    assert isinstance(report["cuda"], bool)


def test_blockers_name_the_missing_pieces(tmp_path):
    spec_path, _ = make_spec(tmp_path)
    clip = fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)[0]
    bare = {"cuda": False, "torch": None, "diffusers": None, "vram_gb": None}
    blockers = fm.check_backend(clip, bare)
    assert any("torch" in b for b in blockers)
    assert any("diffusers" in b for b in blockers)


def test_a_small_card_is_called_out_by_name(tmp_path):
    spec_path, _ = make_spec(tmp_path, {"a": {
        "from": str(tmp_path / "seed.png"), "prompt": "p", "backend": "wan-2.1-i2v"}})
    clip = fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)[0]
    report = {"cuda": True, "torch": "2.5", "diffusers": "0.31", "vram_gb": 8.0}
    blockers = fm.check_backend(clip, report)
    assert any("VRAM" in b for b in blockers), blockers


def test_render_refuses_before_importing_anything_heavy(tmp_path):
    """The refusal path must not attempt a download."""
    spec_path, _ = make_spec(tmp_path)
    clip = fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)[0]
    bare = {"cuda": False, "torch": None, "diffusers": None, "vram_gb": None}
    try:
        fm.render(clip, tmp_path / "out", bare)
    except RuntimeError as e:
        assert "cannot render" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
    assert not (tmp_path / "out").exists(), "a refused render must leave nothing behind"


def test_tokenizer_requirements_are_reported_when_absent():
    """Missing these fails *after* loading 27 GB of weights, so warn up front."""
    blockers = fm.tokenizer_blockers({"tiktoken"})
    assert any("sentencepiece" in b for b in blockers), blockers
    assert any("protobuf" in b for b in blockers), blockers


def test_tokenizer_requirements_pass_when_present():
    assert fm.tokenizer_blockers({"sentencepiece", "protobuf", "tiktoken"}) == []


def test_protobuf_is_found_under_its_import_name(monkeypatch):
    """The package is `protobuf`, the module is `google.protobuf`: asking for the
    package name refused a runner that had it installed."""
    import builtins
    seen = []
    real = builtins.__import__

    def fake(name, *a, **k):
        seen.append(name)
        if name == "google.protobuf":
            return object()
        return real(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", fake)
    assert "protobuf" in fm.importable({"protobuf"})
    assert "google.protobuf" in seen


def test_check_backend_surfaces_the_tokenizer_gap(tmp_path):
    spec_path, _ = make_spec(tmp_path)
    clip = fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)[0]
    report = {"cuda": True, "torch": "2.11", "diffusers": "0.41", "vram_gb": 16.0}
    blockers = fm.check_backend(clip, report, available={"tiktoken"})
    assert any("sentencepiece" in b for b in blockers), blockers
    assert not any("VRAM" in b for b in blockers), "the device is fine, only the tokenizer is not"


def test_a_fully_provisioned_machine_reports_no_blockers(tmp_path):
    spec_path, _ = make_spec(tmp_path)
    clip = fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)[0]
    report = {"cuda": True, "torch": "2.11", "diffusers": "0.41", "vram_gb": 16.0}
    available = {"sentencepiece", "protobuf", "tiktoken"}
    assert fm.check_backend(clip, report, available=available) == []


# ------------------------------------------------------------- cpu / distilled

DISTILLED = "ltx-video-2b-distilled"
CPU_REPORT = {"cuda": False, "torch": "2.11", "diffusers": "0.41", "vram_gb": None}
TOKENIZER_OK = {"sentencepiece", "protobuf", "tiktoken"}


def distilled_clip(tmp_path, **extra):
    spec_path, _ = make_spec(tmp_path, {"a": {"prompt": "p", "backend": DISTILLED, **extra}})
    return fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)[0]


def test_the_distilled_backend_defaults_to_its_fixed_schedule(tmp_path):
    clip = distilled_clip(tmp_path)
    assert clip.steps == len(fm.BACKENDS[DISTILLED]["timesteps"]) == 8


def test_the_distilled_schedule_cannot_be_overridden(tmp_path):
    """The checkpoint is timestep-distilled: other step counts are not a slower
    render, they are a wrong one."""
    try:
        distilled_clip(tmp_path, steps=20)
    except fm.SpecError as e:
        assert "8" in str(e)
    else:
        raise AssertionError("expected SpecError")


def test_the_manifest_records_the_timesteps(tmp_path):
    m = distilled_clip(tmp_path).manifest()
    assert m["timesteps"] == fm.BACKENDS[DISTILLED]["timesteps"]
    assert m["guidance_scale"] == 1.0


def test_a_cpu_render_does_not_ask_for_cuda(tmp_path):
    clip = distilled_clip(tmp_path)
    assert fm.check_backend(clip, CPU_REPORT, available=TOKENIZER_OK, device="cpu") == []


def test_a_backend_without_a_cpu_path_is_refused_on_cpu(tmp_path):
    spec_path, _ = make_spec(tmp_path)
    clip = fm.resolve_clips(fm.load_spec(spec_path), root=tmp_path)[0]
    blockers = fm.check_backend(clip, CPU_REPORT, available=TOKENIZER_OK, device="cpu")
    assert any("CPU" in b for b in blockers), blockers


def test_a_cpu_render_still_needs_torch(tmp_path):
    clip = distilled_clip(tmp_path)
    bare = {"cuda": False, "torch": None, "diffusers": None, "vram_gb": None}
    blockers = fm.check_backend(clip, bare, available=TOKENIZER_OK, device="cpu")
    assert any("torch" in b for b in blockers)
    assert not any("CUDA" in b for b in blockers), "cpu must not demand a CUDA device"


def test_a_cpu_render_refuses_before_importing_anything_heavy(tmp_path):
    clip = distilled_clip(tmp_path)
    bare = {"cuda": False, "torch": None, "diffusers": None, "vram_gb": None}
    try:
        fm.render(clip, tmp_path / "out", bare, device="cpu")
    except RuntimeError as e:
        assert "cannot render" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
    assert not (tmp_path / "out").exists()


def test_the_distilled_backend_renders_from_text_alone(tmp_path):
    spec = {"clips": {"t": {"prompt": "a dragon over a valley", "backend": DISTILLED}}}
    clip = fm.resolve_clips(spec, root=tmp_path)[0]
    assert clip.source is None
    m = clip.manifest()
    assert m["mode"] == "text-to-video" and m["source"] is None


def test_an_image_only_backend_still_needs_a_from(tmp_path):
    spec = {"clips": {"t": {"prompt": "p"}}}
    try:
        fm.resolve_clips(spec, root=tmp_path)
    except fm.SpecError as e:
        assert "from" in str(e)
    else:
        raise AssertionError("expected SpecError")


def test_plan_prints_a_text_clip(tmp_path):
    path = tmp_path / "motion.json"
    path.write_text(json.dumps({"clips": {"t": {"prompt": "rain", "backend": DISTILLED}}}))
    p = run_cli("--spec", str(path), "plan", cwd=tmp_path)
    assert p.returncode == 0, p.stderr
    assert "text-to-video" in p.stdout


FILM_SPEC = {
    "defaults": {"backend": DISTILLED, "size": [960, 416]},
    "cast": {"cat": "an orange tabby cat in a white astronaut suit"},
    "style": "35mm film, anamorphic.",
    "clips": {"s1": {"prompt": "{cat} wakes up.", "seconds": 4}},
}


def test_defaults_apply_to_every_shot(tmp_path):
    clip = fm.resolve_clips(FILM_SPEC, root=tmp_path)[0]
    assert clip.backend == DISTILLED and clip.size == [960, 416]


def test_cast_names_expand_word_for_word(tmp_path):
    """Consistency across cuts comes from repeating the exact same description."""
    clip = fm.resolve_clips(FILM_SPEC, root=tmp_path)[0]
    assert clip.prompt.startswith("an orange tabby cat in a white astronaut suit wakes up.")


def test_the_style_bible_ends_every_prompt(tmp_path):
    clip = fm.resolve_clips(FILM_SPEC, root=tmp_path)[0]
    assert clip.prompt.endswith("35mm film, anamorphic.")


def test_an_unknown_cast_name_is_a_spec_error(tmp_path):
    spec = {**FILM_SPEC, "clips": {"s1": {"prompt": "{dog} barks."}}}
    try:
        fm.resolve_clips(spec, root=tmp_path)
    except fm.SpecError as e:
        assert "dog" in str(e)
    else:
        raise AssertionError("expected SpecError")


def test_the_committed_film_resolves():
    spec = fm.load_spec(REPO / "prompts" / "films" / "one-small-step.json")
    shots = fm.resolve_clips(spec, root=REPO)
    assert len(shots) >= 10
    assert spec["film"]["title"]
    for s in shots:
        assert s.source is None and s.backend == DISTILLED
        assert all(v % 32 == 0 for v in s.size)
        assert "{" not in s.prompt, f"{s.item} has an unexpanded name"
    assert 55 <= sum(s.frames / s.fps for s in shots) <= 100, "a short film, not a clip"


def test_the_film_is_written_for_sound_too():
    spec = fm.load_spec(REPO / "prompts" / "films" / "one-small-step.json")
    for item, raw in spec["clips"].items():
        assert raw.get("sound", "").strip(), f"{item} has no sound line"
    score = spec["film"]["score"]
    assert 0 < len(score["caption"]) <= 512, "ACE-Step caps a caption at 512 characters"
    assert 30 <= score["bpm"] <= 300 and score["key"]


def test_mix_offsets_follow_the_cut(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    import mix
    shots = mix.offsets(REPO / "prompts" / "films" / "one-small-step.json")
    assert shots[0][1] == mix.CARD_SECONDS, "the first shot starts after the title card"
    for (_, s0, d0), (_, s1, _) in zip(shots, shots[1:]):
        assert abs(s0 + d0 - s1) < 1e-9, "each shot starts where the last one ends"


def test_mix_gain_brings_an_effect_to_the_target():
    sys.path.insert(0, str(REPO / "scripts"))
    import mix
    assert mix.gain_to(-24.0, (-3.6, -0.3)) == -20.4, "a shouting effect is turned down"
    assert mix.gain_to(-24.0, (-36.0, -20.0)) == 12.0, "a whisper is turned up"


def test_mix_gain_never_pushes_an_effect_past_its_peak_room():
    sys.path.insert(0, str(REPO / "scripts"))
    import mix
    # quiet on average but with a near-full-scale transient: the peak caps the gain
    assert mix.gain_to(-24.0, (-30.0, -1.0), peak_room=-3.0) == -2.0


def test_the_committed_spec_carries_the_runner_clips():
    clips = {c.item: c for c in fm.resolve_clips(fm.load_spec(REPO / fm.DEFAULT_SPEC), root=REPO)}
    smoke, full = clips["forgemaster-smoke"], clips["forgemaster-10s"]
    for c in (smoke, full):
        assert c.backend == DISTILLED
        assert all(v % 32 == 0 for v in c.size), "LTX pads anything off the 32 grid"
    assert smoke.frames == 49
    assert full.frames == 241


# ----------------------------------------------------------------------- cli

def test_plan_reports_the_frames_and_the_cost(tmp_path):
    spec_path, _ = make_spec(tmp_path)
    p = run_cli("--spec", str(spec_path), "plan", cwd=tmp_path)
    assert p.returncode == 0, p.stderr
    assert "73 frames" in p.stdout
    assert "VRAM" in p.stdout, "the cost must be visible before a download"


def test_plan_json_is_machine_readable(tmp_path):
    spec_path, _ = make_spec(tmp_path)
    p = run_cli("--spec", str(spec_path), "plan", "--json", cwd=tmp_path)
    assert p.returncode == 0, p.stderr
    data = json.loads(p.stdout)
    assert data[0]["frames"] == 73
    assert data[0]["frames_requested"] == 72
    assert data[0]["output"] == "ember-s0.mp4"


def test_plan_exits_nonzero_on_a_bad_spec(tmp_path):
    p = run_cli("--spec", str(tmp_path / "nope.json"), "plan", cwd=tmp_path)
    assert p.returncode == 2
    assert "no spec" in p.stderr


def test_check_is_honest_about_missing_gpu():
    p = run_cli("check")
    assert p.returncode in (0, 1), p.stderr
    assert "CUDA" in p.stdout


def test_render_requires_an_item():
    p = run_cli("render")
    assert p.returncode != 0
    assert "item" in (p.stdout + p.stderr)
