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
