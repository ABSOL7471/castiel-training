"""Tests for the supervisor's checkpoint-resume logic and the harness curriculum."""

import random

import harness_examples as he
import train7b_supervisor as sup


def _make_checkpoint(root, step, with_weights=True):
    d = root / "checkpoints" / f"checkpoint-{step}"
    d.mkdir(parents=True)
    if with_weights:
        (d / "adapter_model.safetensors").write_bytes(b"\x00")
    return d


def test_latest_checkpoint_picks_highest_step(tmp_path, monkeypatch):
    monkeypatch.setattr(sup, "OUT", tmp_path)
    _make_checkpoint(tmp_path, 50)
    best = _make_checkpoint(tmp_path, 150)
    _make_checkpoint(tmp_path, 100)
    step, path = sup.latest_checkpoint()
    assert step == 150
    assert path == best


def test_latest_checkpoint_ignores_incomplete_and_malformed(tmp_path, monkeypatch):
    monkeypatch.setattr(sup, "OUT", tmp_path)
    _make_checkpoint(tmp_path, 200, with_weights=False)  # no safetensors -> unusable
    (tmp_path / "checkpoints" / "checkpoint-abc").mkdir()  # bad name
    good = _make_checkpoint(tmp_path, 40)
    step, path = sup.latest_checkpoint()
    assert (step, path) == (40, good)


def test_latest_checkpoint_none_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(sup, "OUT", tmp_path)
    step, path = sup.latest_checkpoint()
    assert (step, path) == (0, None)


def test_select_harness_samples_clamps_and_does_not_mutate():
    all_samples = he.select_harness_samples()
    assert len(all_samples) > 0
    assert he.select_harness_samples(5) == he.harness_samples()[:5]
    assert len(he.select_harness_samples(10 ** 6)) == len(all_samples)
    assert he.select_harness_samples(-3) == []


def test_select_harness_samples_shuffle_is_deterministic_per_seed():
    a = he.select_harness_samples(8, random.Random(7))
    b = he.select_harness_samples(8, random.Random(7))
    assert a == b
    assert len(a) == 8


def test_harness_samples_have_training_schema():
    for sample in he.harness_samples():
        assert set(sample) >= {"instruction", "input", "output", "source", "category"}
        assert sample["instruction"].strip()
        assert sample["output"].strip()
