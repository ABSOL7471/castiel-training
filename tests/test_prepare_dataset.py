"""Unit tests for the pure dataset-preparation logic in prepare_dataset.py."""

import argparse
import json
import random

import pytest

import prepare_dataset as pd


PY_FILE = "".join(f"line_{i} = {i}\n" for i in range(20))  # 20 lines, >=12


def test_make_samples_long_file_yields_three_categories():
    rng = random.Random(0)
    samples = pd.make_samples("src/app.py", PY_FILE, rng, group="repo/src/app.py")
    assert [s["category"] for s in samples] == ["explain", "complete", "fill_in_middle"]
    assert all(s["group"] == "repo/src/app.py" for s in samples)
    assert all(s["source"] == "repository" for s in samples)


def test_make_samples_short_file_yields_only_explain():
    rng = random.Random(0)
    short = "a = 1\nb = 2\n"  # fewer than 12 lines
    samples = pd.make_samples("x.py", short, rng, group="g")
    assert [s["category"] for s in samples] == ["explain"]
    assert short in samples[0]["output"]


def test_make_samples_language_detection():
    rng = random.Random(0)
    samples = pd.make_samples("main.rs", PY_FILE, rng, group="g")
    assert "Rust" in samples[0]["instruction"] or "Rust" in samples[0]["output"]
    unknown = pd.make_samples("weird.zig", PY_FILE, rng, group="g")
    # .zig is harvestable but has no language name mapped; falls back to "code"
    assert "code" in unknown[0]["output"]


def test_make_samples_completion_splits_reconstruct_source():
    rng = random.Random(7)
    samples = pd.make_samples("app.py", PY_FILE, rng, group="g")
    complete = next(s for s in samples if s["category"] == "complete")
    assert complete["input"] + complete["output"] == PY_FILE


def test_make_samples_fill_in_middle_reconstructs_source():
    rng = random.Random(7)
    samples = pd.make_samples("app.py", PY_FILE, rng, group="g")
    fim = next(s for s in samples if s["category"] == "fill_in_middle")
    prefix = fim["input"].split("<prefix>\n", 1)[1].split("\n</prefix>", 1)[0]
    suffix = fim["input"].split("<suffix>\n", 1)[1].split("\n</suffix>", 1)[0]
    assert prefix + fim["output"] + suffix == PY_FILE
    assert fim["output"]  # the masked middle is never empty


def test_percentage_validator_accepts_valid_and_rejects_out_of_range():
    assert pd.percentage("0") == 0.0
    assert pd.percentage("0.05") == pytest.approx(0.05)
    with pytest.raises(argparse.ArgumentTypeError):
        pd.percentage("1")
    with pytest.raises(argparse.ArgumentTypeError):
        pd.percentage("-0.1")


def test_positive_int_validator():
    assert pd.positive_int("64") == 64
    with pytest.raises(argparse.ArgumentTypeError):
        pd.positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError):
        pd.positive_int("-3")


def test_iter_code_files_filters(tmp_path):
    big_enough = "# padding\n" * 40  # > 200 bytes
    (tmp_path / "keep.py").write_text(big_enough, encoding="utf-8")
    (tmp_path / "notes.txt").write_text(big_enough, encoding="utf-8")  # wrong ext
    (tmp_path / "tiny.py").write_text("x=1\n", encoding="utf-8")  # < 200 bytes
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text(big_enough, encoding="utf-8")
    (tmp_path / ".secret").mkdir()
    (tmp_path / ".secret" / "hidden.py").write_text(big_enough, encoding="utf-8")
    (tmp_path / "huge.py").write_text("# x\n" * 30000, encoding="utf-8")  # > 64 KB

    found = {rel.as_posix() for rel, _ in pd.iter_code_files(tmp_path, max_kb=64)}
    assert found == {"keep.py"}


def test_write_jsonl_roundtrip(tmp_path):
    rows = [{"instruction": "a", "output": "b"}, {"instruction": "é", "output": ""}]
    out = tmp_path / "nested" / "data.jsonl"
    pd.write_jsonl(out, rows)
    read_back = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert read_back == rows
