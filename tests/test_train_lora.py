"""Unit tests for the GPU-free helpers in train_lora.py.

torch/transformers are imported inside main(), so importing the module and
testing build_messages / mask_prompt_labels / load_jsonl needs no GPU deps.
"""

import train_lora as tl


def test_build_messages_without_input():
    sample = {"instruction": "Do X", "input": "", "output": "done"}
    prompt, full = tl.build_messages(sample)
    assert [m["role"] for m in prompt] == ["system", "user"]
    assert prompt[0]["content"] == tl.CASTIEL_TRAINING_SYSTEM_PROMPT
    assert prompt[1]["content"] == "Do X"
    assert full == prompt + [{"role": "assistant", "content": "done"}]


def test_build_messages_appends_input_to_instruction():
    sample = {"instruction": "Continue", "input": "code head", "output": "tail"}
    prompt, _ = tl.build_messages(sample)
    assert prompt[1]["content"] == "Continue\n\ncode head"


def test_mask_prompt_labels_masks_prompt_and_keeps_completion():
    ids = [10, 11, 12, 13, 14]
    labels = tl.mask_prompt_labels(ids, prompt_length=3)
    assert labels == [-100, -100, -100, 13, 14]


def test_mask_prompt_labels_clamps_boundary():
    ids = [1, 2, 3]
    assert tl.mask_prompt_labels(ids, prompt_length=99) == [-100, -100, -100]
    assert tl.mask_prompt_labels(ids, prompt_length=0) == [1, 2, 3]
    assert tl.mask_prompt_labels(ids, prompt_length=-5) == [1, 2, 3]


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"b": 2}\n', encoding="utf-8")
    assert tl.load_jsonl(path) == [{"a": 1}, {"b": 2}]
