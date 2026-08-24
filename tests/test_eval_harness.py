"""Unit tests for the pure scoring functions in eval/eval_harness.py.

No Ollama, no network: only normalize / strip_fences / score are exercised.
"""

import eval_harness as eh


def test_normalize_collapses_blank_lines_and_trailing_space():
    raw = "def f():   \n\n    return 1  \n\n\n"
    assert eh.normalize(raw) == "def f():\n    return 1"


def test_normalize_empty_input():
    assert eh.normalize("") == ""
    assert eh.normalize("\n  \n\t\n") == ""


def test_strip_fences_unwraps_single_fenced_block():
    fenced = "```python\nx = 1\ny = 2\n```"
    assert eh.strip_fences(fenced) == "x = 1\ny = 2"


def test_strip_fences_handles_fence_without_language_and_padding():
    fenced = "  ```\ncode here\n```  "
    assert eh.strip_fences(fenced) == "code here"


def test_strip_fences_leaves_unfenced_and_mixed_text_alone():
    plain = "x = 1\ny = 2"
    assert eh.strip_fences(plain) == plain
    mixed = "Here is the code:\n```python\nx = 1\n```"
    assert eh.strip_fences(mixed) == mixed  # not a pure fenced block


def test_score_exact_match_is_clean_and_similarity_one():
    ref = "def add(a, b):\n    return a + b"
    s = eh.score(ref, ref)
    assert s["exact"] is True
    assert s["clean"] is True
    assert s["similarity"] == 1.0


def test_score_fenced_output_still_matches_code_but_is_not_clean():
    ref = "def add(a, b):\n    return a + b"
    s = eh.score(f"```python\n{ref}\n```", ref)
    assert s["exact"] is True  # fence unwrapped before comparing
    assert s["clean"] is False  # but penalized as not-clean
    assert s["similarity"] == 1.0


def test_score_prose_preamble_is_not_clean():
    ref = "x = 1"
    s = eh.score("Here is the code you asked for:\nx = 1", ref)
    assert s["clean"] is False


def test_score_dissimilar_outputs_score_low():
    s = eh.score("completely unrelated words", "def f():\n    return 42")
    assert 0.0 <= s["similarity"] < 0.5
    assert s["exact"] is False


def test_score_reports_lengths():
    s = eh.score("abc", "abcdef")
    assert s["out_chars"] == 3
    assert s["ref_chars"] == 6
