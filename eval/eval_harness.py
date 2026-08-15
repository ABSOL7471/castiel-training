#!/usr/bin/env python3
"""Castiel eval harness — base model vs fine-tune, on held-out tasks.

Measures whether the LoRA actually bought anything, with numbers a
stranger can check. Tasks come from the SAME validation split the trainer
held out (dataset-v2-full.jsonl.val) — none of these rows were trained on.

Both models get the IDENTICAL prompt the trainer used (system prompt +
instruction + input), sent via Ollama's /api/chat with temperature 0, so
the only variable is the adapter. Passing an explicit system message
overrides the Modelfile's persona SYSTEM — deliberately, for fairness.

Metrics per task, against the held-out reference completion:
  similarity   difflib ratio on whitespace-normalized text (0..1)
  exact        normalized exact match
  clean        output is raw code as instructed — no markdown fences,
               no "Here is..." prose preamble (a trained behavior:
               every task says "Output only the missing code")
  latency      wall seconds per completion

Usage:
    python eval_harness.py --pilot          3 tasks, sanity check
    python eval_harness.py                  full run (default N=40)
    python eval_harness.py --n 60 --seed 7

Writes results.json (raw, every completion kept for audit) and
RESULTS.md (the table) next to this script. Pure Python stdlib -
no dependencies to install.
"""

import argparse
import difflib
import json
import re
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
VAL_FILE = HERE.parent / "dataset-v2-full.jsonl.val"
OLLAMA = "http://127.0.0.1:11434/api/chat"
MODELS = ["qwen2.5-coder:7b", "castiel-tuned:latest"]

# Must byte-match training (train_lora.py) or the comparison is unfair.
SYSTEM_PROMPT = """You are Castiel, a local coding agent working inside the user's project directory.
Work methodically: orient before changing code, briefly plan, make minimal edits, verify when possible, and finish with a concise summary. Use exactly one tool call at a time, never guess unread file contents, recover from tool errors by inspecting current state, and respect the project-root and user-approval boundaries enforced by the harness."""

# Training used max-seq-len 1024; prompts far beyond the context the model
# was tuned at measure truncation behavior, not the adapter. ~4 chars/token
# is a fair heuristic for code.
MAX_PROMPT_CHARS = 12000
NUM_PREDICT = 700


def load_tasks(n, seed):
    rows = []
    with VAL_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    eligible = []
    for i, r in enumerate(rows):
        user = r["instruction"] + ("\n\n" + r["input"] if r.get("input") else "")
        if len(user) <= MAX_PROMPT_CHARS and r.get("output", "").strip():
            eligible.append({"id": i, "user": user, "reference": r["output"]})
    # Deterministic shuffle without random-module version drift: sort by a
    # seed-keyed hash of the row index.
    eligible.sort(key=lambda t: hash((seed, t["id"])))
    picked = eligible[:n]
    return picked, len(rows), len(eligible)


def generate(model, user_content):
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "options": {"temperature": 0, "num_predict": NUM_PREDICT, "seed": 42},
    }
    req = urllib.request.Request(
        OLLAMA, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read())
    return body["message"]["content"], time.monotonic() - start


def normalize(text):
    """Collapse whitespace so formatting noise doesn't drown signal."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines() if line.strip())


FENCE = re.compile(r"^\s*```")
PREAMBLE = re.compile(
    r"^\s*(here('s| is)|sure|certainly|the missing|this (code|section)|below is|i('ll| will))",
    re.IGNORECASE)


def strip_fences(text):
    """If the whole answer is one fenced block, unwrap it before scoring —
    we penalize the fence separately; similarity should judge the code."""
    m = re.match(r"^\s*```[a-zA-Z0-9_+-]*\n(.*?)\n?```\s*$", text, re.DOTALL)
    return m.group(1) if m else text


def score(output, reference):
    fenced = bool(FENCE.search(output))
    preamble = bool(PREAMBLE.search(output))
    code = strip_fences(output)
    a, b = normalize(code), normalize(reference)
    sim = difflib.SequenceMatcher(None, a, b).ratio()
    return {
        "similarity": round(sim, 4),
        "exact": a == b,
        "clean": not fenced and not preamble,
        "out_chars": len(output),
        "ref_chars": len(reference),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pilot", action="store_true", help="3 tasks, quick shakeout")
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()
    n = 3 if args.pilot else args.n

    tasks, total, eligible = load_tasks(n, args.seed)
    print(f"[eval] {len(tasks)} tasks (val file: {total} rows, {eligible} eligible) "
          f"seed={args.seed} vs models: {', '.join(args.models)}")

    # All tasks per model, models sequentially — one GPU swap total,
    # instead of thrashing a 12GB card on every task.
    results = {m: [] for m in args.models}
    for model in args.models:
        print(f"[eval] --- {model} ---")
        for i, t in enumerate(tasks, 1):
            try:
                output, secs = generate(model, t["user"])
            except Exception as e:               # noqa: BLE001 — record, keep going
                print(f"[eval]  {i}/{len(tasks)} id={t['id']} FAILED: {e}")
                results[model].append({"id": t["id"], "error": str(e)})
                continue
            s = score(output, t["reference"])
            s.update({"id": t["id"], "latency_s": round(secs, 1), "output": output})
            results[model].append(s)
            print(f"[eval]  {i}/{len(tasks)} id={t['id']} sim={s['similarity']:.2f} "
                  f"exact={s['exact']} clean={s['clean']} {secs:.0f}s")

    # ── aggregate ──
    summary = {}
    for model, rows in results.items():
        ok = [r for r in rows if "error" not in r]
        if not ok:
            summary[model] = {"completed": 0}
            continue
        lat = sorted(r["latency_s"] for r in ok)
        summary[model] = {
            "completed": len(ok),
            "errors": len(rows) - len(ok),
            "mean_similarity": round(sum(r["similarity"] for r in ok) / len(ok), 4),
            "exact_match_pct": round(100 * sum(r["exact"] for r in ok) / len(ok), 1),
            "clean_output_pct": round(100 * sum(r["clean"] for r in ok) / len(ok), 1),
            "median_latency_s": lat[len(lat) // 2],
        }

    stamp = time.strftime("%Y-%m-%d %H:%M")
    (HERE / "results.json").write_text(
        json.dumps({"ran_at": stamp, "seed": args.seed, "n": len(tasks),
                    "val_file": VAL_FILE.name, "summary": summary,
                    "tasks": [t["id"] for t in tasks], "results": results},
                   indent=2), encoding="utf-8")

    lines = [
        "# Castiel eval — base vs fine-tune", "",
        f"Ran {stamp} · {len(tasks)} held-out tasks from `{VAL_FILE.name}` "
        f"(never trained on) · temperature 0 · identical prompts", "",
        "| model | mean similarity | exact match | clean output* | median latency |",
        "|---|---|---|---|---|",
    ]
    for model, s in summary.items():
        if s.get("completed"):
            lines.append(f"| {model} | {s['mean_similarity']:.3f} | {s['exact_match_pct']}% "
                         f"| {s['clean_output_pct']}% | {s['median_latency_s']}s |")
        else:
            lines.append(f"| {model} | — | — | — | all {len(results[model])} failed |")
    lines += ["",
              "\\* every task instructs “Output only the missing code”; "
              "clean = no markdown fences, no prose preamble.",
              "",
              "Per-task outputs kept in `results.json` for audit.", ""]
    (HERE / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval] wrote {HERE / 'RESULTS.md'}")
    for model, s in summary.items():
        print(f"[eval] {model}: {s}")


if __name__ == "__main__":
    main()
