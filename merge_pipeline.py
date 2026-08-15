#!/usr/bin/env python3
"""Merge the Castiel LoRA into the base weights and rebuild the model.

Why: Ollama applies the adapter at runtime, which costs ~20% per token
(measured: 186 vs 235 chars/sec against the plain base). Baking the
adapter into the weights removes that overhead; the model's outputs
should be near-identical, which the final eval stage verifies.

Pipeline (each stage updates merge-status.json so progress is visible):
  1. load base fp16 from the HF cache + PEFT merge_and_unload
  2. save merged HF model
  3. convert_hf_to_gguf -> f16 GGUF (~15 GB intermediate)
  4. llama-quantize -> q4_K_M (matches the base's quantization)
  5. ollama create castiel-merged
  6. eval_harness --models castiel-merged  (same 35 held-out tasks)

Intermediates live in _merge_work/ and are deleted on success; the final
q4 GGUF is kept next to the adapter. Run with `python -u`.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "_merge_work"
STATUS = HERE / "merge-status.json"
BASE_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER = HERE / "adapters" / "castiel-lora-7b"
CONVERT = HERE / "llama.cpp" / "convert_hf_to_gguf.py"
QUANTIZE = HERE / "llama-bin" / "llama-quantize.exe"
F16_GGUF = WORK / "castiel-merged-f16.gguf"
Q4_GGUF = HERE / "castiel-merged-7b-q4km.gguf"
MODELFILE = HERE / "Modelfile-castiel-7b-merged"
EVAL = HERE / "eval" / "eval_harness.py"

STAGES = [
    "load base + merge adapter",
    "save merged model",
    "convert to f16 GGUF",
    "quantize to q4_K_M",
    "create ollama model",
    "eval on 35 held-out tasks",
]


def status(stage_idx, state="running", message=""):
    STATUS.write_text(json.dumps({
        "stage": stage_idx + 1, "stagesTotal": len(STAGES),
        "label": STAGES[stage_idx] if stage_idx < len(STAGES) else "done",
        "state": state, "message": message,
        "updatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")
    print(f"[merge] stage {stage_idx + 1}/{len(STAGES)}: "
          f"{STAGES[stage_idx] if stage_idx < len(STAGES) else 'done'} {state} {message}",
          flush=True)


def run(cmd, stage_idx):
    print(f"[merge] $ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    tail = (r.stdout or "")[-2000:] + (r.stderr or "")[-2000:]
    print(tail, flush=True)
    if r.returncode != 0:
        status(stage_idx, "error", f"exit {r.returncode}: {tail[-300:]}")
        sys.exit(1)
    return r


def main():
    if not (ADAPTER / "adapter_model.safetensors").is_file():
        status(0, "error", f"no adapter at {ADAPTER}")
        sys.exit(1)
    WORK.mkdir(exist_ok=True)
    merged_dir = WORK / "merged-hf"

    # ── 1. merge in fp16 on CPU (48 GB RAM, model is ~15 GB) ──
    status(0)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ID, dtype=torch.float16, low_cpu_mem_usage=True,
        local_files_only=True)
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model = model.merge_and_unload()

    # ── 2. save ──
    status(1)
    model.save_pretrained(str(merged_dir), safe_serialization=True,
                          max_shard_size="5GB")
    tok = AutoTokenizer.from_pretrained(BASE_ID, local_files_only=True)
    tok.save_pretrained(str(merged_dir))
    del model
    del tok

    # ── 3. HF -> f16 GGUF ──
    status(2)
    run([sys.executable, CONVERT, merged_dir, "--outfile", F16_GGUF,
         "--outtype", "f16"], 2)

    # ── 4. quantize to the same scheme the base blob uses ──
    status(3)
    run([QUANTIZE, F16_GGUF, Q4_GGUF, "q4_K_M"], 3)

    # ── 5. ollama model, same persona/params as the adapter build ──
    status(4)
    MODELFILE.write_text(
        f"FROM {Q4_GGUF}\n"
        "PARAMETER temperature 0.2\n"
        "PARAMETER num_ctx 16384\n"
        'SYSTEM """You are Castiel, a local coding agent working inside the user\'s project directory.\n'
        "Work methodically: orient before changing code, briefly plan, make minimal edits, verify when possible, "
        "and finish with a concise summary. Use exactly one tool call at a time, never guess unread file contents, "
        "recover from tool errors by inspecting current state, and respect the project-root and user-approval "
        'boundaries enforced by the harness."""\n', encoding="utf-8")
    ollama = shutil.which("ollama")
    if not ollama:
        status(4, "error", "ollama not found on PATH")
        sys.exit(1)
    run([ollama, "create", "castiel-merged", "-f", MODELFILE], 4)

    # ── 6. same 35 held-out tasks; similarity must hold or the merge lied ──
    status(5)
    run([sys.executable, "-u", EVAL, "--n", "35", "--models", "castiel-merged"], 5)

    shutil.rmtree(WORK, ignore_errors=True)
    status(len(STAGES) - 1, "done", "pipeline complete; intermediates removed")


if __name__ == "__main__":
    main()
