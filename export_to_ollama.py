#!/usr/bin/env python3
"""Castiel Training — Stage 3: export the LoRA back into Ollama.

Merges the trained adapter into the base model, converts to GGUF via
llama.cpp, and registers the result as a new Ollama model — so Castiel
can run your fine-tuned model exactly like any other tag.

Requires llama.cpp cloned locally for the GGUF conversion step:
    git clone https://github.com/ggerganov/llama.cpp

Usage:
    python export_to_ollama.py --adapter adapters/castiel-lora \\
        --llama-cpp ~/llama.cpp --name castiel-tuned
Then:
    python ../castiel.py --model castiel-tuned
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Resolve the sibling module even when this script is executed via runpy or
# from another working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lora import CASTIEL_TRAINING_SYSTEM_PROMPT  # noqa: E402

# The registered model must run behind the SAME system prompt it was
# fine-tuned with — a mismatched prompt undoes the harness curriculum.
MODELFILE_TEMPLATE = '''FROM {gguf_path}
PARAMETER temperature 0.2
PARAMETER num_ctx 16384
SYSTEM """{system_prompt}"""
'''


def run(cmd, **kw):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kw)
    if result.returncode != 0:
        sys.exit(f"step failed with exit {result.returncode}")


def main():
    ap = argparse.ArgumentParser(description="Merge LoRA and register with Ollama")
    ap.add_argument("--adapter", required=True, help="Adapter dir from train_lora.py")
    ap.add_argument("--base", default=None,
                    help="Base model HF id (default: read from adapter config)")
    ap.add_argument("--llama-cpp", required=True, help="Path to a llama.cpp checkout")
    ap.add_argument("--name", default="castiel-tuned", help="Ollama model name to create")
    ap.add_argument("--quant", default="Q4_K_M",
                    help="GGUF quantization (Q4_K_M, Q5_K_M, Q8_0, F16)")
    args = ap.parse_args()

    adapter = Path(args.adapter).resolve()
    llama_cpp = Path(args.llama_cpp).expanduser().resolve()
    merged_dir = adapter.parent / (adapter.name + "-merged")
    gguf_f16 = adapter.parent / f"{args.name}-f16.gguf"
    gguf_final = adapter.parent / f"{args.name}-{args.quant}.gguf"

    # ---- 1. Merge adapter into base weights ----
    print("\n[1/4] merging adapter into base model...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig

    base_id = args.base or PeftConfig.from_pretrained(adapter).base_model_name_or_path
    print(f"base: {base_id}")

    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    model.save_pretrained(str(merged_dir))
    AutoTokenizer.from_pretrained(base_id, trust_remote_code=True) \
        .save_pretrained(str(merged_dir))
    print(f"merged -> {merged_dir}")

    # ---- 2. Convert to GGUF (F16) ----
    print("\n[2/4] converting to GGUF...")
    convert = llama_cpp / "convert_hf_to_gguf.py"
    if not convert.exists():
        convert = llama_cpp / "convert-hf-to-gguf.py"  # older naming
    run([sys.executable, str(convert), str(merged_dir),
         "--outfile", str(gguf_f16), "--outtype", "f16"])

    # ---- 3. Quantize ----
    print("\n[3/4] quantizing...")
    quant_bin = None
    for candidate in ["build/bin/llama-quantize", "llama-quantize", "quantize"]:
        if (llama_cpp / candidate).exists():
            quant_bin = llama_cpp / candidate
            break
    if quant_bin and args.quant.upper() != "F16":
        run([str(quant_bin), str(gguf_f16), str(gguf_final), args.quant])
    else:
        gguf_final = gguf_f16
        if args.quant.upper() != "F16":
            print("llama-quantize binary not found (build llama.cpp first) — "
                  "keeping F16 GGUF")

    # ---- 4. Register with Ollama ----
    print("\n[4/4] registering with Ollama...")
    modelfile = adapter.parent / "Modelfile"
    modelfile.write_text(
        MODELFILE_TEMPLATE.format(gguf_path=gguf_final,
                                  system_prompt=CASTIEL_TRAINING_SYSTEM_PROMPT),
        encoding="utf-8",
    )
    run(["ollama", "create", args.name, "-f", str(modelfile)])

    print(f"\ndone. Run your fine-tuned model with:")
    print(f"    python castiel.py --model {args.name}")


if __name__ == "__main__":
    main()
