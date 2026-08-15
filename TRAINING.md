# Castiel LoRA Training Pipeline

Teach a local model your own codebases and the Castiel coding harness. Three stages, all offline.

## Why LoRA
Instead of retraining billions of weights, LoRA trains small adapter
matrices (~0.5-2% of parameters) on top of a frozen base model. Result:
a model that knows your project structure, naming conventions, and
patterns — trainable on a single consumer GPU in hours, not weeks.

## Quick start

```bash
pip install -r requirements-training.txt

# 1. Harvest your codebases into a dataset.
# By default this also adds the curated Castiel harness + scaffolding curriculum.
python prepare_dataset.py --repos ~/code/project-a ~/code/project-b --out dataset.jsonl

# Optional: train only on repository-derived code examples
python prepare_dataset.py --repos ~/code/project-a --out code-only.jsonl --no-harness

# 2. Train the adapter (defaults: Qwen2.5-Coder-7B, 4-bit, rank 16)
python train_lora.py --dataset dataset.jsonl --out adapters/castiel-lora

# 3. Merge + convert to GGUF + register with Ollama
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build && cd ..
python export_to_ollama.py --adapter adapters/castiel-lora --llama-cpp ./llama.cpp --name castiel-tuned

# 4. Use it
python ../castiel.py --model castiel-tuned
```

## Castiel harness curriculum

The dataset builder includes up to 24 curated examples by default. These are **not copied from your repositories**. They teach the model how Castiel actually operates: orient before editing, plan briefly, read files before modifying them, make one tool call per step, use exact edit arguments, recover from tool errors, verify work, summarize accurately, respect project-root boundaries, and scaffold small Python, CLI, web, configuration, database, test, and documentation changes in focused increments.

The examples use the same `read_file`, `write_file`, `edit_file`, `list_dir`, `search`, `run_command`, and `task_done` contracts as the live agent. The training formatter also includes a matching Castiel system prompt for every sample. This improves harness adherence without embedding private source code in the reusable curriculum.

Use `--harness-samples N` to adjust the curriculum size, or `--no-harness` when you explicitly want a code-only adapter. Keep the harness enabled for a coding-agent model; disable it only when training a completion-focused model.

## Hardware guide

| GPU | Base model | Settings |
|---|---|---|
| 8GB | Qwen2.5-Coder-1.5B | defaults |
| 12GB | Qwen2.5-Coder-7B (default) | defaults (4-bit) |
| 24GB | Qwen2.5-Coder-14B | `--base Qwen/Qwen2.5-Coder-14B-Instruct` |
| CPU only | 1.5B, small dataset | `--cpu` (slow — overnight runs) |

## Tuning knobs

- `--rank` 8 = light touch, 32-64 = stronger adaptation (more VRAM)
- `--epochs` 1-3; watch for loss flattening — more epochs on a small dataset overfits
- `--max-seq-len` raise to 4096 if your files are long and VRAM allows
- Dataset size: 500+ samples recommended; below ~200 the adapter mostly memorizes

## Loss and validation

Training uses **completion-only loss** by default: prompt tokens (the system
prompt and instruction) are masked out, so the adapter's whole capacity goes
into learning responses rather than memorizing prompts. Pass
`--full-sequence-loss` to restore the legacy behavior.

If `<dataset>.jsonl.val` exists (prepare_dataset.py writes it automatically),
it is evaluated at the end of every epoch so you can see overfitting as it
happens — rising eval loss with falling train loss means stop earlier or use a
lower rank. Point `--val-dataset` elsewhere to override.

The train/validation split is made at **source-file granularity**: all samples
derived from one file land on the same side, and files with identical contents
are harvested only once. Both keep the validation loss honest — a row-level
split would ask the model to "predict" halves of files it already saw in
training.

## What the dataset teaches

`prepare_dataset.py` generates three sample types per source file: explain (file comprehension), complete (style-consistent continuation), and fill-in-the-middle (surgical edits — the skill the agent loop uses most). It then mixes those with the selected harness curriculum before making the train/validation split. The command prints the repository-versus-harness composition so you can record exactly what trained an adapter.
