# Castiel — fine-tune a local coding agent on your own code, on a 12 GB GPU

Castiel is a QLoRA fine-tune of Qwen2.5-Coder-7B-Instruct that learns **your**
codebase conventions and a disciplined coding-agent workflow — trained and served
entirely on a single consumer GPU (RTX 3060, 12 GB), fully offline. This repo is
the complete pipeline: dataset preparation, training, a reboot-surviving
supervisor, Ollama export, an adapter-merge pipeline, and an eval harness with
published results.

## Does it work? Measured, not vibed

35 held-out tasks (never trained on), temperature 0, identical prompts:

| model | mean similarity | exact match | clean output* | chars/sec |
|---|---|---|---|---|
| qwen2.5-coder:7b (base) | 0.073 | 0% | 37.1% | 235 |
| **castiel-tuned** (base + runtime LoRA) | **0.126** | 0% | **100%** | 186 |
| castiel-merged (LoRA baked in, fresh q4_K_M) | 0.111 | 0% | **100%** | 229 |

\* every task instructs "Output only the missing code"; clean = no markdown fences, no prose preamble.

The fine-tune's gains are **behavioral**: instruction adherence went 37% → 100%
and similarity to house style rose +73% relative. Exact match is 0% for *all*
models — nobody reproduces bespoke code verbatim, and a nonzero score here would
have meant data leakage, not skill. Full analysis, tradeoffs, and a negative
result we kept: [eval/EVAL.md](eval/EVAL.md). Per-task numbers:
[eval/castiel-eval-results.xlsx](eval/castiel-eval-results.xlsx).

## Pipeline

```
prepare_dataset.py      your repos -> instruction/input/output JSONL (+ synthetic agent curriculum)
train_lora.py           QLoRA (4-bit, completion-only loss) on a 12 GB card
train7b_supervisor.py   optional: survives reboots/crashes, resumes from checkpoints
export_to_ollama.py     adapter -> GGUF -> ollama model
merge_pipeline.py       optional: bake the adapter into the weights (see EVAL.md for the tradeoff)
eval/eval_harness.py    base vs fine-tune on your held-out split, stdlib only
```

Quickstart: see [TRAINING.md](TRAINING.md). The short version:

```bash
pip install -r requirements-training.txt
python prepare_dataset.py --repos ~/code/project-a ~/code/project-b --out dataset.jsonl
python train_lora.py --dataset dataset.jsonl --out adapters/castiel-lora \
    --max-seq-len 1024 --rank 8 --alpha 16          # the 12 GB recipe
python export_to_ollama.py --adapter adapters/castiel-lora --name castiel-tuned
python eval/eval_harness.py --n 35                   # measure before you believe
```

## Hardware notes (the 12 GB recipe)

- 7B QLoRA at 1024 tokens needs ~10.8 GiB with torch 2.13 / transformers 5.15:
  `--vram-fraction 0.90` plus `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256`.
- The trainer caps its own VRAM so the desktop compositor never starves —
  running out raises a clean Python OOM instead of freezing the machine.
- Training and inference can't share the card: the supervisor evicts Ollama
  models before each run.

## What is deliberately not here

- **No datasets and no session transcripts** — they are generated from private
  codebases. Run `prepare_dataset.py` on your own repos; that's the point.
- **No trained weights** — an adapter trained on private code can echo fragments
  of it. The eval's 0% exact-match rate is evidence *against* memorization, but
  publishing weights is a separate decision from publishing the pipeline.
- Raw eval completions (they reconstruct private code); the spreadsheet carries
  the per-task metrics instead.

## Provenance

Designed and operated as part of a personal fleet of local-first AI systems;
developed AI-paired (Claude Code alongside the very model being trained).
The safety stance throughout: capability comes from the model, boundaries come
from the harness.

## License

MIT — see [LICENSE](LICENSE).
