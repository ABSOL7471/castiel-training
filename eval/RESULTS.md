# Castiel eval — base vs fine-tune vs merged

35 held-out tasks from `dataset-v2-full.jsonl.val` (never trained on) · temperature 0 · identical prompts · runs on 2026-08-15

| model | mean similarity | exact match | clean output* | avg latency | chars/sec |
|---|---|---|---|---|---|
| qwen2.5-coder:7b (base) | 0.073 | 0% | 37.1% | 5.6s | 235 |
| castiel-tuned (base + runtime LoRA) | **0.126** | 0% | **100%** | 12.5s | 186 |
| castiel-merged (LoRA baked in, fresh q4_K_M) | 0.111 | 0% | **100%** | 10.1s | **229** |

\* every task instructs "Output only the missing code"; clean = no markdown fences, no prose preamble.

## Findings

- The fine-tune's gains are behavioral: instruction adherence 37% → 100%, similarity to
  house style +73% relative. Exact match is 0% for **all** models — nobody reproduces
  bespoke code verbatim, and a nonzero score here would have meant data leakage, not skill.
- The runtime LoRA adapter costs ~20% per token. An earlier experiment proved `num_ctx`
  (16384 vs 8192) changes nothing: byte-identical outputs at identical speed — KV-cache
  *allocation* is VRAM, not speed, at these prompt lengths.
- Merging the adapter into the weights (PEFT fp16 merge → f16 GGUF → q4_K_M) recovers
  nearly all the speed (186 → 229 chars/sec) but re-quantization costs fidelity:
  similarity drops 0.126 → 0.111, i.e. the merged model keeps ~72% of the fine-tune's
  style improvement. Applying a LoRA on top of a quantized base and quantizing merged
  fp16 weights round differently; the eval made that tradeoff visible.

## Decision (2026-08-15)

`castiel-tuned:latest` stays on the runtime-adapter build: for an agent doing real coding
work, full style fidelity beats ~2.4s per completion. `castiel-merged` remains installed
for A/B testing (weights: `castiel-merged-7b-q4km.gguf`).

Raw per-task outputs: `results.json` (canonical, base + adapter run),
`results-merged.json` (merged run). Harness: `eval_harness.py`.
