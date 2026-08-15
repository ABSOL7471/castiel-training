# Evaluating a personal coding fine-tune honestly

**TL;DR** — A rank-8 QLoRA on Qwen2.5-Coder-7B, trained on transcripts and code
from the author's own projects, was evaluated against its base on 35 held-out
tasks. The fine-tune's win is behavioral, not memorization: instruction
adherence 37% → 100%, style similarity +73% relative, exact match 0% for all
models. Merging the adapter into the weights recovers ~20% speed but costs ~28%
of the style gain to re-quantization; production kept the adapter. One clean
negative result is included.

## 1. Setup

- **Base**: Qwen/Qwen2.5-Coder-7B-Instruct, q4_K_M via Ollama.
- **Fine-tune**: QLoRA rank 8, alpha 16, max seq len 1024, ~2 epochs,
  completion-only loss (prompt tokens masked), trained in 4-bit on a single
  RTX 3060 12 GB.
- **Data**: instruction/input/output triples generated from real coding
  sessions and repositories ("fill in the missing section of this file, output
  only the missing code"), plus a small synthetic agent-workflow curriculum.
  A validation split was held out before training and never trained on.

## 2. Protocol

35 eligible validation rows (prompt ≤ 12,000 chars, non-empty reference).
Every model gets the byte-identical prompt used in training (same system
prompt, same instruction + input), via Ollama `/api/chat`, temperature 0,
fixed seed, `num_predict` 700. Models run sequentially — all tasks on one
model, then the next — so a 12 GB card never thrashes between models.

**Metrics** (per task, against the held-out reference):

| metric | definition |
|---|---|
| similarity | `difflib.SequenceMatcher` ratio on whitespace-normalized text; a fully-fenced answer is unwrapped first so the fence is judged separately from the code |
| exact_match | normalized output equals normalized reference |
| clean_output | no markdown fences and no prose preamble — the tasks all instruct "Output only the missing code", so this measures instruction adherence |
| latency / chars_per_sec | wall-clock per completion; output chars ÷ seconds |

## 3. Results

| model | mean similarity | exact match | clean output | avg latency | chars/sec |
|---|---|---|---|---|---|
| qwen2.5-coder:7b (base) | 0.073 | 0% | 37.1% | 5.6 s | 235 |
| castiel-tuned (runtime LoRA) | **0.126** | 0% | **100%** | 12.5 s | 186 |
| castiel-merged (baked-in, fresh q4_K_M) | 0.111 | 0% | **100%** | 10.1 s | **229** |

Per-task data: [castiel-eval-results.xlsx](castiel-eval-results.xlsx).

## 4. Analysis

**The gains are behavioral.** Absolute similarity is low for every model —
no one, human or model, reproduces bespoke code verbatim from a prefix. What
moved is *how* the model answers: it stopped wrapping answers in fences and
prose (37% → 100% adherence) and got measurably closer to the codebase's
house style (+73% relative). For a local coding agent, those behaviors were
the goal.

**0% exact match is a feature of the result.** If the fine-tune had scored
nonzero exact matches on held-out completions, the likeliest explanation
would be leakage between splits — memorization masquerading as skill. Its
absence, on a model trained directly on this codebase, is evidence the
adapter learned conventions rather than content.

**The speed/fidelity tradeoff is real and quantified.** The runtime adapter
costs ~20% per token vs. the plain base. Merging it into the weights (PEFT
fp16 merge → f16 GGUF → fresh q4_K_M) recovers nearly all of that speed —
but similarity drops 0.126 → 0.111. Applying a LoRA on top of an
already-quantized base and quantizing merged fp16 weights round differently;
the merged model keeps ~72% of the fine-tune's style improvement. Production
stayed on the adapter build: for agent coding work, fidelity beat ~2.4 s per
completion.

**A negative result, kept on purpose.** The initial speed hypothesis was
that the fine-tune's larger `num_ctx` (16384 vs. 4096) explained its
slowness. Tested at 8192: byte-identical outputs at identical speed.
KV-cache *allocation* costs VRAM, not speed, at these prompt lengths. The
remaining per-token gap was the adapter itself — which the merge experiment
then confirmed.

## 5. Limitations

- n = 35 is small; differences of a few thousandths in similarity are not
  meaningful. The adherence gap (37% vs. 100%) is far outside noise.
- Similarity to a single reference is a harsh, imperfect proxy for quality;
  a correct alternative implementation scores low.
- Latency was measured on one consumer GPU with models resident in VRAM;
  absolute numbers will not transfer, ratios should.
- All tasks come from one codebase and one author's style — that is the
  intended deployment, but nothing here claims general coding improvement.

## 6. Reproduce

```bash
python eval_harness.py --pilot        # 3 tasks, sanity check
python eval_harness.py --n 35         # full run, writes RESULTS.md + results.json
python eval_harness.py --models your-model:tag   # eval any Ollama model
```

The harness is pure Python stdlib. Raw per-task completions are written to
`results.json` locally; they are not published here because they reconstruct
private code.
