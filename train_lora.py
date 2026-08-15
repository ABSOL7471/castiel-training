#!/usr/bin/env python3
"""Castiel Training — Stage 2: LoRA fine-tuning.

Trains a LoRA adapter on the dataset from prepare_dataset.py so a base
model internalizes your codebase and conventions. Runs fully locally.

Recommended bases (HuggingFace IDs — these match the Ollama tags):
    Qwen/Qwen2.5-Coder-7B-Instruct     (default; trains on a 12GB GPU in 4-bit)
    Qwen/Qwen2.5-Coder-14B-Instruct    (24GB GPU in 4-bit)
    Qwen/Qwen2.5-Coder-1.5B-Instruct   (CPU/8GB experiments)

Install deps first:
    pip install -r requirements-training.txt

Usage:
    python train_lora.py --dataset dataset.jsonl --out adapters/castiel-lora
    python train_lora.py --base Qwen/Qwen2.5-Coder-14B-Instruct --epochs 2
"""

import argparse
import json
from pathlib import Path


CASTIEL_TRAINING_SYSTEM_PROMPT = """You are Castiel, a local coding agent working inside the user's project directory.
Work methodically: orient before changing code, briefly plan, make minimal edits, verify when possible, and finish with a concise summary. Use exactly one tool call at a time, never guess unread file contents, recover from tool errors by inspecting current state, and respect the project-root and user-approval boundaries enforced by the harness."""


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_messages(sample):
    """Return (prompt_messages, full_messages) for one dataset row.

    prompt_messages is everything the model sees before it must respond;
    full_messages appends the target assistant answer.
    """
    user = sample["instruction"]
    if sample.get("input"):
        user += "\n\n" + sample["input"]
    prompt = [
        {"role": "system", "content": CASTIEL_TRAINING_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    return prompt, prompt + [{"role": "assistant", "content": sample["output"]}]


def mask_prompt_labels(input_ids, prompt_length):
    """Labels for completion-only loss: prompt tokens are ignored (-100).

    Training the loss only on the assistant's answer stops the adapter from
    spending its capacity memorizing the (constant) system prompt and the
    instructions, which measurably improves instruction adherence.
    """
    boundary = max(0, min(prompt_length, len(input_ids)))
    return [-100] * boundary + list(input_ids[boundary:])


def main():
    ap = argparse.ArgumentParser(description="Train a LoRA adapter on your code dataset")
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--val-dataset", default=None,
                    help="Validation JSONL for eval during training "
                         "(default: <dataset>.val if it exists)")
    ap.add_argument("--full-sequence-loss", action="store_true",
                    help="Train on prompt tokens too (legacy behavior; "
                         "default is completion-only loss)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the latest checkpoint in <out>/checkpoints")
    ap.add_argument("--init-adapter", default=None,
                    help="Initialize LoRA weights from a saved adapter/checkpoint "
                         "directory and continue training (fresh optimizer/schedule; "
                         "use when full trainer-state resume is unavailable)")
    ap.add_argument("--bench-steps", type=int, default=None,
                    help="Benchmark mode: run exactly N optimizer steps, skip "
                         "eval and saving — for measuring step speed")
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="Disable gradient checkpointing (faster if VRAM allows)")
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--out", default="adapters/castiel-lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank (8-64 typical)")
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--no-4bit", action="store_true",
                    help="Disable 4-bit quantized training (needs much more VRAM)")
    ap.add_argument("--cpu", action="store_true", help="Force CPU training (slow)")
    ap.add_argument("--vram-fraction", type=float, default=0.90,
                    help="Cap this process at this fraction of GPU memory "
                         "(the rest stays free for the Windows desktop). "
                         "torch 2.13/transformers 5.15 needs ~10.8 GiB for "
                         "the 7B QLoRA run at 1024 tokens — 0.85 of a 12 GiB "
                         "card no longer fits it; 0.90 does, with "
                         "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256")
    args = ap.parse_args()

    # Heavy imports deferred so --help is instant
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, Trainer,
                              DataCollatorForSeq2Seq)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    use_cuda = torch.cuda.is_available() and not args.cpu
    use_4bit = use_cuda and not args.no_4bit
    if use_cuda:
        # Never take the whole card: leave VRAM for the Windows desktop
        # compositor. Exhausting VRAM starves the display driver and can
        # freeze the entire machine; with this cap, running out raises a
        # clean Python OOM error instead.
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction)

    print(f"base model:  {args.base}")
    print(f"device:      {'cuda (4-bit)' if use_4bit else 'cuda' if use_cuda else 'cpu'}")
    print(f"lora:        r={args.rank} alpha={args.alpha} dropout={args.dropout}")

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    elif use_cuda:
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.base, **model_kwargs)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Standard attention+MLP targets — correct for Qwen/Llama/Mistral-family
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    if args.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter,
                                          is_trainable=True)
        print(f"init:        LoRA weights loaded from {args.init_adapter}")
    else:
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    rows = load_jsonl(args.dataset)
    if not rows:
        raise ValueError(f"Dataset contains no training examples: {args.dataset}")
    harness_count = sum(1 for row in rows if row.get("source") == "castiel-harness")
    print(f"dataset:     {len(rows)} samples from {args.dataset}")
    print(f"curriculum:  {harness_count} Castiel harness / {len(rows) - harness_count} repository samples")

    truncated = 0
    fully_masked = 0

    def encode_rows(sample_rows):
        nonlocal truncated
        features = []
        for row in sample_rows:
            prompt_messages, full_messages = build_messages(row)
            full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
            input_ids = tokenizer(full_text, truncation=True,
                                  max_length=args.max_seq_len,
                                  add_special_tokens=False)["input_ids"]
            if len(tokenizer(full_text, add_special_tokens=False)["input_ids"]) > args.max_seq_len:
                truncated += 1
            if args.full_sequence_loss:
                labels = list(input_ids)
            else:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True)
                if not full_text.startswith(prompt_text):
                    # Template doesn't render prompts as a strict prefix —
                    # fall back to full-sequence loss for this sample.
                    labels = list(input_ids)
                else:
                    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                    labels = mask_prompt_labels(input_ids, len(prompt_ids))
            if not any(label != -100 for label in labels):
                # Truncation swallowed the whole answer — a fully-masked
                # sample teaches nothing and turns the loss into NaN.
                nonlocal fully_masked
                fully_masked += 1
                continue
            features.append({"input_ids": input_ids,
                             "attention_mask": [1] * len(input_ids),
                             "labels": labels})
        return features

    train_features = encode_rows(rows)
    if truncated:
        print(f"note:        {truncated} samples exceed --max-seq-len {args.max_seq_len} and were truncated")
    if fully_masked:
        print(f"note:        {fully_masked} samples dropped (prompt alone filled the "
              "sequence budget — nothing left to learn from)")
    print(f"loss:        {'full sequence' if args.full_sequence_loss else 'completion-only (prompt tokens masked)'}")
    ds = Dataset.from_list(train_features)

    val_path = args.val_dataset
    if val_path is None:
        candidate = Path(args.dataset + ".val")
        val_path = str(candidate) if candidate.is_file() else ""
    eval_ds = None
    if val_path:
        val_rows = load_jsonl(val_path)
        if val_rows:
            eval_ds = Dataset.from_list(encode_rows(val_rows))
            print(f"validation:  {len(val_rows)} samples from {val_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TrainingArguments gains/loses keywords across transformers major
    # versions (e.g. v5 dropped warmup_ratio); keep only what this install
    # accepts so the pipeline works on any supported version.
    import inspect
    desired_args = {
        "output_dir": str(out_dir / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        # Frequent saves + a keep-limit: an interruption (crash, reboot)
        # costs at most save_steps of work, and a *complete* checkpoint
        # always exists even if the newest save was cut off mid-write.
        "save_strategy": "steps",
        "save_steps": 25,
        "save_total_limit": 2,
        "eval_strategy": "epoch" if eval_ds is not None else "no",
        "bf16": use_cuda,
        "gradient_checkpointing": use_cuda and not args.no_grad_checkpoint,
        "report_to": [],
        "optim": "paged_adamw_8bit" if use_4bit else "adamw_torch",
    }
    if args.bench_steps:
        desired_args["max_steps"] = args.bench_steps
        desired_args["eval_strategy"] = "no"
        desired_args["save_strategy"] = "no"
        eval_ds = None
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = sorted(k for k in desired_args if k not in supported)
    if dropped:
        print(f"note:        this transformers version does not accept: {', '.join(dropped)} (skipped)")
    train_args = TrainingArguments(**{k: v for k, v in desired_args.items() if k in supported})

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100,
                                             padding=True),
    )

    trainer.train(resume_from_checkpoint=args.resume or None)

    if use_cuda:
        print(f"peak GPU memory: {torch.cuda.max_memory_reserved() / 2**30:.2f}"
              f" GiB reserved (cap was {args.vram_fraction:.2f} of "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB)")

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"\nadapter saved -> {out_dir}")
    print("next: python export_to_ollama.py --adapter", out_dir)


if __name__ == "__main__":
    main()
