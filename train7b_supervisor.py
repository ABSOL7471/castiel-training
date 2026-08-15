#!/usr/bin/env python3
"""Reboot-surviving supervisor for a long Castiel LoRA training run.

Put a shortcut to this in the Startup folder (or a systemd unit on Linux).
On every boot (and after every crash) it:
  1. exits immediately if training already finished (DONE marker);
  2. finds the newest usable checkpoint under the adapter output dir;
  3. relaunches training, initializing from that checkpoint's weights and
     scaling --epochs down to roughly the remaining work;
  4. on successful completion writes the DONE marker and stops.

transformers v5 checkpoints carry no trainer_state.json, so instead of
native resume this uses --init-adapter (weights carry over; the optimizer
and LR schedule restart over the remaining epochs — a sound approximation).

Adjust the constants below for your run: STEPS_PER_EPOCH comes from the
first epoch's logs (dataset size / effective batch size).
"""

import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOME = Path(__file__).resolve().parent
OUT = HOME / "adapters" / "castiel-lora-7b"
DONE = OUT / "TRAINING_DONE.marker"
TRAIN = HOME / "train_lora.py"
LOG = HOME / "train7b.log"
ERRLOG = HOME / "train7b.err.log"
STEPS_PER_EPOCH = 87.0
TOTAL_EPOCHS = 2.0
PYTHON = sys.executable.replace("pythonw.exe", "python.exe")

# Models to evict from the GPU before training starts — training and
# inference cannot share a 12 GB card.
OLLAMA_MODELS_TO_UNLOAD = ("castiel-tuned:latest",)


def latest_checkpoint():
    best_step, best_path = 0, None
    for candidate in (OUT / "checkpoints").glob("checkpoint-*"):
        match = re.match(r"checkpoint-(\d+)$", candidate.name)
        if not match or not (candidate / "adapter_model.safetensors").is_file():
            continue
        step = int(match.group(1))
        if step > best_step:
            best_step, best_path = step, candidate
    return best_step, best_path


def unload_ollama():
    for model in OLLAMA_MODELS_TO_UNLOAD:
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/generate",
                data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
        except OSError:
            pass


def main():
    while True:
        if DONE.is_file():
            return
        if (OUT / "adapter_model.safetensors").is_file():
            DONE.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            return
        unload_ollama()
        step, checkpoint = latest_checkpoint()
        remaining = max(0.15, TOTAL_EPOCHS - step / STEPS_PER_EPOCH)
        command = [PYTHON, str(TRAIN),
                   "--dataset", str(HOME / "dataset.jsonl"),
                   "--val-dataset", str(HOME / "dataset.jsonl.val"),
                   "--base", "Qwen/Qwen2.5-Coder-7B-Instruct",
                   "--out", str(OUT),
                   "--max-seq-len", "1024", "--rank", "8", "--alpha", "16",
                   "--epochs", f"{remaining:.2f}"]
        if checkpoint is not None:
            command += ["--init-adapter", str(checkpoint)]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with LOG.open("a", encoding="utf-8") as log, \
                ERRLOG.open("a", encoding="utf-8") as errlog:
            log.write(f"\n=== supervisor: starting from step {step} "
                      f"({remaining:.2f} epochs remaining) ===\n")
            log.flush()
            subprocess.run(command, stdout=log, stderr=errlog, cwd=str(HOME),
                           creationflags=creationflags)
        if (OUT / "adapter_model.safetensors").is_file():
            DONE.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
