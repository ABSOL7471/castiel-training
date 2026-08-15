#!/usr/bin/env python3
"""Castiel Training — Stage 1: dataset preparation.

Walks one or more project directories and builds an instruction-tuning dataset
from your own code. By default it also adds a curated Castiel harness curriculum
so a fine-tuned model learns the local agent's orient → plan → edit → verify
workflow, tool conventions, safety boundaries, recovery behavior, and focused
project scaffolding patterns.

Usage:
    python prepare_dataset.py --repos ~/code/project1 ~/code/project2 \
        --out dataset.jsonl --max-file-kb 64
    python prepare_dataset.py --repos ~/code/project1 --no-harness

Everything runs locally. Nothing is uploaded anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from harness_examples import HARNESS_DATASET_VERSION, select_harness_samples

CODE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".sh",
    ".sql", ".html", ".css", ".vue", ".svelte", ".lua", ".zig",
    ".cjs", ".mjs",
}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
             "dist", "build", ".next", ".cache", "target", "vendor"}

LANG_BY_EXT = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "React JSX",
    ".ts": "TypeScript", ".tsx": "React TSX", ".go": "Go", ".rs": "Rust",
    ".java": "Java", ".kt": "Kotlin", ".c": "C", ".cpp": "C++",
    ".cs": "C#", ".rb": "Ruby", ".php": "PHP", ".swift": "Swift",
    ".sh": "Shell", ".sql": "SQL", ".html": "HTML", ".css": "CSS",
    ".cjs": "Node.js CommonJS", ".mjs": "Node.js ESM",
}


def iter_code_files(root, max_kb):
    root = Path(root).expanduser().resolve()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CODE_EXTS:
            continue
        if any(part in SKIP_DIRS or part.startswith(".")
               for part in path.relative_to(root).parts[:-1]):
            continue
        try:
            if path.stat().st_size > max_kb * 1024 or path.stat().st_size < 200:
                continue
            yield path.relative_to(root), path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeDecodeError):
            continue


def make_samples(rel_path, code, rng, group):
    """Create explain, completion, and fill-in-the-middle examples per source file.

    Every sample carries a ``group`` key naming its source file so the
    train/validation split can keep all samples derived from one file on the
    same side — otherwise the validation loss just measures memorization.
    """
    language = LANG_BY_EXT.get(Path(rel_path).suffix.lower(), "code")
    samples = []

    samples.append({
        "instruction": f"Explain what the file `{rel_path}` in this project does, "
                       "then show its full contents.",
        "input": "",
        "output": f"This {language} file implements part of the project.\n\n"
                  f"```{language.lower().split()[0]}\n{code}\n```",
        "source": "repository",
        "category": "explain",
        "group": group,
    })

    lines = code.splitlines(keepends=True)
    if len(lines) >= 12:
        split = len(lines) // 2
        head, tail = "".join(lines[:split]), "".join(lines[split:])
        samples.append({
            "instruction": f"Continue this {language} file (`{rel_path}`) in the "
                           "project's existing style. Output only the remaining code.",
            "input": head,
            "output": tail,
            "source": "repository",
            "category": "complete",
            "group": group,
        })

        span = max(2, len(lines) // 6)
        lower = len(lines) // 4
        upper = max(lower + 1, len(lines) - span - len(lines) // 4)
        start = rng.randint(lower, upper)
        prefix = "".join(lines[:start])
        middle = "".join(lines[start:start + span])
        suffix = "".join(lines[start + span:])
        samples.append({
            "instruction": f"Fill in the missing section of this {language} file "
                           f"(`{rel_path}`). Output only the missing code.",
            "input": f"<prefix>\n{prefix}\n</prefix>\n<suffix>\n{suffix}\n</suffix>",
            "output": middle,
            "source": "repository",
            "category": "fill_in_middle",
            "group": group,
        })

    return samples


def write_jsonl(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def percentage(value):
    parsed = float(value)
    if not 0 <= parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 0 and less than 1")
    return parsed


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Build a LoRA dataset from code and the Castiel harness")
    parser.add_argument("--repos", nargs="+", required=True, help="Project directories to harvest")
    parser.add_argument("--out", default="dataset.jsonl", help="Output training JSONL path")
    parser.add_argument("--max-file-kb", type=positive_int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-split", type=percentage, default=0.05,
                        help="Validation fraction in [0, 1), written to <out>.val")
    parser.add_argument("--no-harness", action="store_true",
                        help="Exclude the curated Castiel workflow and scaffolding curriculum")
    parser.add_argument("--harness-samples", type=positive_int, default=24,
                        help="Maximum harness curriculum examples to add (default: 24)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    samples = []
    repository_sample_count = 0
    duplicate_count = 0
    seen_content = set()
    for repo in args.repos:
        count = 0
        for relative_path, code in iter_code_files(repo, args.max_file_kb):
            digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
            if digest in seen_content:
                duplicate_count += 1
                continue
            seen_content.add(digest)
            group = f"{Path(repo).expanduser().resolve().name}/{relative_path.as_posix()}"
            generated = make_samples(relative_path, code, rng, group)
            samples.extend(generated)
            repository_sample_count += len(generated)
            count += 1
        print(f"harvested {count} files from {repo}")
    if duplicate_count:
        print(f"skipped {duplicate_count} duplicate files (identical contents already harvested)")

    harness_sample_count = 0
    if not args.no_harness:
        harness = select_harness_samples(args.harness_samples, rng)
        for index, sample in enumerate(harness):
            sample["group"] = f"castiel-harness/{index}"
        samples.extend(harness)
        harness_sample_count = len(harness)
        print(f"added {harness_sample_count} Castiel harness examples (curriculum v{HARNESS_DATASET_VERSION})")

    # Split at file granularity: all samples derived from one source file stay
    # on the same side, so validation loss measures generalization, not recall
    # of a file the model already saw two-thirds of during training.
    groups = {}
    for sample in samples:
        groups.setdefault(sample["group"], []).append(sample)
    group_keys = list(groups)
    rng.shuffle(group_keys)

    validation_target = int(len(samples) * args.val_split) if samples and args.val_split else 0
    if args.val_split and len(group_keys) > 1:
        validation_target = max(1, validation_target)
    validation, train = [], []
    for key in group_keys:
        bucket = validation if len(validation) < validation_target else train
        bucket.extend(groups[key])
    rng.shuffle(train)
    rng.shuffle(validation)

    output = Path(args.out).expanduser()
    validation_output = output.with_suffix(output.suffix + ".val")
    write_jsonl(output, train)
    write_jsonl(validation_output, validation)

    print(f"\ncomposition: {repository_sample_count} repository / {harness_sample_count} harness samples")
    print(f"wrote {len(train)} train / {len(validation)} validation samples")
    print(f"-> {output} and {validation_output}")
    if repository_sample_count and repository_sample_count < 200:
        print("note: <200 repository-derived samples is thin for a LoRA — add more repos or lower --max-file-kb exclusions.")
    if not repository_sample_count:
        print("note: no repository code samples were found; the output contains only the reusable Castiel harness curriculum.")


if __name__ == "__main__":
    main()
