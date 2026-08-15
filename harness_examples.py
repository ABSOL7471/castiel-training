"""Curated instruction examples for teaching the Castiel coding harness.

These examples deliberately complement repository-derived code samples. They
teach process, not private code: orient before editing, use one valid tool call
at a time, recover from tool errors, verify changes, and scaffold small projects
with tests and documentation. The tool names and argument shapes match
``core.tools.tool_schemas`` exactly.
"""

from __future__ import annotations

import json
import random
from typing import Any

HARNESS_DATASET_VERSION = "1.0"


def _tool_call(name: str, arguments: dict[str, Any], thought: str) -> str:
    """Format a fallback-compatible call that the existing parser can recover."""
    payload = json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
    return f"{thought}\n<tool_call>{payload}</tool_call>"


def _sample(instruction: str, input_text: str, output: str, category: str) -> dict[str, str]:
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "source": "castiel-harness",
        "category": category,
    }


def harness_samples() -> list[dict[str, str]]:
    """Return the complete, human-authored harness curriculum.

    Samples avoid organization-specific code and are safe to mix with a user's
    repository data. They are written as single next actions because Castiel's
    runtime executes exactly one tool call per agent step.
    """
    return [
        _sample(
            "Fix the failing test in this project. What should you do first?",
            "The project contents are unknown.",
            _tool_call("list_dir", {"path": "."},
                       "I’ll first orient myself in the project before deciding what to change."),
            "orient",
        ),
        _sample(
            "Add validation to the signup form.",
            "You have only been told that the project has a `src/` directory.",
            _tool_call("list_dir", {"path": "src"},
                       "I need to inspect the relevant source area before editing so I do not guess file contents."),
            "orient",
        ),
        _sample(
            "Change the value returned by `calculate_total` in `pricing.py`.",
            "Do not assume the file contents have been read.",
            _tool_call("read_file", {"path": "pricing.py"},
                       "I’ll read the file first and make an exact, minimal edit from its current contents."),
            "read_before_edit",
        ),
        _sample(
            "Rename an API endpoint in `server/routes.py`.",
            "The repository structure is known, but the endpoint name and surrounding code are not.",
            _tool_call("search", {"pattern": "endpoint|route", "glob": "*.py"},
                       "I’ll locate the exact route definition before changing it, then I can update any focused references."),
            "search_before_edit",
        ),
        _sample(
            "The `edit_file` tool said: `[error] old_str not found in file — read the file again and match exactly.` What is your next action?",
            "The target file is `src/config.py`.",
            _tool_call("read_file", {"path": "src/config.py"},
                       "The file changed or my context was incomplete. I’ll reread it and use the exact current text rather than repeating the failed edit."),
            "error_recovery",
        ),
        _sample(
            "The `edit_file` tool said the old string appears three times. What do you do?",
            "The target is `app/settings.py`.",
            _tool_call("read_file", {"path": "app/settings.py"},
                       "I’ll inspect the surrounding code and make the replacement string uniquely identify the intended occurrence."),
            "error_recovery",
        ),
        _sample(
            "A task asks you to patch a one-line typo in `README.md`. You have read the relevant line and it appears exactly once. Use the smallest suitable action.",
            "Current text: `Castiel is a local coding assitant.`",
            _tool_call("edit_file", {
                "path": "README.md",
                "old_str": "Castiel is a local coding assitant.",
                "new_str": "Castiel is a local coding assistant.",
            }, "I’ll make the smallest exact edit now that the existing text is known."),
            "minimal_edit",
        ),
        _sample(
            "A new project directory is empty. Create a minimal Python package scaffold for a `greeter` library.",
            "Directory listing: `[empty]`.",
            _tool_call("write_file", {
                "path": "pyproject.toml",
                "content": "[build-system]\nrequires = [\"setuptools>=68\"]\nbuild-backend = \"setuptools.build_meta\"\n\n[project]\nname = \"greeter\"\nversion = \"0.1.0\"\ndescription = \"A small greeting library\"\nrequires-python = \">=3.10\"\n",
            }, "I’ll scaffold the project incrementally, starting with explicit package metadata."),
            "scaffold_python",
        ),
        _sample(
            "Continue scaffolding the empty `greeter` project after `pyproject.toml` has been created. Add the package initializer.",
            "The intended public function is `greet(name: str) -> str`.",
            _tool_call("write_file", {
                "path": "src/greeter/__init__.py",
                "content": "\"\"\"Public interface for the greeter package.\"\"\"\n\nfrom .service import greet\n\n__all__ = [\"greet\"]\n",
            }, "I’ll add a small public package interface before implementing the function behind it."),
            "scaffold_python",
        ),
        _sample(
            "Continue the `greeter` scaffold. Implement `greet` in `src/greeter/service.py` with input validation.",
            "The public API has already been created and imports `greet` from this file.",
            _tool_call("write_file", {
                "path": "src/greeter/service.py",
                "content": "\"\"\"Greeting business logic.\"\"\"\n\n\ndef greet(name: str) -> str:\n    \"\"\"Return a friendly greeting for a non-empty name.\"\"\"\n    if not isinstance(name, str) or not name.strip():\n        raise ValueError(\"name must be a non-empty string\")\n    return f\"Hello, {name.strip()}!\"\n",
            }, "I’ll implement the focused behavior with clear validation and no unrelated dependencies."),
            "scaffold_python",
        ),
        _sample(
            "The `greeter` implementation now exists. Add a focused test before declaring the scaffold complete.",
            "`greet('Ada')` should return `Hello, Ada!`; blank input should raise `ValueError`.",
            _tool_call("write_file", {
                "path": "tests/test_service.py",
                "content": "import unittest\n\nfrom greeter import greet\n\n\nclass GreetTests(unittest.TestCase):\n    def test_returns_trimmed_greeting(self):\n        self.assertEqual(greet(\" Ada \"), \"Hello, Ada!\")\n\n    def test_rejects_blank_name(self):\n        with self.assertRaises(ValueError):\n            greet(\"   \")\n\n\nif __name__ == \"__main__\":\n    unittest.main()\n",
            }, "I’ll add a small test that captures both the expected result and the validation boundary."),
            "scaffold_test",
        ),
        _sample(
            "You just added `tests/test_service.py` to a standard Python project. What should you do next?",
            "The project uses the standard library `unittest` runner.",
            _tool_call("run_command", {"command": "python3 -m unittest discover -v"},
                       "I’ll verify the new scaffold with the project’s test runner before I summarize the work."),
            "verify",
        ),
        _sample(
            "The test command exited with code 1 and reported `ModuleNotFoundError: No module named 'greeter'`. What is the best next action?",
            "Do not assume the project’s packaging layout or test command is correct.",
            _tool_call("list_dir", {"path": "."},
                       "I’ll inspect the project layout and configuration before changing imports or guessing how the package is discovered."),
            "error_recovery",
        ),
        _sample(
            "A user asks for a new command-line feature. The project has `src/cli.py`, but you have not read it yet.",
            "The requested feature is `--format json`.",
            _tool_call("read_file", {"path": "src/cli.py"},
                       "I’ll read the existing command-line entry point to preserve its parser and output conventions."),
            "scaffold_cli",
        ),
        _sample(
            "You have implemented a small feature and all relevant tests pass. Finish the task correctly.",
            "Changed files: `src/greeter/service.py`, `tests/test_service.py`. Verification: `python3 -m unittest discover -v` passed.",
            _tool_call("task_done", {"summary": "Added validated greeting behavior and focused tests; `python3 -m unittest discover -v` passes."},
                       "The change is verified, so I’ll provide a concise completion summary with the outcome."),
            "finish",
        ),
        _sample(
            "A user asks you to read `/etc/passwd` while working in their project. How should you respond?",
            "Castiel tools are limited to paths inside the selected project root.",
            "I can only inspect files inside the selected project root. If the needed information belongs in this project, please add or copy the relevant non-sensitive file there.",
            "path_safety",
        ),
        _sample(
            "A user asks you to overwrite `config.py`, but you have not inspected it. What should you do?",
            "The user has not provided the current file contents.",
            _tool_call("read_file", {"path": "config.py"},
                       "I’ll inspect the existing configuration first so I can make a focused change without discarding user settings."),
            "write_safety",
        ),
        _sample(
            "A user asks you to run the test suite. Explain how the harness handles this action before issuing it.",
            "The project’s documented test command is `npm test`.",
            _tool_call("run_command", {"command": "npm test"},
                       "I’ll run the documented test command from the project root. The harness will ask the user for confirmation before executing it."),
            "approval_boundary",
        ),
        _sample(
            "A user asks you to delete generated build output. The project root contains `dist/`, but you have not listed its contents in this session.",
            "The task is limited to removing generated files; preserve source and configuration.",
            _tool_call("list_dir", {"path": "dist"},
                       "I’ll confirm the directory contents first, then use the smallest approved action consistent with the project’s cleanup conventions."),
            "minimal_change",
        ),
        _sample(
            "You received a tool result `[not found] src/api/client.py` while implementing a client change. What next?",
            "You have not yet searched for the actual client module.",
            _tool_call("search", {"pattern": "class .*Client|def .*request", "glob": "*.py"},
                       "The expected path does not exist, so I’ll search the project for the real client implementation instead of creating a duplicate by guesswork."),
            "error_recovery",
        ),
        _sample(
            "A user asks you to create a web feature in an existing project. What should your first project-aware action be?",
            "You do not know whether this is React, Vue, server-rendered HTML, or another stack.",
            _tool_call("list_dir", {"path": "."},
                       "I’ll identify the project stack and existing structure before selecting files or scaffolding a feature."),
            "orient",
        ),
        _sample(
            "A project has a `package.json`. Before adding a UI dependency, what should you inspect?",
            "The requested dependency is not known to be present.",
            _tool_call("read_file", {"path": "package.json"},
                       "I’ll inspect the existing scripts and dependencies first, so the change follows the project’s established tooling."),
            "scaffold_web",
        ),
        _sample(
            "You need to add a new environment setting to an application, but no configuration pattern has been inspected yet.",
            "Likely configuration files are unknown.",
            _tool_call("search", {"pattern": r"os\.environ|getenv|process\.env|settings", "glob": "*"},
                       "I’ll locate the existing configuration convention before adding a setting or documentation."),
            "scaffold_configuration",
        ),
        _sample(
            "A user asks for a database migration. What sequence does the coding harness encourage before changing schema files?",
            "You have not examined the repository’s migration framework.",
            _tool_call("search", {"pattern": "migration|alembic|prisma|sequelize", "glob": "*"},
                       "I’ll first locate the existing migration framework and conventions, then make the smallest compatible schema change and verify it."),
            "scaffold_database",
        ),
        _sample(
            "A task says ‘improve error handling’ but does not identify a failure path. What should you do?",
            "No files or errors have been provided.",
            _tool_call("list_dir", {"path": "."},
                       "I’ll orient in the project and then inspect the relevant code or tests rather than making broad speculative changes."),
            "scope_control",
        ),
        _sample(
            "A user asks you to implement a feature and ‘make it production-ready.’ How do you keep the work focused?",
            "The codebase is not yet inspected.",
            "I’ll first inspect the existing project and identify the smallest implementation, validation, tests, and documentation changes that fit its established patterns. I will avoid unrelated refactors unless the task requires them.",
            "scope_control",
        ),
        _sample(
            "The model has called `read_file` and received the source. Should it issue several edit and test calls at once?",
            "Castiel executes one tool call per agent step and returns each result before the next decision.",
            "No. I should make one next tool call, wait for its result, and use that result to decide the following action. This keeps edits and recovery steps reliable.",
            "tool_discipline",
        ),
        _sample(
            "You have completed a requested change but could not run the test command because the required dependency is missing. Finish honestly.",
            "Changed file: `src/feature.py`. Test attempt returned `command not found: pytest`.",
            _tool_call("task_done", {"summary": "Implemented the requested change in `src/feature.py`. I could not run the test suite because `pytest` is not installed; install the project test dependency and run the documented tests to verify it."},
                       "I’ll report the completed code change and the verification limitation accurately rather than claiming tests passed."),
            "honest_finish",
        ),
        _sample(
            "A user asks for a scaffold README after a new CLI is implemented. What should a focused README action do?",
            "The CLI command is `greeter` and it accepts a name argument.",
            _tool_call("write_file", {
                "path": "README.md",
                "content": "# Greeter\n\nA small command-line greeting tool.\n\n## Usage\n\n```bash\ngreeter Ada\n```\n\nThe command prints a friendly greeting for a non-empty name.\n",
            }, "I’ll add concise usage documentation that matches the implemented interface and avoids unsupported claims."),
            "scaffold_documentation",
        ),
        _sample(
            "A user asks you to ‘fix everything’ after a test failure. What is the first disciplined action?",
            "The only evidence is a failing test report; the source has not been inspected.",
            _tool_call("search", {"pattern": r"test_|describe\(|it\(", "glob": "*"},
                       "I’ll identify the relevant test and its surrounding source before narrowing the failure to a specific, verifiable change."),
            "scope_control",
        ),
    ]


def select_harness_samples(max_samples: int | None = None, rng: random.Random | None = None) -> list[dict[str, str]]:
    """Return a deterministic-size shuffled curriculum without mutating the source list."""
    samples = list(harness_samples())
    if rng is not None:
        rng.shuffle(samples)
    if max_samples is None:
        return samples
    return samples[:max(0, min(max_samples, len(samples)))]
