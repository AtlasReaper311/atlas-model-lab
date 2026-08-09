#!/usr/bin/env python3
"""Run the complete local validation gate for Atlas Model Lab."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence

COMMANDS: tuple[tuple[str, Sequence[str]], ...] = (
    (
        "Compile",
        (
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "src",
            "tests",
            "examples",
            "scripts",
        ),
    ),
    ("Ruff lint", (sys.executable, "-m", "ruff", "check", ".")),
    ("Ruff format", (sys.executable, "-m", "ruff", "format", "--check", ".")),
    ("mypy", (sys.executable, "-m", "mypy", "src", "tests")),
    ("pytest", (sys.executable, "-m", "pytest", "-q")),
    ("Build", (sys.executable, "-m", "build")),
)


def main() -> int:
    for label, command in COMMANDS:
        print(f"\n== {label} ==")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"\nValidation failed at: {label}", file=sys.stderr)
            return result.returncode

    print("\nAll Atlas Model Lab validation gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
