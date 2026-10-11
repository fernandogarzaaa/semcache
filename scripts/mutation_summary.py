"""Aggregate mutation/<module>.json reports from scripts/mutation.sh into one Markdown table."""

import json
import sys
from pathlib import Path


def counts(report: dict) -> tuple[int, int]:
    total = int(report.get("total") or 0)
    killed = int(report.get("killed", 0)) + int(report.get("timeouts", 0))
    return killed, total


def main() -> int:
    rows = []
    for path in sorted(Path("mutation").glob("*.json")):
        killed, total = counts(json.loads(path.read_text()))
        rows.append((path.stem, killed, total))
    k = sum(r[1] for r in rows)
    t = sum(r[2] for r in rows)
    print("| module | killed / mutants | score |\n|---|---|---|")
    for name, killed, total in rows:
        score = f"{100 * killed / total:.1f}%" if total else "n/a"
        print(f"| {name}.py | {killed} / {total} | {score} |")
    if t:
        print(f"| **total** | **{k} / {t}** | **{100 * k / t:.1f}%** |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
