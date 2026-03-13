#!/usr/bin/env python3

import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path


def geomean(values: list[float]) -> float:
    if not values:
        raise ValueError("no values provided")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def git_commit(worktree: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: parse_criterion_json.py <jsonl> <profile-name> <worktree>")

    jsonl_path = Path(sys.argv[1])
    profile_name = sys.argv[2]
    worktree = Path(sys.argv[3])

    benchmarks: dict[str, dict[str, float]] = {}

    for raw_line in jsonl_path.read_text().splitlines():
      if not raw_line.startswith("{"):
        continue
      record = json.loads(raw_line)
      if record.get("reason") != "benchmark-complete":
        continue
      benchmark_id = record["id"]
      benchmarks[benchmark_id] = {
          "typical_estimate_ns": float(record["typical"]["estimate"]),
          "lower_bound_ns": float(record["typical"]["lower_bound"]),
          "upper_bound_ns": float(record["typical"]["upper_bound"]),
      }

    if not benchmarks:
        raise SystemExit("no benchmark-complete records found")

    ordered_ids = sorted(benchmarks)
    estimates = [benchmarks[name]["typical_estimate_ns"] for name in ordered_ids]
    payload = {
        "profile": profile_name,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commit": git_commit(worktree),
        "benchmark_count": len(ordered_ids),
        "geomean_estimate_ns": geomean(estimates),
        "benchmarks": {name: benchmarks[name] for name in ordered_ids},
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

