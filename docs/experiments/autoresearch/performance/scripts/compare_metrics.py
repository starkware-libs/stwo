#!/usr/bin/env python3

import json
import math
import sys
from pathlib import Path


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_metrics.py <baseline.json> <candidate.json>")

    baseline = json.loads(Path(sys.argv[1]).read_text())
    candidate = json.loads(Path(sys.argv[2]).read_text())

    base_ids = set(baseline["benchmarks"])
    cand_ids = set(candidate["benchmarks"])
    if base_ids != cand_ids:
        missing = sorted(base_ids - cand_ids)
        extra = sorted(cand_ids - base_ids)
        raise SystemExit(
            json.dumps(
                {
                    "error": "benchmark-id-mismatch",
                    "missing_in_candidate": missing,
                    "extra_in_candidate": extra,
                },
                indent=2,
                sort_keys=True,
            )
        )

    ratios = {}
    for name in sorted(base_ids):
        base_value = float(baseline["benchmarks"][name]["typical_estimate_ns"])
        cand_value = float(candidate["benchmarks"][name]["typical_estimate_ns"])
        ratios[name] = cand_value / base_value

    composite_ratio = geomean(list(ratios.values()))
    best_improvement = min(ratios.items(), key=lambda item: item[1])
    worst_regression = max(ratios.items(), key=lambda item: item[1])
    payload = {
        "baseline_commit": baseline["commit"],
        "candidate_commit": candidate["commit"],
        "benchmark_count": len(ratios),
        "composite_ratio": composite_ratio,
        "improvement_pct": (1.0 - composite_ratio) * 100.0,
        "best_improvement": {
            "benchmark": best_improvement[0],
            "ratio": best_improvement[1],
        },
        "worst_regression": {
            "benchmark": worst_regression[0],
            "ratio": worst_regression[1],
        },
        "ratios": ratios,
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

