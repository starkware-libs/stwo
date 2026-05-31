#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-baseline}"
OUT_DIR="${ZK_STARK_BASELINE_DIR:-target/infinite-zk-stark-baseline}"
RUN_ID="${ZK_STARK_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${OUT_DIR}/${RUN_ID}-${MODE}"
REPORT="${RUN_DIR}/report.md"
JSONL="${RUN_DIR}/commands.jsonl"
METRICS="${RUN_DIR}/metrics.tsv"

mkdir -p "${RUN_DIR}"

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read())[1:-1])'
}

write_metadata() {
  local cargo_lock_hash
  cargo_lock_hash="$(shasum -a 256 Cargo.lock 2>/dev/null | awk '{print $1}' || true)"
  {
    echo "# Infinite STWO ZK STARK ${MODE} run"
    echo
    echo "- run_id: ${RUN_ID}"
    echo "- mode: ${MODE}"
    echo "- cwd: $(pwd)"
    echo "- date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "- rustc: $(rustc --version 2>/dev/null || true)"
    echo "- cargo: $(cargo --version 2>/dev/null || true)"
    echo "- git_commit: $(git rev-parse HEAD 2>/dev/null || true)"
    echo "- git_status: $(git status --short 2>/dev/null | wc -l | tr -d ' ') dirty entries"
    echo "- rayon_num_threads: ${RAYON_NUM_THREADS:-unset}"
    echo "- rustflags: ${RUSTFLAGS:-unset}"
    echo "- cargo_target_dir: ${CARGO_TARGET_DIR:-unset}"
    echo "- cargo_lock_sha256: ${cargo_lock_hash:-unavailable}"
    echo "- os: $(uname -a 2>/dev/null || true)"
    echo "- cpu_model: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
    echo "- cpu_count: $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || true)"
    echo "- memory_bytes: $(sysctl -n hw.memsize 2>/dev/null || true)"
    echo "- cargo_features: prover,parallel where supported"
    echo
    echo "## Commands"
    echo
  } > "${REPORT}"
}

run_command() {
  local name="$1"
  local command="$2"
  local start end elapsed status
  start="$(date +%s)"
  echo "### ${name}" >> "${REPORT}"
  echo >> "${REPORT}"
  echo '```sh' >> "${REPORT}"
  echo "${command}" >> "${REPORT}"
  echo '```' >> "${REPORT}"
  echo >> "${REPORT}"

  set +e
  bash -lc "${command}" > "${RUN_DIR}/${name}.stdout" 2> "${RUN_DIR}/${name}.stderr"
  status="$?"
  set -e

  end="$(date +%s)"
  elapsed="$((end - start))"
  printf '%s\t%s\tseconds\n' "${name}.elapsed" "${elapsed}" >> "${METRICS}"
  printf '%s\t%s\tstatus\n' "${name}.status" "${status}" >> "${METRICS}"

  local escaped_command
  escaped_command="$(printf '%s' "${command}" | json_escape)"
  printf '{"name":"%s","command":"%s","status":%s,"elapsed_seconds":%s}\n' \
    "${name}" "${escaped_command}" "${status}" "${elapsed}" >> "${JSONL}"

  {
    echo "- status: ${status}"
    echo "- elapsed_seconds: ${elapsed}"
    echo "- stdout: ${RUN_DIR}/${name}.stdout"
    echo "- stderr: ${RUN_DIR}/${name}.stderr"
    echo
  } >> "${REPORT}"

  if [ "${status}" -ne 0 ] && [ "${ZK_STARK_ALLOW_FAILURES:-0}" != "1" ]; then
    echo "Command failed: ${name}" >&2
    exit "${status}"
  fi
}

parse_criterion_stdout() {
  local name="$1"
  python3 - "${name}" "${RUN_DIR}/${name}.stdout" "${METRICS}" <<'PY'
import re
import sys

command_name, stdout_path, metrics_path = sys.argv[1:]
pending = None

def sanitize(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")

def normalize_unit(unit: str) -> str:
    return unit.replace("\u00b5", "u")

with open(stdout_path, "r", encoding="utf-8", errors="replace") as source, open(
    metrics_path, "a", encoding="utf-8"
) as metrics:
    for raw_line in source:
        line = raw_line.rstrip()
        if not line.strip():
            continue

        match = re.search(
            r"time:\s+\[([0-9.]+)\s+([^\s]+)\s+([0-9.]+)\s+([^\s]+)\s+([0-9.]+)\s+([^\s]+)\]",
            line,
        )
        if match:
            benchmark_name = line[: match.start()].strip() or pending
            if not benchmark_name:
                continue
            key = f"criterion.{command_name}.{sanitize(benchmark_name)}"
            low, low_unit, mid, mid_unit, high, high_unit = match.groups()
            metrics.write(f"{key}.low\t{low}\t{normalize_unit(low_unit)}\n")
            metrics.write(f"{key}.mid\t{mid}\t{normalize_unit(mid_unit)}\n")
            metrics.write(f"{key}.high\t{high}\t{normalize_unit(high_unit)}\n")
            pending = None
            continue

        stripped = line.strip()
        if (
            stripped.startswith("Benchmarking ")
            or stripped.startswith("Analyzing ")
            or stripped.startswith("Found ")
            or stripped.startswith("change:")
            or stripped.startswith("sample")
        ):
            continue
        pending = stripped
PY
}

write_metadata
: > "${JSONL}"
: > "${METRICS}"

run_command "stwo-no-default-tests" \
  "${ZK_STARK_CMD_NO_DEFAULT_TESTS:-cargo test --locked --no-default-features --package stwo}"

run_command "stwo-prover-tests" \
  "${ZK_STARK_CMD_PROVER_TESTS:-cargo test --locked --package stwo --features prover}"

run_command "stwo-pcs-bench" \
  "${ZK_STARK_CMD_PCS_BENCH:-cargo bench --locked --features prover,parallel --bench pcs}"
parse_criterion_stdout "stwo-pcs-bench"

run_command "stwo-fri-bench" \
  "${ZK_STARK_CMD_FRI_BENCH:-cargo bench --locked --features prover,parallel --bench fri}"
parse_criterion_stdout "stwo-fri-bench"

run_command "stwo-zk-phase1-bench" \
  "${ZK_STARK_CMD_ZK_PHASE1_BENCH:-cargo bench --locked --features prover,parallel --bench zk_phase1}"
parse_criterion_stdout "stwo-zk-phase1-bench"

if [ -n "${ZK_STARK_CMD_SMALL_TRACE:-}" ]; then
  run_command "zk-small-trace" "${ZK_STARK_CMD_SMALL_TRACE}"
fi

if [ -n "${ZK_STARK_CMD_MEDIUM_TRACE:-}" ]; then
  run_command "zk-medium-trace" "${ZK_STARK_CMD_MEDIUM_TRACE}"
fi

if [ -n "${ZK_STARK_CMD_LARGE_TRACE:-}" ]; then
  run_command "zk-large-trace" "${ZK_STARK_CMD_LARGE_TRACE}"
fi

{
  echo "## Artifacts"
  echo
  echo "- jsonl: ${JSONL}"
  echo "- metrics: ${METRICS}"
} >> "${REPORT}"

echo "${REPORT}"
