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

write_metadata
: > "${JSONL}"
: > "${METRICS}"

run_command "stwo-no-default-tests" \
  "${ZK_STARK_CMD_NO_DEFAULT_TESTS:-cargo test --locked --no-default-features --package stwo}"

run_command "stwo-prover-tests" \
  "${ZK_STARK_CMD_PROVER_TESTS:-cargo test --locked --package stwo --features prover}"

run_command "stwo-pcs-bench" \
  "${ZK_STARK_CMD_PCS_BENCH:-cargo bench --locked --features prover,parallel --bench pcs}"

run_command "stwo-fri-bench" \
  "${ZK_STARK_CMD_FRI_BENCH:-cargo bench --locked --features prover,parallel --bench fri}"

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
