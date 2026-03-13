#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd "$EXPERIMENT_ROOT/../../../.." && pwd)
DEFAULT_ENV_FILE="$EXPERIMENT_ROOT/.env"
EXAMPLE_ENV_FILE="$EXPERIMENT_ROOT/config/experiment.env.example"

log() {
  printf '[autoresearch] %s\n' "$*"
}

die() {
  printf '[autoresearch] ERROR: %s\n' "$*" >&2
  exit 1
}

ensure_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

load_env() {
  local env_file="${1:-$DEFAULT_ENV_FILE}"

  if [[ -f "$EXAMPLE_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$EXAMPLE_ENV_FILE"
  fi
  if [[ -f "$env_file" ]]; then
    # shellcheck disable=SC1090
    source "$env_file"
  fi

  : "${AUTORESEARCH_RUN_TAG:=mar13-simd}"
  : "${AUTORESEARCH_BASE_BRANCH:=$(git -C "$REPO_ROOT" branch --show-current)}"
  : "${AUTORESEARCH_AGENT:=codex}"
  : "${AUTORESEARCH_AGENT_MODEL:=}"
  : "${AUTORESEARCH_AGENT_SANDBOX:=workspace-write}"
  : "${AUTORESEARCH_AGENT_MAX_ITERATIONS:=0}"
  : "${AUTORESEARCH_RUSTFLAGS:=-Awarnings -C target-cpu=native -C opt-level=3}"
  : "${AUTORESEARCH_USE_AVX_WRAPPER:=0}"
  : "${AUTORESEARCH_FAST_THRESHOLD:=0.995}"
  : "${AUTORESEARCH_PROMOTION_THRESHOLD:=0.998}"
  : "${AUTORESEARCH_ENABLE_PROMOTION:=1}"
  : "${AUTORESEARCH_ENABLE_POSEIDON_PROMOTION:=0}"
  : "${AUTORESEARCH_UPSTREAM_URL:=https://github.com/karpathy/autoresearch.git}"
  : "${AUTORESEARCH_UPSTREAM_COMMIT:=c2450add72cc80317be1fe8111974b892da10944}"

  : "${AUTORESEARCH_STATE_ROOT:=$EXPERIMENT_ROOT/.state}"
  : "${AUTORESEARCH_RUNS_ROOT:=$AUTORESEARCH_STATE_ROOT/runs}"
  : "${AUTORESEARCH_WORKTREE_ROOT:=$AUTORESEARCH_STATE_ROOT/worktrees}"
  : "${AUTORESEARCH_UPSTREAM_ROOT:=$AUTORESEARCH_STATE_ROOT/upstream}"

  : "${AUTORESEARCH_POLICY_FILE:=$EXPERIMENT_ROOT/config/policy.json}"
  : "${AUTORESEARCH_FAST_PROFILE:=$EXPERIMENT_ROOT/config/profile.fast.json}"
  : "${AUTORESEARCH_PROMOTION_PROFILE:=$EXPERIMENT_ROOT/config/profile.promotion.json}"
  : "${AUTORESEARCH_SCHEMA_FILE:=$EXPERIMENT_ROOT/AGENT_RESPONSE.schema.json}"
  : "${AUTORESEARCH_PROGRAM_FILE:=$EXPERIMENT_ROOT/program.md}"

  : "${AUTORESEARCH_WORKTREE_PATH:=$AUTORESEARCH_WORKTREE_ROOT/$AUTORESEARCH_RUN_TAG}"
  : "${AUTORESEARCH_RUN_DIR:=$AUTORESEARCH_RUNS_ROOT/$AUTORESEARCH_RUN_TAG}"
  : "${AUTORESEARCH_UPSTREAM_DIR:=$AUTORESEARCH_UPSTREAM_ROOT/autoresearch}"

  export SCRIPT_DIR
  export EXPERIMENT_ROOT
  export REPO_ROOT
  export AUTORESEARCH_RUN_TAG
  export AUTORESEARCH_BASE_BRANCH
  export AUTORESEARCH_AGENT
  export AUTORESEARCH_AGENT_MODEL
  export AUTORESEARCH_AGENT_SANDBOX
  export AUTORESEARCH_AGENT_MAX_ITERATIONS
  export AUTORESEARCH_RUSTFLAGS
  export AUTORESEARCH_USE_AVX_WRAPPER
  export AUTORESEARCH_FAST_THRESHOLD
  export AUTORESEARCH_PROMOTION_THRESHOLD
  export AUTORESEARCH_ENABLE_PROMOTION
  export AUTORESEARCH_ENABLE_POSEIDON_PROMOTION
  export AUTORESEARCH_UPSTREAM_URL
  export AUTORESEARCH_UPSTREAM_COMMIT
  export AUTORESEARCH_STATE_ROOT
  export AUTORESEARCH_RUNS_ROOT
  export AUTORESEARCH_WORKTREE_ROOT
  export AUTORESEARCH_UPSTREAM_ROOT
  export AUTORESEARCH_POLICY_FILE
  export AUTORESEARCH_FAST_PROFILE
  export AUTORESEARCH_PROMOTION_PROFILE
  export AUTORESEARCH_SCHEMA_FILE
  export AUTORESEARCH_PROGRAM_FILE
  export AUTORESEARCH_WORKTREE_PATH
  export AUTORESEARCH_RUN_DIR
  export AUTORESEARCH_UPSTREAM_DIR
}

ensure_state_dirs() {
  mkdir -p "$AUTORESEARCH_STATE_ROOT" "$AUTORESEARCH_RUNS_ROOT" "$AUTORESEARCH_WORKTREE_ROOT" "$AUTORESEARCH_UPSTREAM_ROOT"
}

git_current_commit() {
  git -C "$1" rev-parse --short HEAD
}

git_is_clean() {
  [[ -z "$(git -C "$1" status --porcelain)" ]]
}

reset_worktree() {
  local worktree="$1"
  git -C "$worktree" reset --hard HEAD >/dev/null
  git -C "$worktree" clean -fd >/dev/null
}

