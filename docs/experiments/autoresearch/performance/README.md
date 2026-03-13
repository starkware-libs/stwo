# STWO Autoresearch Performance Lab

This folder adapts the `karpathy/autoresearch` methodology to `stwo` performance work.
The goal is not to copy the upstream training setup verbatim. The goal is to reproduce the
research loop that made the Shopify Liquid case study effective:

1. lock down a benchmark contract,
2. constrain the agent's scope,
3. run one measured experiment at a time,
4. keep only improvements that pass safety gates.

## Status

- External methodology reviewed:
  - `karpathy/autoresearch` pinned at `c2450add72cc80317be1fe8111974b892da10944`
  - Shopify Liquid PR `#2056` reviewed as the reference case study
- Local repo survey completed against branch `perf/autoresearch-experiments` at `66d4f147`
- Existing benchmark substrate confirmed:
  - Criterion benches in `crates/stwo/benches/`
  - `scripts/bench.sh` for native-CPU benchmark runs
  - self-hosted AVX benchmark runners in `.github/workflows/ci.yaml`

## Platform note

The upstream `autoresearch` example repository is CUDA-specific. On this machine
(`aarch64-apple-darwin`), `uv sync` fails because upstream pins `torch==2.9.1+cu128`.
This lab therefore:

- installs the upstream source checkout locally for reference,
- records the platform incompatibility in install metadata,
- uses the same methodology against `stwo` with local Rust tooling,
- keeps the full upstream dependency sync for Linux/CUDA backend hosts.

## What is in here

- `CASE_STUDY.md`
  - the Liquid pattern we are replicating
- `RESOURCES.md`
  - external and repo-local references
- `PLAN.md`
  - full execution plan for the first experiment
- `program.md`
  - the agent charter for one measured performance experiment
- `config/`
  - run configuration, benchmark profiles, and safety policy
- `scripts/`
  - bootstrap, benchmark, guard, and loop automation

## First experiment scope

The first autonomous lane is intentionally narrow:

- Allowed edits:
  - `crates/stwo/src/prover/backend/simd/{bit_reverse,blake2s,blake2s_lifted,cm31,column,conversion,domain,m31,prefix_sum,qm31,poseidon252_lifted,very_packed_m31}.rs`
  - `crates/stwo/src/prover/backend/simd/fft/`
  - `crates/stwo/src/prover/mempool.rs`
  - `crates/stwo/benches/`
  - `crates/examples/benches/`
- Explicitly off limits:
  - verifier/core/PCS/FRI/lookups paths listed in `config/policy.json`
  - any new or edited `unsafe`

The benchmark profiles focus on low-level SIMD and allocation-sensitive prover paths that fit
those boundaries.

## Quick start

From the repo root:

```bash
cp docs/experiments/autoresearch/performance/config/experiment.env.example \
  docs/experiments/autoresearch/performance/.env

make -C docs/experiments/autoresearch/performance bootstrap
make -C docs/experiments/autoresearch/performance create-run
make -C docs/experiments/autoresearch/performance iteration
```

Run the autonomous loop:

```bash
make -C docs/experiments/autoresearch/performance loop
```

The example env defaults to the fast profile only. For stricter acceptance on a backend host,
set `AUTORESEARCH_ENABLE_PROMOTION=1` before `create-run` or `loop`.

For CLI-free smoke testing of the orchestration itself, set `AUTORESEARCH_AGENT=noop`.

## How to run it

All commands below assume you start at the repo root.

### Prerequisites

- Rust nightly from `rust-toolchain.toml` (`nightly-2025-07-14`)
- `git`
- `cargo`
- `python3`
- `codex` if `AUTORESEARCH_AGENT=codex`

Bootstrap installs `cargo-criterion` automatically if it is missing.

### 1. Local smoke test of the harness

Use this when you want to validate the worktree setup, benchmark parsing, and run bookkeeping
without asking an agent to edit code.

```bash
cp docs/experiments/autoresearch/performance/config/experiment.env.example \
  docs/experiments/autoresearch/performance/.env

printf '\nAUTORESEARCH_AGENT=noop\n' >> \
  docs/experiments/autoresearch/performance/.env

make -C docs/experiments/autoresearch/performance bootstrap
make -C docs/experiments/autoresearch/performance create-run
make -C docs/experiments/autoresearch/performance iteration
```

Expected result:

- a dedicated worktree under `.state/worktrees/<run-tag>`
- baseline benchmark JSON in `.state/runs/<run-tag>/`
- a `noop` row appended to `results.tsv`

### 2. Local autonomous run with Codex

Use this when you want one measured experiment proposed and applied by Codex on your machine.

```bash
cp docs/experiments/autoresearch/performance/config/experiment.env.example \
  docs/experiments/autoresearch/performance/.env

make -C docs/experiments/autoresearch/performance bootstrap
make -C docs/experiments/autoresearch/performance create-run
make -C docs/experiments/autoresearch/performance iteration
```

To keep iterating until stopped:

```bash
make -C docs/experiments/autoresearch/performance loop
```

To cap the loop to a fixed number of iterations:

```bash
printf '\nAUTORESEARCH_AGENT_MAX_ITERATIONS=5\n' >> \
  docs/experiments/autoresearch/performance/.env

make -C docs/experiments/autoresearch/performance loop
```

To stop a long-running loop cleanly, create:

```bash
touch docs/experiments/autoresearch/performance/.state/runs/<run-tag>/STOP
```

### 3. Backend host run with stricter acceptance

Use this on a quieter AVX-capable benchmark machine when you want the promotion profile enabled.

Add these settings to `docs/experiments/autoresearch/performance/.env` before `create-run` or
`loop`:

```bash
AUTORESEARCH_ENABLE_PROMOTION=1
AUTORESEARCH_USE_AVX_WRAPPER=1
```

Then run:

```bash
make -C docs/experiments/autoresearch/performance bootstrap
make -C docs/experiments/autoresearch/performance create-run
make -C docs/experiments/autoresearch/performance loop
```

Notes:

- `AUTORESEARCH_ENABLE_PROMOTION=1` adds the promotion benchmark profile.
- `AUTORESEARCH_USE_AVX_WRAPPER=1` routes the test gate through `scripts/test_avx.sh`.
- The example env already sets `AUTORESEARCH_RUSTFLAGS="-Awarnings -C target-cpu=native -C opt-level=3"`.

### 4. Optional end-to-end proving benchmark

This is separate from the autonomous loop. It runs the example proving benchmark mentioned in
`RESOURCES.md`.

```bash
RUSTFLAGS="-Awarnings -C target-cpu=native -C opt-level=3" \
cargo criterion -p stwo-examples --bench poseidon
```

### 5. Where results go

Runtime artifacts are intentionally kept out of git under:

- `.state/upstream/`
- `.state/worktrees/`
- `.state/runs/`

Each run directory contains:

- `results.tsv`
- `results.jsonl`
- baseline and candidate benchmark JSON
- stderr and test logs
- agent response snapshots

## Recommended execution environments

- Local workstation:
  - good for validating the harness, prompts, and benchmark parsing
- Dedicated backend host:
  - recommended for long autonomous runs
  - use a bare-metal or strongly isolated host, not a noisy shared laptop
  - prefer the repo's AVX-capable benchmark hardware for promotion runs
