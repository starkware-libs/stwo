# Implementation Plan

## Objective

Prepare an autonomous, benchmark-driven performance research lane for `stwo` that:

- reproduces the `autoresearch` experiment style,
- is runnable locally and on a backend host,
- stays within repo safety boundaries,
- keeps a durable record of every measured attempt.

## Non-goals for phase one

- no autonomous edits to soundness-critical files
- no autonomous edits to lookup prover logic
- no autonomous edits that introduce or modify `unsafe`
- no attempt to reproduce Karpathy's CUDA training example on this macOS host

## Workstreams

### 1. Method capture

- pin upstream `autoresearch`
- summarize the Liquid pattern
- encode the relevant ideas in `program.md`

### 2. Benchmark contract

- use Criterion JSON output instead of scraping human-readable tables
- define two profiles:
  - `simd-core-fast`
  - `simd-core-promotion`
- compare runs by normalized geometric mean against the current best result

### 3. Safety policy

- allow only explicitly enumerated performance-owned files
- block soundness-critical paths
- block changes involving `unsafe`
- run targeted module tests before every benchmarked acceptance decision

### 4. Experiment orchestration

- create a dedicated git worktree per run tag
- keep branch state isolated from the main working tree
- run one agent-proposed experiment at a time
- benchmark and accept/reject outside the agent
- commit only measured wins
- log every attempt to `results.tsv` and `results.jsonl`

### 5. Backend-host portability

- bootstrap Rust nightly and `cargo-criterion`
- support native-CPU flags via environment variables
- optionally enable the repo's AVX512 wrapper on compatible Linux hosts

## First experiment profile

### Allowed optimization surface

- `crates/stwo/src/prover/backend/simd/fft/`
- `crates/stwo/src/prover/backend/simd/bit_reverse.rs`
- `crates/stwo/src/prover/backend/simd/blake2s.rs`
- `crates/stwo/src/prover/backend/simd/blake2s_lifted.rs`
- `crates/stwo/src/prover/backend/simd/cm31.rs`
- `crates/stwo/src/prover/backend/simd/column.rs`
- `crates/stwo/src/prover/backend/simd/conversion.rs`
- `crates/stwo/src/prover/backend/simd/domain.rs`
- `crates/stwo/src/prover/backend/simd/m31.rs`
- `crates/stwo/src/prover/backend/simd/prefix_sum.rs`
- `crates/stwo/src/prover/backend/simd/qm31.rs`
- `crates/stwo/src/prover/backend/simd/poseidon252_lifted.rs`
- `crates/stwo/src/prover/backend/simd/very_packed_m31.rs`
- `crates/stwo/src/prover/mempool.rs`
- `crates/stwo/benches/`
- `crates/examples/benches/`

### Fast profile

- package: `stwo`
- benches:
  - `field`
  - `bit_rev`
  - `prefix_sum`
  - `fft`

### Promotion profile

- package: `stwo`
- benches:
  - `field`
  - `bit_rev`
  - `prefix_sum`
  - `fft`
  - `merkle`

### Optional promotion confirmation

- package: `stwo-examples`
- bench:
  - `poseidon`

This is opt-in because it is slower and crosses from microbenchmarks into proving workload shape.

## Acceptance protocol

An experiment is kept only if all of the following hold:

1. diff stays inside the allowlist
2. no forbidden paths are touched
3. no `unsafe` edits appear in the diff
4. targeted tests pass
5. fast-profile score improves beyond the configured threshold
6. promotion-profile score also improves, if promotion is enabled

Otherwise the dedicated worktree is reset to the previous accepted commit and the result is logged
as `discard`, `blocked`, `gate_fail`, or `noop`.

## Readiness exit criteria

This prep phase is complete when:

- the folder contains the benchmark contract, safety policy, and plan,
- the bootstrap script can clone upstream `autoresearch`,
- the benchmark parser can consume Criterion JSON,
- a dedicated run can be initialized from a git worktree,
- the loop script is ready to call a local agent CLI.

