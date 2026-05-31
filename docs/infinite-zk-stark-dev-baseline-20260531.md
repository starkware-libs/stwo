# Infinite ZK STARK dev baseline

This baseline was captured before continuing the paper integration work. It is
the reference point for preserving original STWO behavior and measuring the
performance cost of the ZK path.

## Source

- baseline branch: `dev`
- baseline commit: `cca98119f4641c3e096de74b915c1aa55453bcad`
- worktree: `/private/tmp/stwo-dev-baseline-cca98119-20260531`
- run id: `20260531T122603Z`
- run date UTC: `2026-05-31T12:26:03Z`
- git status: `0 dirty entries`
- rustc: `rustc 1.90.0-nightly (e9182f195 2025-07-13)`
- cargo: `cargo 1.90.0-nightly (eabb4cd92 2025-07-09)`
- `RAYON_NUM_THREADS`: unset
- `RUSTFLAGS`: unset
- features: `prover,parallel` where supported

The first attempt reached the benchmark phase but failed because sandboxed DNS
blocked Cargo from downloading `wasm-bindgen-test` from `static.crates.io`.
The baseline was rerun with network approval and completed successfully.

## Compatibility baseline

| Command | Status | Command elapsed |
| --- | ---: | ---: |
| `cargo test --locked --no-default-features --package stwo` | 0 | 1s |
| `cargo test --locked --package stwo --features prover` | 0 | 3s |

Observed test summaries:

| Suite | Result |
| --- | --- |
| no-default library tests | 69 passed; 0 failed |
| no-default doc tests | 9 passed; 0 failed |
| prover library tests | 268 passed; 0 failed |
| prover doc tests | 9 passed; 0 failed |

## Performance baseline

| Benchmark | Criterion interval |
| --- | ---: |
| `simd polynomial commitment 2^20` | `[39.822 ms, 40.079 ms, 40.368 ms]` |
| `cpu polynomial commitment 2^20` | `[326.69 ms, 327.64 ms, 328.77 ms]` |
| `fold_line` | `[386.42 us, 387.27 us, 388.12 us]` |

Command elapsed:

| Command | Status | Command elapsed |
| --- | ---: | ---: |
| `cargo bench --locked --features prover,parallel --bench pcs` | 0 | 53s |
| `cargo bench --locked --features prover,parallel --bench fri` | 0 | 21s |

## Match requirements for the ZK integration

- The original non-ZK STWO APIs and proof formats must keep passing the same
  compatibility commands above.
- The original non-ZK PCS and FRI benchmarks should remain within normal
  Criterion noise unless a reviewed performance note explains the delta.
- ZK-specific overhead must be measured separately from the original path. The
  `R` oracle commitment, `R` query openings, and FRI batching changes must not
  be hidden inside only command-level elapsed times.
- Any later witness-randomization phase must add benchmarks that separate trace
  randomization, quotient integration, PCS commitment, and FRI proving costs.

## Phase 2 performance gate status

This baseline is sufficient to anchor the clean-`dev` compatibility and
existing PCS/FRI Criterion measurements. It is not sufficient for Phase 2
witness-randomization signoff.

Before Phase 2 starts, the benchmark matrix must add repeated current-vs-dev
comparisons, machine metadata, `Cargo.lock` hash, small/medium/large fixtures,
private-column ratios, proof bytes, allocation or peak-memory proxies, and
non-overlapping ZK stage timings for `R` sampling, `R` evaluation, `R`
commitment, `R` opening, verifier authentication, `R(query)` answer addition,
and `H_batch` addition.

## Raw artifacts

The raw successful run artifacts are committed under `docs/baselines`:

- `docs/baselines/20260531T122603Z-baseline/report.md`
- `docs/baselines/20260531T122603Z-baseline/metrics.tsv`
- `docs/baselines/20260531T122603Z-baseline/commands.jsonl`
- `docs/baselines/20260531T122603Z-baseline/stwo-pcs-bench.stdout`
- `docs/baselines/20260531T122603Z-baseline/stwo-fri-bench.stdout`
