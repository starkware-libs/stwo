# Infinite ZK STARK current performance run

This report captures the first clean current-branch run after the Phase 1
paper-aligned FRI `R` infrastructure and Phase 1 benchmark instrumentation were
committed.

## Source

- branch: `infinite/zk-stark-paper-integration-v1`
- current commit: `71304ac6d7870b8296b8fc6be5aea59025f7d869`
- baseline commit: `cca98119f4641c3e096de74b915c1aa55453bcad`
- run id: `20260531T130537Z`
- run date UTC: `2026-05-31T13:05:37Z`
- git status: `0 dirty entries`
- rustc: `rustc 1.90.0-nightly (e9182f195 2025-07-13)`
- cargo: `cargo 1.90.0-nightly (eabb4cd92 2025-07-09)`
- `Cargo.lock` SHA-256: `9885b1c54b89c32229a62966b1e7f97c7533f9a966336a4ecccf98bbbb2aed3d`
- `RAYON_NUM_THREADS`: unset
- `RUSTFLAGS`: unset

## Compatibility

| Command | Status | Command elapsed |
| --- | ---: | ---: |
| `cargo test --locked --no-default-features --package stwo` | 0 | 2s |
| `cargo test --locked --package stwo --features prover` | 0 | 4s |

## Existing non-ZK benchmark comparison

| Benchmark | Clean dev baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| `simd polynomial commitment 2^20` | `40.079 ms` | `39.912 ms` | `-0.42%` |
| `cpu polynomial commitment 2^20` | `327.64 ms` | `330.91 ms` | `+1.00%` |
| `fold_line` | `387.27 us` | `384.74 us` | `-0.65%` |

These deltas are below the plan thresholds for default-path regression review.

## Phase 1 ZK benchmark measurements

| Benchmark | Current Criterion interval |
| --- | ---: |
| `zk phase1 r sampling and evaluation cpu 2^16` | `[1.1713 ms, 1.1725 ms, 1.1743 ms]` |
| `zk phase1 r commitment cpu 2^16` | `[7.3511 ms, 7.3679 ms, 7.3948 ms]` |
| `zk phase1 h_batch addition cpu 2^16` | `[27.957 us, 28.060 us, 28.189 us]` |
| `zk phase1 r opening construction cpu 2^16` | `[6.8653 us, 6.9546 us, 7.0013 us]` |
| `zk phase1 r opening verification cpu 2^16` | `[6.3476 us, 6.3646 us, 6.3863 us]` |
| `zk phase1 r answer addition 6 queries` | `[17.685 ns, 17.741 ns, 17.791 ns]` |

## Phase status

- Phase 1 FRI `R` infrastructure is benchmarked and separately accountable.
- Original non-ZK STWO compatibility and existing PCS/FRI performance remain in
  bounds for this run.
- Phase 2 witness randomization remains blocked until the signed derivation,
  small/medium/large fixture matrix, private-column ratio fixtures, proof-size
  metrics, and memory/allocation proxy metrics are added and reviewed.

## Raw artifacts

The raw successful current-run artifacts are committed under
`docs/baselines/20260531T130537Z-current`.
