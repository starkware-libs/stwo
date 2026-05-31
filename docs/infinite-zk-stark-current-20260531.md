# Infinite ZK STARK current performance run

This report captures the first clean current-branch run after the Phase 1
paper-aligned FRI `R` infrastructure and Phase 1 benchmark instrumentation were
committed.

## Source

- branch: `infinite/zk-stark-paper-integration-v1`
- current commit: `a1218a8e4859190ccb84ee6fd12119bf3e63a0f5`
- baseline commit: `cca98119f4641c3e096de74b915c1aa55453bcad`
- run id: `20260531T132122Z`
- run date UTC: `2026-05-31T13:21:22Z`
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
| `simd polynomial commitment 2^20` | `40.079 ms` | `41.872 ms` | `+4.47%` |
| `cpu polynomial commitment 2^20` | `327.64 ms` | `331.45 ms` | `+1.16%` |
| `fold_line` | `387.27 us` | `381.87 us` | `-1.39%` |

These deltas are below the plan thresholds for default-path regression review.
The SIMD PCS delta is above the 3% noise band in this single run, so the
repeated-run gate remains open before Phase 2 signoff.

## Phase 1 ZK benchmark measurements

| Benchmark | Current Criterion interval |
| --- | ---: |
| `zk phase1 r sampling and evaluation cpu 2^12` | `[62.576 us, 62.639 us, 62.671 us]` |
| `zk phase1 r sampling and evaluation cpu 2^16` | `[1.1721 ms, 1.1728 ms, 1.1735 ms]` |
| `zk phase1 r sampling and evaluation cpu 2^20` | `[21.694 ms, 21.840 ms, 21.963 ms]` |
| `zk phase1 r commitment cpu 2^12` | `[1.1091 ms, 1.1170 ms, 1.1230 ms]` |
| `zk phase1 r commitment cpu 2^16` | `[7.3934 ms, 7.4306 ms, 7.4766 ms]` |
| `zk phase1 r commitment cpu 2^20` | `[98.857 ms, 98.919 ms, 98.981 ms]` |
| `zk phase1 h_batch addition cpu 2^12` | `[2.1161 us, 2.1624 us, 2.1875 us]` |
| `zk phase1 h_batch addition cpu 2^16` | `[27.817 us, 28.001 us, 28.104 us]` |
| `zk phase1 h_batch addition cpu 2^20` | `[516.68 us, 517.48 us, 518.70 us]` |
| `zk phase1 r opening construction cpu 2^12` | `[5.6328 us, 5.7772 us, 6.2119 us]` |
| `zk phase1 r opening construction cpu 2^16` | `[7.1208 us, 7.2258 us, 7.3577 us]` |
| `zk phase1 r opening construction cpu 2^20` | `[11.901 us, 12.078 us, 12.280 us]` |
| `zk phase1 r opening verification cpu 2^12` | `[5.5519 us, 5.5661 us, 5.5837 us]` |
| `zk phase1 r opening verification cpu 2^16` | `[6.9172 us, 6.9620 us, 7.0006 us]` |
| `zk phase1 r opening verification cpu 2^20` | `[8.9879 us, 9.1208 us, 9.2861 us]` |
| `zk phase1 r answer addition 6 queries` | `[21.106 ns, 21.790 ns, 22.234 ns]` |

## Phase status

- Phase 1 FRI `R` infrastructure is benchmarked and separately accountable.
- Original non-ZK STWO compatibility and existing PCS/FRI performance remain in
  bounds for this run.
- Small/medium/large Phase 1 `R` fixtures are now present.
- Phase 2 witness randomization remains blocked until repeated-run statistics,
  private-column ratio fixtures, proof-size metrics, memory/allocation proxy
  metrics, and the signed derivation gates are added and reviewed.

## Raw artifacts

The raw successful current-run artifacts are committed under
`docs/baselines/20260531T132122Z-current`.
