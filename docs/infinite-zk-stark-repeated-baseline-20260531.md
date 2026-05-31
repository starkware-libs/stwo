# Infinite ZK STARK repeated baseline report - 2026-05-31

## Scope

This report records the repeated performance evidence used before starting Phase 2 witness-randomization integration.

- Original STWO baseline: `cca98119f49ec36f2c8e13e99fdce351d5061dfee`.
- Current branch measurement: `24be57be9f8c7ccd4d954a531efb9a566bad764e`.
- Evidence commit: `43fa7565`.
- Current branch status during measurement: clean, `0 dirty entries`.
- Repetitions: `3`.
- Baseline artifact: `docs/baselines/20260531T135020Z-baseline-repeated/report.md`.
- Current artifact: `docs/baselines/20260531T135231Z-current-repeated/report.md`.

The evidence commit `43fa7565` only copied benchmark artifacts and added this
report. It did not change executable Rust code relative to measured commit
`24be57be`.

Later pre-Phase 2 guardrail changes are limited to metadata validation, review
gates, and tests. They are outside the measured default PCS/FRI prover hot
paths. Any semantic witness-randomization, quotient, FRI, or prover hot-path
change requires a fresh matched run before performance sign-off.

## Default-path regression check

| Benchmark | Baseline mids | Current mids | Baseline mean | Current mean | Delta |
|---|---:|---:|---:|---:|---:|
| PCS prover bench, SIMD | `39.899 ms`, `39.726 ms`, `39.860 ms` | `39.776 ms`, `40.084 ms`, `39.833 ms` | `39.828 ms` | `39.898 ms` | `+0.17%` |
| PCS prover bench, CPU | `357.08 ms`, `361.42 ms`, `359.36 ms` | `334.36 ms`, `335.68 ms`, `331.49 ms` | `359.287 ms` | `333.843 ms` | `-7.08%` |
| FRI fold-line bench | `412.64 us`, `405.08 us`, `401.08 us` | `415.83 us`, `405.72 us`, `398.37 us` | `406.267 us` | `406.640 us` | `+0.09%` |

Interpretation: the current Phase 1 guardrail/benchmark increment does not show a measurable default-path regression in the repeated PCS/FRI baseline matrix. SIMD PCS and FRI are within noise; CPU PCS measured faster on this machine.

## Phase 1 ZK overhead component baselines

These measurements are current-branch component costs for the random-column commitment and opening skeleton. They are not end-to-end Phase 2 proof costs.

| Component | `2^12` mean | `2^16` mean | `2^20` mean |
|---|---:|---:|---:|
| `R` coefficient sampling | `6.233 us` | `98.317 us` | `1.565 ms` |
| `R` evaluation | `56.835 us` | `1.076 ms` | `20.076 ms` |
| `R` sampling + evaluation | `62.711 us` | `1.174 ms` | `21.634 ms` |
| `R` commitment | `1.112 ms` | `7.442 ms` | `98.709 ms` |
| `H_batch` addition | `2.056 us` | `28.036 us` | `519.943 us` |
| `R` opening construction | `5.674 us` | `8.328 us` | `11.175 us` |
| `R` opening verification | `5.479 us` | `6.862 us` | `9.058 us` |

| Component | Mean |
|---|---:|
| Answer-field addition | `17.022 ns` |

## Evidence captured

- Repeated no-default-feature STWO tests passed on the original STWO baseline.
- Repeated prover-feature STWO tests passed on the original STWO baseline.
- Repeated PCS and FRI benchmarks completed on the original STWO baseline.
- Repeated no-default-feature STWO tests passed on the current branch.
- Repeated prover-feature STWO tests passed on the current branch.
- Repeated PCS, FRI, and Phase 1 ZK component benchmarks completed on the current branch.

## Evidence still required before Phase 2 performance sign-off

- End-to-end small, medium, and large trace fixtures with real masked witness paths.
- Private-column ratio matrix: `0%`, `25%`, `50%`, `100%`.
- Actual proof-size deltas after Phase 2 changes, not only script hooks.
- Actual memory/allocation deltas after Phase 2 changes, not only script hooks.
- Confirmation that original STWO test vectors still pass after Phase 2 changes.

The baseline runner now exposes command hooks for these dimensions, but these measurements cannot be completed until the real Phase 2 path exists.
