# Infinite STWO ZK STARK baseline run

- run_id: 20260531T122603Z
- mode: baseline
- cwd: /private/tmp/stwo-dev-baseline-cca98119-20260531
- date_utc: 2026-05-31T12:26:03Z
- rustc: rustc 1.90.0-nightly (e9182f195 2025-07-13)
- cargo: cargo 1.90.0-nightly (eabb4cd92 2025-07-09)
- git_commit: cca98119f4641c3e096de74b915c1aa55453bcad
- git_status: 0 dirty entries
- rayon_num_threads: unset
- rustflags: unset
- cargo_features: prover,parallel where supported

## Commands

### stwo-no-default-tests

```sh
cargo test --locked --no-default-features --package stwo
```

- status: 0
- elapsed_seconds: 1
- stdout: target/infinite-zk-stark-baseline/20260531T122603Z-baseline/stwo-no-default-tests.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T122603Z-baseline/stwo-no-default-tests.stderr

### stwo-prover-tests

```sh
cargo test --locked --package stwo --features prover
```

- status: 0
- elapsed_seconds: 3
- stdout: target/infinite-zk-stark-baseline/20260531T122603Z-baseline/stwo-prover-tests.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T122603Z-baseline/stwo-prover-tests.stderr

### stwo-pcs-bench

```sh
cargo bench --locked --features prover,parallel --bench pcs
```

- status: 0
- elapsed_seconds: 53
- stdout: target/infinite-zk-stark-baseline/20260531T122603Z-baseline/stwo-pcs-bench.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T122603Z-baseline/stwo-pcs-bench.stderr

### stwo-fri-bench

```sh
cargo bench --locked --features prover,parallel --bench fri
```

- status: 0
- elapsed_seconds: 21
- stdout: target/infinite-zk-stark-baseline/20260531T122603Z-baseline/stwo-fri-bench.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T122603Z-baseline/stwo-fri-bench.stderr

## Artifacts

- jsonl: target/infinite-zk-stark-baseline/20260531T122603Z-baseline/commands.jsonl
- metrics: target/infinite-zk-stark-baseline/20260531T122603Z-baseline/metrics.tsv
