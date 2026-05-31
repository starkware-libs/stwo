# Infinite STWO ZK STARK baseline-repeated run

- run_id: 20260531T135020Z
- mode: baseline-repeated
- cwd: /private/tmp/stwo-dev-baseline-cca98119-20260531
- date_utc: 2026-05-31T13:50:20Z
- rustc: rustc 1.90.0-nightly (e9182f195 2025-07-13)
- cargo: cargo 1.90.0-nightly (eabb4cd92 2025-07-09)
- git_commit: cca98119f4641c3e096de74b915c1aa55453bcad
- git_status: 0 dirty entries
- rayon_num_threads: unset
- rustflags: unset
- cargo_target_dir: unset
- cargo_lock_sha256: 9885b1c54b89c32229a62966b1e7f97c7533f9a966336a4ecccf98bbbb2aed3d
- os: Darwin Mac 25.4.0 Darwin Kernel Version 25.4.0: Thu Mar 19 19:33:25 PDT 2026; root:xnu-12377.101.15~1/RELEASE_ARM64_T6041 arm64
- cpu_model: 
- cpu_count: 16
- memory_bytes: 
- repetitions: 3
- cargo_features: prover,parallel where supported

## Commands

### stwo-no-default-tests-run-1

```sh
cargo test --locked --no-default-features --package stwo
```

- status: 0
- elapsed_seconds: 2
- stdout: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-no-default-tests-run-1.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-no-default-tests-run-1.stderr

### stwo-no-default-tests-run-2

```sh
cargo test --locked --no-default-features --package stwo
```

- status: 0
- elapsed_seconds: 2
- stdout: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-no-default-tests-run-2.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-no-default-tests-run-2.stderr

### stwo-no-default-tests-run-3

```sh
cargo test --locked --no-default-features --package stwo
```

- status: 0
- elapsed_seconds: 1
- stdout: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-no-default-tests-run-3.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-no-default-tests-run-3.stderr

### stwo-prover-tests-run-1

```sh
cargo test --locked --package stwo --features prover
```

- status: 0
- elapsed_seconds: 4
- stdout: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-prover-tests-run-1.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-prover-tests-run-1.stderr

### stwo-prover-tests-run-2

```sh
cargo test --locked --package stwo --features prover
```

- status: 0
- elapsed_seconds: 3
- stdout: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-prover-tests-run-2.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-prover-tests-run-2.stderr

### stwo-prover-tests-run-3

```sh
cargo test --locked --package stwo --features prover
```

- status: 0
- elapsed_seconds: 3
- stdout: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-prover-tests-run-3.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-prover-tests-run-3.stderr

### stwo-pcs-bench-run-1

```sh
cargo bench --locked --features prover,parallel --bench pcs
```

- status: 0
- elapsed_seconds: 26
- stdout: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-pcs-bench-run-1.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-pcs-bench-run-1.stderr

### stwo-pcs-bench-run-2

```sh
cargo bench --locked --features prover,parallel --bench pcs
```

- status: 0
- elapsed_seconds: 25
- stdout: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-pcs-bench-run-2.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-pcs-bench-run-2.stderr

### stwo-pcs-bench-run-3

```sh
cargo bench --locked --features prover,parallel --bench pcs
```

- status: 0
- elapsed_seconds: 25
- stdout: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-pcs-bench-run-3.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-pcs-bench-run-3.stderr

### stwo-fri-bench-run-1

```sh
cargo bench --locked --features prover,parallel --bench fri
```

- status: 0
- elapsed_seconds: 10
- stdout: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-fri-bench-run-1.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-fri-bench-run-1.stderr

### stwo-fri-bench-run-2

```sh
cargo bench --locked --features prover,parallel --bench fri
```

- status: 0
- elapsed_seconds: 10
- stdout: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-fri-bench-run-2.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-fri-bench-run-2.stderr

### stwo-fri-bench-run-3

```sh
cargo bench --locked --features prover,parallel --bench fri
```

- status: 0
- elapsed_seconds: 10
- stdout: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-fri-bench-run-3.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/stwo-fri-bench-run-3.stderr

## Artifacts

- jsonl: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/commands.jsonl
- metrics: target/infinite-zk-stark-baseline/20260531T135020Z-baseline-repeated/metrics.tsv
