# Infinite STWO ZK STARK current run

- run_id: 20260531T132122Z
- mode: current
- cwd: /Users/ehjc/workspace/github.com/infinite/stwo
- date_utc: 2026-05-31T13:21:22Z
- rustc: rustc 1.90.0-nightly (e9182f195 2025-07-13)
- cargo: cargo 1.90.0-nightly (eabb4cd92 2025-07-09)
- git_commit: a1218a8e4859190ccb84ee6fd12119bf3e63a0f5
- git_status: 0 dirty entries
- rayon_num_threads: unset
- rustflags: unset
- cargo_target_dir: unset
- cargo_lock_sha256: 9885b1c54b89c32229a62966b1e7f97c7533f9a966336a4ecccf98bbbb2aed3d
- os: Darwin Mac 25.4.0 Darwin Kernel Version 25.4.0: Thu Mar 19 19:33:25 PDT 2026; root:xnu-12377.101.15~1/RELEASE_ARM64_T6041 arm64
- cpu_model: 
- cpu_count: 
- memory_bytes: 
- cargo_features: prover,parallel where supported

## Commands

### stwo-no-default-tests

```sh
cargo test --locked --no-default-features --package stwo
```

- status: 0
- elapsed_seconds: 3
- stdout: target/infinite-zk-stark-baseline/20260531T132122Z-current/stwo-no-default-tests.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T132122Z-current/stwo-no-default-tests.stderr

### stwo-prover-tests

```sh
cargo test --locked --package stwo --features prover
```

- status: 0
- elapsed_seconds: 4
- stdout: target/infinite-zk-stark-baseline/20260531T132122Z-current/stwo-prover-tests.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T132122Z-current/stwo-prover-tests.stderr

### stwo-pcs-bench

```sh
cargo bench --locked --features prover,parallel --bench pcs
```

- status: 0
- elapsed_seconds: 24
- stdout: target/infinite-zk-stark-baseline/20260531T132122Z-current/stwo-pcs-bench.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T132122Z-current/stwo-pcs-bench.stderr

### stwo-fri-bench

```sh
cargo bench --locked --features prover,parallel --bench fri
```

- status: 0
- elapsed_seconds: 9
- stdout: target/infinite-zk-stark-baseline/20260531T132122Z-current/stwo-fri-bench.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T132122Z-current/stwo-fri-bench.stderr

### stwo-zk-phase1-bench

```sh
cargo bench --locked --features prover,parallel --bench zk_phase1
```

- status: 0
- elapsed_seconds: 163
- stdout: target/infinite-zk-stark-baseline/20260531T132122Z-current/stwo-zk-phase1-bench.stdout
- stderr: target/infinite-zk-stark-baseline/20260531T132122Z-current/stwo-zk-phase1-bench.stderr

## Artifacts

- jsonl: target/infinite-zk-stark-baseline/20260531T132122Z-current/commands.jsonl
- metrics: target/infinite-zk-stark-baseline/20260531T132122Z-current/metrics.tsv
