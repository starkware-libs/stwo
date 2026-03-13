# Resources

## External references

| Resource | Why it matters |
| --- | --- |
| `https://github.com/karpathy/autoresearch` | Original methodology and baseline `program.md` structure |
| `https://github.com/karpathy/autoresearch/blob/c2450add72cc80317be1fe8111974b892da10944/README.md` | Upstream design assumptions and quick-start |
| `https://github.com/karpathy/autoresearch/blob/c2450add72cc80317be1fe8111974b892da10944/program.md` | The minimal autonomous research prompt pattern |
| `https://github.com/Shopify/liquid/pull/2056` | The case study that translated the technique into performance engineering |
| `https://patch-diff.githubusercontent.com/raw/Shopify/liquid/pull/2056.patch` | Fastest way to inspect the PR's benchmark harness and experiment docs |

## Key Liquid artifacts from the PR

- `performance/bench_quick.rb`
  - machine-readable performance metrics
- `auto/bench.sh`
  - correctness and benchmark gate
- `auto/autoresearch.md`
  - objective, safety constraints, scope, and progress log
- `autoresearch.jsonl`
  - experiment trace

## Repo-local references

| File | Why it matters |
| --- | --- |
| `README.md` | benchmark and toolchain overview |
| `CLAUDE.md` | repo-local architecture and performance priorities |
| `AGENTS.md` | role boundaries and soundness escalation protocol |
| `rust-toolchain.toml` | pinned nightly toolchain |
| `scripts/bench.sh` | existing native-CPU Criterion entrypoint |
| `scripts/test_avx.sh` | AVX512 test wrapper for backend hosts |
| `.github/workflows/ci.yaml` | current benchmark automation on self-hosted runners |
| `.github/workflows/benchmarks-pages.yaml` | benchmark publishing workflow |
| `crates/stwo/Cargo.toml` | Criterion bench targets and feature gates |
| `crates/stwo/benches/` | existing performance surface |
| `crates/examples/benches/poseidon.rs` | optional end-to-end proving benchmark |

## Observed repo facts that shape this lab

- The workspace already uses Criterion `0.5.1`.
- `cargo-criterion` is available locally.
- Benchmarks are already treated as CI gates on AVX-capable self-hosted runners.
- The current machine is `Darwin arm64`, so the upstream CUDA-only `autoresearch` training
  environment cannot be fully synchronized here.
- The first safe autonomous lane is low-level prover SIMD and mempool work, not verifier or lookup
  logic.

