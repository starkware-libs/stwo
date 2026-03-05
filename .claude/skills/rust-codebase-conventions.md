---
name: rust-codebase-conventions
description: >
  Repo-specific Rust patterns for STWO: feature flags, error handling, type
  patterns, unsafe policy, no_std compatibility, formatting, and clippy
  configuration. Use when contributing code, reviewing PRs, or understanding
  code style decisions.
---

# Rust Codebase Conventions

## Toolchain

See `rust-toolchain.toml` for pinned nightly version. CI also runs on stable for verifier tests.
Nightly features used (behind `prover` feature): `stdarch_x86_avx512`, `array_chunks`,
`iter_array_chunks`, `portable_simd`, `slice_ptr_get`.

## Feature Flags

| Flag | Purpose | Enables |
|------|---------|---------|
| `std` (default) | Standard library | Hash, IO, etc. |
| `prover` | Prover code | Implies `std`. SIMD, trace gen, etc. |
| `parallel` | Rayon parallelism | Requires `prover` |
| `slow-tests` | Long-running tests | Release mode only |
| `tracing` | Tracing instrumentation | Span/event logging |

**Architecture**: The `core/` module is `no_std`-compatible (verifier path).
The `prover/` module requires `prover` feature. This separation enables
on-chain verifier deployment.

## Formatting

Configured in `rustfmt.toml`. Run: `scripts/rust_fmt.sh`.
Key: imports grouped `StdExternalCrate`, granularity `Module`, comments wrapped at 100.

## Clippy

Strict: `-D warnings` in CI (all warnings are errors). Runs per-crate individually.
See `.cargo/config.toml` for `rustflags` and per-crate `Cargo.toml` `[lints]` sections.

## Error Handling

- **Verifier errors**: Use `thiserror` derive for structured error types
  (e.g., `FriVerificationError`, `VerificationError`)
- **Prover errors**: `ProvingError` enum
- **No panics in verifier**: The verifier must return `Result`, never panic
  (except `debug_assert!` in non-release builds)
- **Prover panics**: `assert!` is acceptable for invariant violations in the prover

## Type Patterns

### Field Type Aliases
```rust
pub type BaseField = M31;
pub type SecureField = QM31;
pub const SECURE_EXTENSION_DEGREE: usize = 4;
```

### Collection Types
```rust
pub type ColumnVec<T> = Vec<T>;           // Indexed by column
pub struct ComponentVec<T>(pub Vec<ColumnVec<T>>); // Indexed by component
pub struct TreeVec<T>(pub Vec<T>);        // Indexed by commitment tree
```

### Backend Abstraction
The `Backend` trait (in `prover/backend/mod.rs`) abstracts over CPU and SIMD.
`BackendForChannel` connects a backend with a channel type.

## Unsafe Policy

1. **Justified**: Every `unsafe` block must have a clear performance justification
2. **Documented**: Safety invariants documented in comments
3. **Minimized**: Unsafe scope as small as possible
4. **Reviewed**: All unsafe in soundness-critical paths requires math reviewer approval

Common patterns:
- `uninit_vec()` — skip zero-init for large buffers (write-before-read guaranteed)
- `from_simd_unchecked()` — skip M31 range check (values known < P)
- `packed_at()` / `set_packed()` — unchecked SIMD column access (bounds proven)
- `transmute` — SIMD type reinterpretation (layout compatibility verified)

## Serialization

- `serde` with `Serialize`/`Deserialize` for proof types
- Default features disabled (`default-features = false, features = ["derive"]`)

## Dependencies

- `no_std`-compatible by default (features = `["alloc"]` not `["std"]`)
- `std_shims` crate provides `Vec` etc. for `no_std` environments
- `hashbrown` instead of `std::collections::HashMap`

## Workspace Structure

- Version: `2.1.0` (workspace-level)
- Resolver: `"2"` (Cargo's v2 feature resolver)
- `ensure-verifier-no_std` is excluded from workspace (separate build target)

## Typos

`.typos.toml` configured with `extend-ignore-re = ['excluder']`
