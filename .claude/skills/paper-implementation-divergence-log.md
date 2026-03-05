---
name: paper-implementation-divergence-log
description: >
  Living record of all known divergences between the Circle STARK paper /
  STWO Whitepaper and the production codebase. Agents MUST read this before
  modifying any theoretically-grounded component. Agents MUST update this
  when finding new divergences. Never resolve a divergence silently.
---

# Paper-Implementation Divergence Log

Last analyzed: 2026-03-05

## How to Use This Log

- **Before modifying**: Search this log for the component you are modifying.
  If a divergence exists, understand it before changing anything.
- **After discovering**: Add a new entry immediately. Do not proceed without documenting.
- **Resolution**: Only a human with cryptography expertise can close a divergence
  as RESOLVED-SAFE. Agents may propose resolution, never confirm it.

## Active Divergences

### DIVERGENCE-001: Dimension Gap Handling in FRI

Paper: Circle STARK paper Section 6, Protocol 1 — FRI commit phase begins with
a "decomposition step" where prover sends scalar lambda and computes
g = f - lambda * v_n to move from L_N(F) (dim N+1) into L'_N(F) (dim N).

Code: `crates/stwo/src/core/fri.rs` and `crates/stwo/src/prover/fri.rs` —
The FRI verifier has a `first_layer` / `inner_layers` / `last_layer` structure.
The circle-to-line fold (first layer) handles the J-split. There is no explicit
lambda scalar transmission in the current protocol flow. The quotient decomposition
in `crates/stwo/src/core/pcs/quotients.rs` handles the dimension gap implicitly
through the DEEP quotient construction.

Type: Intentional deviation
Risk: NEUTRAL (the constraint system ensures polynomials are in L'_N by construction
via the quotient identity; the dimension gap parameter lambda = 0 when constraint
degree is odd, which is the common case)
Status: OPEN
Notes: Verify that even-degree constraint systems correctly handle lambda != 0.

### DIVERGENCE-002: FRI Folding Order

Paper: STWO Whitepaper "Circle FRI.tex" — Protocol describes folding as:
Round 1 = J-split (y-twiddle), Rounds 2..r = pi-split (x-twiddle).

Code: `crates/stwo/src/core/fri.rs` — First layer performs circle-to-line
fold (`fold_circle`), then inner layers perform line folds. The constant
`CIRCLE_TO_LINE_FOLD_STEP` is used for the first fold. Inner layer queries are
derived by folding the original queries.

Type: Optimization (batched structure)
Risk: NEUTRAL
Status: OPEN
Notes: The mathematical operation is equivalent; the batching structure differs
from the sequential presentation in the paper.

### DIVERGENCE-003: QM31 as 4 Base Field Polynomials

Paper: Circle STARK paper Section 5 — Trace polynomials p_j are in L'_N(F_p).
The secure field QM31 is used for random challenges and composition polynomial.

Code: Throughout `crates/stwo/src/core/pcs/` — QM31 polynomials are decomposed
into 4 base field coordinate polynomials for commitment and FRI. The composition
polynomial is split into 2 * SECURE_EXTENSION_DEGREE = 8 coordinate polynomials
(see `verify()` in `crates/stwo/src/core/verifier.rs`).

Type: Intentional deviation (efficiency)
Risk: NEUTRAL (mathematically equivalent; reduces FRI to base field operations)
Status: OPEN
Notes: This is standard practice. The `from_partial_evals` method in
`crates/stwo/src/core/fields/qm31.rs:51-57` reconstructs the combined value.

### DIVERGENCE-004: Composition Polynomial Split

Paper: STWO Whitepaper — Composition polynomial q is decomposed via
Lemma 7 (Decomposition Lemma) into d-1 components on disjoint twin-cosets.

Code: `crates/stwo/src/core/verifier.rs` — `COMPOSITION_LOG_SPLIT: u32 = 1`
is hardcoded. The split produces `2 * SECURE_EXTENSION_DEGREE` columns.
A TODO in the module notes this should be configurable.

Type: Intentional deviation (simplified)
Risk: PERFORMANCE (limits flexibility for higher-degree constraints)
Status: OPEN
Notes: The TODO acknowledges this limitation. For production stwo-cairo usage,
the current split may be sufficient.

### DIVERGENCE-005: Pairwise LogUp Column Grouping

Paper: STWO Whitepaper "Optimizations.tex" — Describes pairwise grouping of
logup entries to halve the interaction trace columns.

Code: `crates/constraint-framework/src/logup.rs` and
`crates/constraint-framework/src/prover/logup.rs` — LogUp implementation
groups fractions pairwise. The `Batching` type in `lib.rs:47` controls
which entries are batched together.

Type: Optimization (matches paper)
Risk: NEUTRAL
Status: RESOLVED-SAFE
Notes: Implementation matches the optimization described in the whitepaper.

### DIVERGENCE-006: Non-Transposed Merkle Tree

Paper: STWO Whitepaper describes Merkle commitments over evaluation columns.

Code: `crates/stwo/src/core/vcs_lifted/` — Uses a "lifted" Merkle tree
variant where multiple polynomials of different sizes are committed in a
single tree by lifting smaller polynomials to the largest domain size.
The `lifting_log_size` parameter in `PcsConfig` controls this.

Type: Intentional deviation (efficiency)
Risk: NEUTRAL (reduces number of Merkle trees and proof size)
Status: OPEN
Notes: The `vcs_lifted` module is a newer addition alongside the original `vcs`.

### DIVERGENCE-007: Security Parameter Defaults

Paper: STWO Whitepaper "Soundness.tex" — Targets 100-bit security with
26 grinding bits.

Code: `PcsConfig::default()` in `crates/stwo/src/core/pcs/mod.rs` — Default uses
`pow_bits: 10`, `log_blowup_factor: 1`, `n_queries: 3`. This yields
`security_bits() = 10 + 1*3 = 13` bits — far below production requirements.

Type: Intentional deviation (test defaults)
Risk: SOUNDNESS (if used in production without override)
Status: OPEN
Notes: The default config is clearly for testing. Production deployments
(e.g., stwo-cairo) must set appropriate parameters. Document this prominently.

### DIVERGENCE-008: Proof of Work Grinding Bit Limit

Paper: STWO Whitepaper targets 26 grinding bits for 100-bit security.

Code: `crates/stwo/src/prover/backend/simd/grind.rs` — TODO comment:
"support more than 32 bits." Current implementation limited to 32 PoW bits.

Type: Implementation limitation
Risk: NEUTRAL (26 < 32, so current limit is sufficient for target security)
Status: OPEN
Notes: The 32-bit limit is adequate for the 26-bit target but leaves no room
for future security parameter increases.

### DIVERGENCE-009: Poseidon2 Constants and Round Order

Paper: Poseidon2 paper (https://eprint.iacr.org/2023/323.pdf) Section 5.

Code: `crates/examples/src/poseidon/mod.rs` — Three critical TODOs:
- "Use poseidon's real constants" — placeholder constants in use
- Coefficients unverified against Section 5.3
- Round matrix may be applied in wrong order relative to paper

Type: Bug candidate
Risk: SOUNDNESS (for any system using this Poseidon2 implementation)
Status: OPEN — ESCALATION REQUIRED
Notes: This is in the examples crate, not the core prover. However, any
downstream user of this code (e.g., for a Poseidon2 hash proof) would
inherit these issues. The placeholder constants make this unsuitable for
production use.

### DIVERGENCE-010: LogUp Validation Missing in Blake Example

Code: `crates/examples/src/blake/round/constraints.rs` — TODO: "validate logup"
Code: `crates/examples/src/blake/scheduler/mod.rs` — TODO: "validate logup"

Type: Implementation gap
Risk: SOUNDNESS (for Blake example only — constraints may be under-specified)
Status: OPEN
Notes: Example code only, but users may copy patterns from examples.

## Resolved Divergences

### DIVERGENCE-005 (see above)
Resolved: Pairwise LogUp grouping matches whitepaper Optimizations.tex.
