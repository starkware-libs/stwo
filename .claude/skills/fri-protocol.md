---
name: fri-protocol
description: >
  Circle FRI protocol specifics for STWO: commitment phase, query phase,
  folding operations, security parameter derivation, and multi-step folding.
  Use when modifying FRI prover or verifier code, changing FRI parameters,
  or reviewing folding operations.
---

# FRI Protocol (Circle Variant)

## Canonical Theory Sources

- `.agents/papers/llm/INDEX.llm.md` — unified symbol map and conflict notes
- `.agents/papers/llm/Circle_STARKs.llm.md` — core circle FRI decomposition/folding soundness
- `.agents/papers/llm/Stwo_Whitepaper.llm.md` — multi-domain FRI and parameter/security framing

## Protocol Structure

### Configuration

```rust
// crates/stwo/src/core/fri.rs — FriConfig
pub struct FriConfig {
    pub log_blowup_factor: u32,         // Rate = 1/2^B
    pub log_last_layer_degree_bound: u32, // Degree of final polynomial
    pub n_queries: usize,                // Number of query positions
    pub line_fold_step: u32,             // Folding step for inner layers
}
```

**Parameter ranges** (enforced in `FriConfig::new()`):
- `log_last_layer_degree_bound`: 0..=10
- `log_blowup_factor`: 1..=16
- `line_fold_step`: must be > 0

**Security**: `security_bits() = log_blowup_factor * n_queries`
(plus `pow_bits` from the PCS config).

### Commit Phase

**Source**: `.agents/papers/llm/Circle_STARKs.llm.md`
(`prot:IOP:proximity`, `prot:IOP:proximity:batch`)

1. **First layer** (circle-to-line fold):
   - Input: polynomial evaluations on a circle domain
   - Operation: J-split fold with random alpha
   - The circle domain collapses to a line domain
   - Constant: `CIRCLE_TO_LINE_FOLD_STEP` in verifier

2. **Inner layers** (line folds):
   - Input: evaluations on a line domain
   - Operation: fold by `line_fold_step` with random alpha per step
   - Each fold halves the domain by `fold_step` doublings
   - Last inner layer may use a smaller fold_step to land exactly on
     `log_last_layer_degree_bound`

3. **Last layer**:
   - Prover sends the coefficients of the final polynomial
   - Degree must be <= `2^log_last_layer_degree_bound`

### Verification (Query Phase)

**Implementation**: `crates/stwo/src/core/fri.rs` — `FriVerifier` struct + impl

1. **Sample queries**: Draw `n_queries` random positions from the first layer domain
2. **Decommit first layer**: Verify Merkle openings, fold circle evaluations to line
3. **Decommit inner layers**: For each layer, verify Merkle openings and fold
4. **Verify last layer**: Check folded evaluations match the last layer polynomial

### Folding Operations

**Circle-to-line fold** (`fold_circle`):
```
Given f(x,y) at point P and conjugate -P:
  f_folded(x) = (f(P) + f(-P))/2 + alpha * (f(P) - f(-P))/(2y)
```
This is the J-split: even + alpha * odd component.

**Line fold** (`fold_line`):
```
Given g(x) at point x and -x:
  g_folded(2x^2-1) = (g(x) + g(-x))/2 + alpha * (g(x) - g(-x))/(2x)
```

**Implementation**:
- Verifier: `crates/stwo/src/core/fri.rs` — `SparseEvaluation::fold_circle()`,
  `FriInnerLayerVerifier::verify_and_fold()`
- Prover: `crates/stwo/src/prover/fri.rs` — `FriProver`
- Backend ops: `crates/stwo/src/prover/backend/*/fri.rs`

### Multi-Step Folding

When `line_fold_step > 1`, multiple folding rounds are batched into a single
FRI layer commitment. The verifier must unfold `fold_step` times using
`fold_step` many folding alphas (drawn from a single alpha via powers).

**Source**: `.agents/papers/llm/Stwo_Whitepaper.llm.md`
(`prot:cFRI:multi`, `e:cFRI:multi:folding`)

## Sparse Evaluation

FRI queries produce a "sparse evaluation" — values at query positions and
their conjugate/symmetric positions needed for folding.

```rust
// crates/stwo/src/core/fri.rs — SparseEvaluation
// Subset of evaluations at specific query positions, grouped for folding
```

## Error Types

```rust
pub enum FriVerificationError {
    InvalidNumFriLayers,           // Wrong number of folding layers
    FirstLayerEvaluationsInvalid,  // Circle fold check failed
    FirstLayerCommitmentInvalid,   // Merkle proof invalid
    InnerLayerEvaluationsInvalid,  // Line fold check failed
    InnerLayerCommitmentInvalid,   // Merkle proof invalid
    LastLayerDegreeInvalid,        // Final polynomial too large
    LastLayerEvaluationsInvalid,   // Final polynomial doesn't match
}
```

## Security Analysis

**Source**: `.agents/papers/llm/Circle_STARKs.llm.md`
(`thm:FRI:soundness:round:by:round`) and
`.agents/papers/llm/Stwo_Whitepaper.llm.md`
(`thm:cFRI:multi:soundness`, Section "6. Parameter Rules")

Soundness error has three components:
1. **Proximity gap**: Depends on rate (rho = 1/2^B), list-decoding radius
2. **Folding error**: Per-round error from random folding challenges
3. **Query error**: alpha^s where alpha depends on rate, s = n_queries

Total security bits = pow_bits + log_blowup_factor * n_queries

**Default config** (PcsConfig::default): 10 + 1*3 = 13 bits. **TEST ONLY.**
**Production target**: 100+ bits (e.g., pow_bits=26, log_blowup=4, n_queries=20)

## Security Invariants

INVARIANT-FRI-1: Every FRI layer commitment must be mixed into the
Fiat-Shamir channel BEFORE drawing the folding alpha.

INVARIANT-FRI-2: The last layer polynomial degree must be strictly
checked against `log_last_layer_degree_bound`.

INVARIANT-FRI-3: Query positions must be sampled uniformly after all
commitments are mixed.

INVARIANT-FRI-4: Folding operations must use the correct twiddle factors
(conjugate y-coordinates for circle fold, x-coordinates for line fold).

INVARIANT-FRI-5: The domain chain must be correct: each folded domain
is derived from the previous by the squaring map.

## Forbidden Actions

In this domain, agents must NEVER:
- Change FRI parameter range bounds without re-deriving security analysis
- Skip any FRI layer verification step
- Modify the folding operation (alpha mixing) without mathematical proof
- Reorder commitment-then-challenge in the Fiat-Shamir transcript
- Change the last layer degree check to be non-strict
