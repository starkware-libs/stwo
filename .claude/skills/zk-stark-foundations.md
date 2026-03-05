---
name: zk-stark-foundations
description: >
  STWO-specific STARK architecture and protocol flow. Provides implementation
  locations, security parameter configuration, proof flow mapping, and
  invariants. Use when working on proof system code, reviewing constraint
  logic, modifying FRI parameters, or auditing soundness.
---

# ZK-STARK Foundations for STWO

## Canonical Theory Sources

- `.agents/papers/llm/INDEX.llm.md` — notation harmonization and source map
- `.agents/papers/llm/Circle_STARKs.llm.md` — core circle STARK AIR/FRI theory
- `.agents/papers/llm/Stwo_Whitepaper.llm.md` — STWO protocol layering and parameterization

## STWO Protocol Architecture

### AIR Constraints

Constraint polynomial identities over the trace
(`Circle_STARKs.llm.md` -> `e:overall:identity`):

```
P_i(s_i, p_1, ..., p_w, p_1 o T, ..., p_w o T) = 0   over H
```

**Implementation**: `crates/constraint-framework/src/lib.rs` — `EvalAtRow` trait

### FRI Low-Degree Test

Circle FRI variant operating over circle group domains
(`Circle_STARKs.llm.md` -> `prot:IOP:proximity`):
- **Verifier**: `crates/stwo/src/core/fri.rs` — `FriVerifier`
- **Prover**: `crates/stwo/src/prover/fri.rs` — `FriProver`

### Polynomial Commitment Scheme

FRI-based PCS with Merkle-committed evaluations and DEEP quotient openings:
- **Verifier**: `crates/stwo/src/core/pcs/`
- **Prover**: `crates/stwo/src/prover/pcs/`

### DEEP-ALI (Algebraic Linking)

OODS point sampling + DEEP quotient `(p(x) - p(z)) / (x - z)` + FRI.
Links committed evaluations to constraint identity
(`Circle_STARKs.llm.md` -> `prop:deep:quotients`, `thm:AIR:soundness`).
- **Quotients**: `crates/stwo/src/core/pcs/quotients.rs`
- **Verifier**: `crates/stwo/src/core/verifier.rs` — `verify()`

### Proof of Work (Grinding)

Pre-query PoW nonce adding `pow_bits` of security.
- `crates/stwo/src/core/proof_of_work.rs` (verifier)
- `crates/stwo/src/prover/backend/*/grind.rs` (prover)

## Security Parameters

| Parameter | Symbol | Config Field | Security Impact |
|-----------|--------|-------------|-----------------|
| Blowup factor | 2^B | `fri_config.log_blowup_factor` | Rate = 1/2^B. Higher = more secure but slower |
| FRI queries | s | `fri_config.n_queries` | Each query adds `log_blowup_factor` security bits |
| Grinding bits | g | `pcs_config.pow_bits` | Adds `g` bits of security |
| Last layer degree | - | `fri_config.log_last_layer_degree_bound` | Degree bound for the final FRI layer |
| Total security | - | `pcs_config.security_bits()` | = pow_bits + log_blowup_factor * n_queries |

**WARNING**: Default PcsConfig has only ~13 bits of security (test config).
Production must use appropriate parameters. See DIVERGENCE-007.

## STARK Proof Flow

```
1. Trace Generation      → Witness polynomials p_1,...,p_w
2. Trace Commitment       → Merkle commit evaluations on domain D
3. Constraint Evaluation  → Composition polynomial from random combination
4. Composition Commitment → Merkle commit composition evaluations
5. OODS Challenge         → Random point z on circle
6. DEEP Quotients         → (p(x) - p(z)) / vanishing(x,z) for each poly
7. FRI                    → Prove quotients are low-degree
8. Proof of Work          → Find grinding nonce
9. Query Phase            → Verify folding chain at random positions
```

**Implementation entry points**:
- Prover: `crates/stwo/src/prover/mod.rs` — `prove()`
- Verifier: `crates/stwo/src/core/verifier.rs` — `verify()`

## Security Invariants

INVARIANT-ZK-1: The composition polynomial identity must hold over the
entire trace domain H. A single violated constraint at any row breaks soundness.

INVARIANT-ZK-2: The OODS point must be sampled uniformly from the circle
group over the secure field, excluding the commitment domain.

INVARIANT-ZK-3: FRI folding challenges must be drawn from the Fiat-Shamir
transcript AFTER mixing in the previous layer's commitment.

INVARIANT-ZK-4: The Merkle tree commitment must be binding — the hash
function must be collision-resistant.

INVARIANT-ZK-5: The proof-of-work nonce must be verified by the verifier
before accepting the proof.

## Forbidden Actions

In this domain, agents must NEVER:
- Reduce security parameters without explicit human approval and documented justification
- Skip the OODS sampling step or use a deterministic point
- Reorder the Fiat-Shamir transcript (commitment → challenge ordering is critical)
- Remove or weaken any FRI verification check
- Accept a proof without verifying all layers including the last layer
