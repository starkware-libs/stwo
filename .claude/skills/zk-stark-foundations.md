---
name: zk-stark-foundations
description: >
  Core ZK-STARK theory required before working on any STWO component.
  Load this skill when: working on any proof system code, reviewing
  constraint logic, modifying FRI parameters, or auditing soundness.
  This provides the theoretical vocabulary for all other STWO skills.
---

# ZK-STARK Foundations for STWO

## Purpose

Provides the theoretical foundation for STARKs (Scalable Transparent Arguments
of Knowledge) as implemented in STWO. Every agent working on STWO must
understand these concepts before modifying proof-system code.

## Core Concepts

### STARK Overview

A STARK is an interactive oracle proof (IOP) made non-interactive via
Fiat-Shamir. It proves that a prover knows a witness satisfying an
Algebraic Intermediate Representation (AIR) — a set of polynomial
constraints over a trace.

Key properties:
- **Transparency**: No trusted setup. Security is information-theoretic.
- **Scalability**: Prover is quasi-linear, verifier is polylogarithmic.
- **Post-quantum**: No reliance on discrete log or factoring assumptions.

### Algebraic Intermediate Representation (AIR)

An AIR defines constraints as polynomial identities over a trace:

```
P_i(s_i, p_1, ..., p_w, p_1 o T, ..., p_w o T) = 0   over H
```

Where:
- `p_1, ..., p_w` are trace polynomials (the witness)
- `T` is the trace step operator (group translation)
- `s_i` are selector polynomials (activation subdomains)
- `H` is the trace domain (evaluation domain of the trace)
- `P_i` are constraint polynomials of bounded degree

**Source**: Circle STARK paper Section 5, Eq. 1

**Implementation**: `crates/constraint-framework/src/lib.rs` — `EvalAtRow` trait

### FRI (Fast Reed-Solomon IOP of Proximity)

FRI is the core proximity test in STARKs. It proves that a committed
function is close to a low-degree polynomial.

Protocol structure:
1. **Commit phase**: Prover sends Merkle commitments to polynomial evaluations
   at each folding round. Verifier sends random folding challenges.
2. **Query phase**: Verifier samples random positions and checks the
   folding chain is consistent.

**Soundness**: The probability of a cheating prover succeeding decreases
exponentially with the number of queries.

**Source**: Circle STARK paper Section 6

**Implementation**: `crates/stwo/src/core/fri.rs` (verifier), `crates/stwo/src/prover/fri.rs` (prover)

### Polynomial Commitment Scheme (PCS)

STWO uses FRI as the basis for its polynomial commitment scheme:

1. Prover evaluates polynomials on a domain D (larger than trace domain H
   by a blowup factor).
2. Evaluations are committed via Merkle trees.
3. Openings at challenged points are proved via DEEP quotients + FRI.

**Source**: `crates/stwo/src/core/pcs/mod.rs` lines 1-8 (module doc comment)

**Implementation**: `crates/stwo/src/core/pcs/` (verifier), `crates/stwo/src/prover/pcs/` (prover)

### DEEP-ALI (Algebraic Linking Identity)

After the prover commits to trace and composition polynomials:

1. Verifier challenges with a random Out-Of-Domain Sampling (OODS) point.
2. Prover evaluates polynomials at the OODS point.
3. DEEP quotients `(p(x) - p(z)) / (x - z)` are constructed.
4. FRI proves these quotients are low-degree.

This links the committed evaluations to the algebraic constraint identity.

**Source**: Circle STARK paper Section 5 (DEEP Algebraic Linking)

**Implementation**: `crates/stwo/src/core/pcs/quotients.rs`, `crates/stwo/src/core/verifier.rs`

### Proof of Work (Grinding)

Before FRI queries, the prover must find a nonce satisfying a hash
difficulty target. This adds `pow_bits` of security cheaply.

**Implementation**: `crates/stwo/src/core/proof_of_work.rs`, `crates/stwo/src/prover/backend/*/grind.rs`

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
- Prover: `crates/stwo/src/prover/mod.rs:29` — `prove()`
- Verifier: `crates/stwo/src/core/verifier.rs:19` — `verify()`

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
