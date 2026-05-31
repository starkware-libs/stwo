# Infinite STWO ZK STARK Derivation Gates

This document is the Phase 0 soundness gate for implementing the
Habock-Kindi zero-knowledge STARK construction in STWO.

It is intentionally a blocker checklist, not a proof. Code that changes STWO
proof-system semantics must not land until the relevant gate is completed and
reviewed.

## Gate 1: STWO split query expansion

Status: `REVIEW_READY`

Required result:

- Derive the STWO query-expansion factor for `split_at_mid`, if any.
- Do not map the paper's Protocol 3 `d` to `COMPOSITION_LOG_SPLIT`, `2`, or
  `2^COMPOSITION_LOG_SPLIT`.
- Account for STWO's identity:

```text
p(z) = p_left(z) + pi^{L-2}(z.x) * p_right(z)
```

Required review:

- Math Reviewer.
- Crypto Specialist.

## Gate 2: Circle randomizer space

Status: `REVIEW_READY`

Required result:

- Define the exact STWO circle-polynomial randomizer space.
- Define `h_witness` and its basis.
- Define exact uniform base-field coefficient sampling for witness
  randomizers.
- Define the coefficient-level construction of `v_H * r_i`.
- Prove `w_hat_i(p) = w_i(p)` for all `p in H`.
- Prove witness independence for the reviewed OODS/FRI query closure.

Required review:

- Math Reviewer.
- Security/Pentest Reviewer.

## Gate 3: OODS and domain exclusion

Status: `REVIEW_READY`

Required result:

- Specify deterministic verifier/prover reject or resample behavior for OODS
  points colliding with:
  - trace domain `H`.
  - commitment domain `D`.
  - translated/shifted query domains.
  - line-denominator degeneracy cases.
- Resampling, if used, must be derived only from the public Fiat-Shamir stream.

Required review:

- Math Reviewer.
- Security/Pentest Reviewer.

## Gate 4: ZK-aware degree metadata

Status: `REVIEW_READY`

Required result:

- Update the design for:
  - `Component::trace_log_degree_bounds`.
  - `FrameworkComponent`.
  - `FrameworkEval::max_constraint_log_degree_bound`.
  - `EvaluationMode::infer`.
  - composition accumulator sizing.
  - PCS column log sizes.
  - verifier commitment log sizes.
- Show that prover and verifier use the same randomized degree profile.

Required review:

- Math Reviewer.
- Staff Rust Engineer.

## Gate 5: FRI batch mask degree

Status: `REVIEW_READY`

Required result:

- Define `h_batch`, the STWO equivalent of the paper's Protocol 2 batch-mask
  degree budget.
- Define the exact degree and commitment domain for `R`.
- Prove `H_batch(X) = raw_quotient(X) + R(X)` is checked against the reviewed
  FRI first-layer degree bound.
- Confirm `R` is sampled uniformly over `SecureField[X]`.

Required review:

- Math Reviewer.
- Crypto Specialist.
- Performance Reviewer for expected cost impact.

## Implementation rule

Before these gates are signed off, implementation may add only:

- explicit ZK API/proof types;
- verifier-owned public configuration types;
- benchmark/instrumentation scaffolding;
- Phase 1 separate `R` oracle mechanics that are explicitly not claimed as
  complete witness zero-knowledge.

Witness polynomial randomization and ZK degree-bound changes remain blocked
until Gates 1 through 5 are signed off.
