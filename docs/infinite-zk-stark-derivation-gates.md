# Infinite STWO ZK STARK derivation gates

This document is the Phase 0 soundness gate for implementing the
Habock-Kindi zero-knowledge STARK construction in STWO.

Code that changes STWO proof-system semantics must not activate until every
gate below is completed, implemented where applicable, and reviewed.

## Gate 1: STWO query closure

Status: `REVIEW_READY`

Required result:

- Derive the STWO query-closure set from verifier-owned sampled-point,
  domain, private-column, and FRI-query metadata.
- Do not map the paper's Protocol 3 `d` to `COMPOSITION_LOG_SPLIT`, `2`, or
  `2^COMPOSITION_LOG_SPLIT`.
- Treat the current `split_at_mid` representation as inducing no additional
  witness-column openings while quotient components remain internal to the PCS
  and FRI path.
- Reopen this gate if explicit quotient-component openings are added.

Required review:

- Math Reviewer.
- Crypto Specialist.

## Gate 2: Circle randomizer space

Status: `REVIEW_READY`

Required result:

- Define the exact STWO circle-polynomial randomizer basis.
- Define `h_i` as the randomizer-space dimension for private column `i`.
- Define exact uniform base-field coefficient sampling for witness
  randomizers.
- Define the coefficient-level construction of `v_H * r_i`.
- Prove `w_hat_i(p) = w_i(p)` for all `p in H`.
- Compute the query-closure evaluation matrix `M_i`.
- Reject activation unless `rank(M_i) = q_i`.

Required review:

- Math Reviewer.
- Security/Pentest Reviewer.

## Gate 3: OODS and domain exclusion

Status: `REVIEW_READY`

Required result:

- Specify deterministic verifier/prover reject-or-resample behavior for OODS
  points colliding with:
  - trace domain `H`.
  - commitment domain `D`.
  - translated or shifted query domains.
  - line-denominator degeneracy cases.
  - future quotient-component preimage sets.
- Resampling, if used, must be derived only from the public Fiat-Shamir stream.
- A bounded retry limit must fail closed.

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
- Show that prover and verifier use the same verifier-owned randomized degree
  profile.
- Reject proof metadata that differs from verifier-owned metadata.

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
- Confirm `R` is committed before the quotient-batching challenge.

Required review:

- Math Reviewer.
- Crypto Specialist.
- Performance Reviewer for expected cost impact.

## Gate 6: private lookup/permutation exclusion

Status: `REVIEW_READY`

Required result:

- Exclude private columns that participate in lookup, permutation, fractional,
  LogUp, memory, grand-product, or other multiset arguments.
- Bind the exclusion decision into verifier-owned metadata with
  `private_column_scope_hash`.
- Reject missing or mismatched scope hashes.
- Treat Appendix A support as a separate future design.

Required review:

- Math Reviewer.
- Crypto Specialist.
- Security/Pentest Reviewer.

## Implementation rule

Before these gates are signed off, implementation may add only:

- explicit ZK API/proof types;
- verifier-owned public configuration types;
- benchmark/instrumentation scaffolding;
- Phase 1 separate `R` oracle mechanics that are explicitly not claimed as
  complete witness zero-knowledge;
- fail-closed guardrails and tests.

Witness polynomial randomization and ZK degree-bound activation remain blocked
until Gates 1 through 6 are signed off and the terminal activation block is
reviewed for removal.
