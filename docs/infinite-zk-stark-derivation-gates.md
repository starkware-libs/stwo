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
- Define the canonical construction of `v_H * r_i`; the first accepted
  construction is evaluation on the reviewed randomized commitment domain,
  pointwise multiplication by `coset_vanishing(H, P)`, interpolation, and
  coefficient addition in the same STWO FFT basis.
- Prove `w_hat_i(p) = w_i(p)` for all `p in H`.
- Compute the query-closure evaluation matrix `M_i`.
- Reject activation unless `rank(M_i) = q_i`.
- Define `randomizer_space_hash` as a canonical no-std-safe encoding of the
  basis, domain, dimension, construction algorithm, and rank-matrix algorithm.
- For multiple private columns, define `randomizer_space_hash` over a sorted
  per-`ZkColumnRange` randomizer-space vector. A single global `h_witness`
  policy is allowed only as a minimum dimension policy; the hash must still bind
  each private column's trace domain and randomized degree bound.

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
- Before any private-column opening is serialized, prove that every private
  commitment/FRI query domain is disjoint from `H`, or deterministically
  reject/resample every private opening where `v_H = 0`.
- Rejection after proof receipt is not sufficient for zero-knowledge.

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
- Add ZK-only degree APIs/config paths. Existing non-ZK degree inference,
  transcript order, proof bytes, and verifier behavior must remain unchanged.
- Define canonical no-std-safe encodings for `private_column_scope_hash`,
  `randomizer_space_hash`, and `split_derivation_hash`.
- Define the digest as Blake2s-256 and define field/circle/coset encodings:
  canonical `M31` as `u32` little-endian, circle point as `(x, y)`, and coset
  as log size, point indices, initial point, and step point.
- Mix verifier-owned ZK metadata and degree/profile hashes into Fiat-Shamir
  before OODS sampling, quotient batching, FRI commitments, and query
  sampling.

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
- The first accepted STWO rule is conservative:
  `h_batch = 2^(fri_first_layer_log_size - log_blowup_factor)`,
  `deg(R) < h_batch`, and `deg(H_batch) < h_batch`.
- Reject activation unless `h_batch` covers the reviewed raw quotient degree
  after witness randomization and the reviewed Protocol 2 mask budget.
- Keep the public-last-layer privacy guard from the Phase 1 `R` query sampler.

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

## Gate 7: proof-data secrecy and instrumentation

Status: `REVIEW_READY`

Required result:

- Secret randomizer coefficients, RNG state, seeds, `v_H * r_i` coefficients,
  intermediate randomized deltas, raw private openings, and value/mask pairs do
  not appear in proof data, aux data, serialization, debug output, logs, traces,
  or benchmark output.
- Secret material types do not implement public `Serialize`, `Deserialize`, or
  `Debug` unless a later staff review proves the implementation is safe.
- Instrumentation reports only aggregate counts, durations, byte sizes, domain
  sizes, allocation proxies, and proof-size deltas.

Required review:

- Security/Pentest Reviewer.
- Staff Rust Engineer.
- Performance Reviewer.

## Gate 8: Phase 2/3 performance controls

Status: `REVIEW_READY`

Required result:

- Phase 2/3 benchmarks run only through explicit ZK entry points.
- Default STWO `prove` / `verify` benchmarks are rerun as unchanged controls.
- Capture query-closure construction, rank-matrix construction, rank check,
  randomizer sampling, `v_H * r_i` construction, randomized commitment,
  randomized composition, quotient/split generation, `R` costs, proof bytes,
  aux bytes, and allocation-count proxy.
- Enforce the plan's regression thresholds. Any default-path regression is a
  blocker.

Required review:

- Performance Reviewer.
- Staff Rust Engineer.

## Implementation rule

Before these gates are signed off, implementation may add only:

- explicit ZK API/proof types;
- verifier-owned public configuration types;
- benchmark/instrumentation scaffolding;
- Phase 1 separate `R` oracle mechanics that are explicitly not claimed as
  complete witness zero-knowledge;
- fail-closed guardrails and tests.

Witness polynomial randomization and ZK degree-bound activation remain blocked
until Gates 1 through 8 are signed off and the terminal activation block is
reviewed for removal.
