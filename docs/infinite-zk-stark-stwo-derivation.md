# STWO-specific ZK STARK derivation

Status: `REVIEW_READY`

This document instantiates the paper's witness-randomization and FRI-masking
requirements for STWO's current proof structure. It is a review artifact, not
an activation switch.

## 1. Names

- `H`: original trace domain for a private witness column.
- `D`: committed evaluation domain for that column.
- `v_H`: STWO circle-domain vanishing polynomial for `H`.
- `r_i`: prover-private base-field randomizer polynomial for private column
  `i`.
- `h_i`: dimension of the selected randomizer space for private column `i`.
- `q_i`: verifier-computed number of base-field query-closure functionals for
  private column `i`.
- `R`: prover-private extension-field FRI batch-mask polynomial.
- `H_batch`: first FRI-layer polynomial `raw_quotient + R`.

## 2. Non-assumptions

- Do not identify the paper's Protocol 3 `d` with `COMPOSITION_LOG_SPLIT`.
- Do not identify Protocol 3 `d` with the number of halves returned by
  `split_at_mid`.
- Do not assume `h_i = |H|` is sufficient without computing the actual STWO
  query closure and rank.
- Do not apply Phase 2 to private lookup, permutation, fractional, LogUp, or
  multiset-argument columns. Those require a separate Appendix A treatment.

## 3. STWO split query expansion

STWO's current composition split is an FFT-basis split:

```text
p(z) = p_left(z) + pi^{L-2}(z.x) * p_right(z)
```

The paper's Protocol 3 split is written as:

```text
q(X) = sum_j X^(j-1) * q_j(X^d)
```

The paper's implicit `d`-root query expansion applies when quotient components
are opened as separate oracles. STWO's current verifier does not open
`split_at_mid` pieces as private witness-column oracles. It checks quotient
answers through committed trace openings, sampled OODS values, and the FRI
first layer.

Therefore the current STWO witness-randomizer query closure has no extra
`split_at_mid` multiplier. It consists of:

- extension-field sampled values involving the private column;
- translated base-domain sampled values involving the private column;
- FRI query openings of the private column;
- zero future quotient-component preimages while the current proof format is
  unchanged.

If STWO later exposes quotient-component openings that depend on private
columns, this derivation is invalid until the corresponding preimage
functionals are added to `q_i`.

## 4. Query-closure construction

For every private column range, the verifier builds a deduplicated set of
base-field linear functionals:

```text
(column_range, query_kind, domain_id, point_or_position_encoding, coordinate_index)
```

Rules:

- A `QM31` sampled point contributes four coordinate functionals.
- A base-field translated-domain point contributes one functional.
- A FRI query position contributes one functional.
- A future quotient-component preimage contributes the functionals induced by
  that preimage.
- Duplicate tuples are counted once.

The count of this set is `q_i`. The prover cannot supply `q_i`; it is derived
from verifier-owned privacy metadata, sampled-point metadata, domain metadata,
and public FRI query positions.

## 5. Circle randomizer space and rank condition

For private base-field column `w_i`, the prover commits to:

```text
w_hat_i(P) = w_i(P) + v_H(P) * r_i(P)
```

The selected randomizer space is exact, ordered, and hash-bound:

- Basis family: STWO `CircleCoefficients` FFT basis, stored in the existing
  bit-reversed coefficient order. This is the tensor-product basis documented
  in `crates/stwo/src/prover/poly/circle/poly.rs`:
  `y, x, pi(x), pi^2(x), ...`, where `pi(x) = 2x^2 - 1`.
- Randomizer dimension: `h_i` is the number of active basis coefficients.
- Coefficient storage: the concrete `CircleCoefficients` vector length is
  `next_power_of_two(h_i)`, or the reviewed randomized degree vector length if
  larger. Coefficients at basis indices `0..h_i` are sampled uniformly from
  `M31`; all remaining coefficients are zero.
- Sampling: each active coefficient is sampled by exact rejection sampling from
  a prover-private `RngCore + CryptoRng`. No modulo reduction, transcript seed,
  deterministic test RNG, or public randomness is allowed in production.

The initial implementation uses a canonical evaluation-domain construction for
`v_H * r_i`, not an ad hoc symbolic multiply:

1. Extend `w_i` to the reviewed randomized private-column degree bound.
2. Build `r_i` in the exact basis above.
3. Evaluate `r_i` on the reviewed randomized commitment domain `D_i`.
4. For each `P in D_i`, compute
   `delta_i(P) = coset_vanishing(H_i.coset, P) * r_i(P)`.
5. Interpolate `delta_i` over `D_i` with STWO's reviewed circle interpolation.
6. Add the interpolated `delta_i` coefficients to the extended `w_i`
   coefficients in the same FFT basis, producing `w_hat_i`.

This construction is the only Phase 2 construction until a later staff-reviewed
coefficient-multiply implementation proves byte-for-byte equivalence.

Correctness over the trace domain follows because `v_H(P) = 0` for every
`P in H`.

Privacy on the query closure requires the verifier to form the `q_i x h_i`
base-field evaluation matrix:

```text
M_i[a, b] = functional_a(v_H * basis_b)
```

Activation must reject unless:

```text
rank(M_i) = q_i
```

This rank condition is the STWO-specific replacement for blindly importing the
paper's symbolic Protocol 3 degree parameter. The inequality `h_i >= q_i` is a
cheap precheck, not a proof.

The rank check is a pre-proof activation check. It must complete before the
prover serializes or logs any private-column opening, aux data, or benchmark
artifact for the proof attempt.

## 6. Private query exclusion and OODS exclusion

Every Phase 2 OODS point is drawn by public deterministic rejection sampling
from the Fiat-Shamir channel. The exclusion set includes:

- `H`;
- every committed evaluation domain `D`;
- transition/lookup translated domains;
- denominator-degenerate line cases;
- future quotient-component preimage sets.

Prover and verifier consume the same rejected candidates in the same order. A
bounded retry limit fails closed.

The same fail-closed rule applies to private-column PCS/FRI openings. Since
`w_hat_i(P) = w_i(P)` on `H`, any private-column opening at a point where
`v_H(P) = 0` reveals the raw private witness value. Therefore Phase 2 must do
one of the following before any private opening is serialized:

- prove from verifier-owned domain metadata that every private commitment and
  FRI query domain is disjoint from the corresponding trace domain `H`; or
- derive the private-column query closure before decommitment and determinis-
  tically reject/resample every private opening whose point is in `H`.

Post-proof rejection is not sufficient for zero-knowledge. The rejection or
disjointness proof is part of the public verifier algorithm and transcript
definition.

## 7. Degree metadata

For each private column:

```text
randomized_log_degree_i >= max(original_log_degree_i, log2_ceil(|H| + h_i))
```

The following must consume verifier-owned randomized degree metadata:

- `Component::trace_log_degree_bounds`;
- `FrameworkComponent`;
- `FrameworkEval::max_constraint_log_degree_bound`;
- `EvaluationMode::infer`;
- composition accumulator sizing;
- PCS column log sizes;
- verifier commitment log sizes.

Proof metadata may echo these values, but verification rejects any mismatch
against verifier-owned metadata.

Randomized degree propagation is explicit and ZK-only:

- Default `prove`, `prove_ex`, `verify`, `verify_ex`, `prove_values`, and
  `verify_values` continue to use the existing degree inference and proof
  semantics.
- ZK APIs use a verifier-owned degree profile. Non-ZK callers must not observe
  randomized bounds, altered transcript order, or changed proof bytes.
- For a private column, `randomized_log_degree_i` is the smallest reviewed log
  bound whose basis can represent `w_i + v_H * r_i`; the conservative first
  candidate is `ceil_log2(|H_i| + h_i)`.
- ZK composition degree is computed by running the existing component/framework
  degree formulas with private-column trace bounds replaced by their
  randomized bounds. No default component degree method may be changed in a way
  that affects non-ZK proofs.
- The split composition degree bound is derived from that ZK composition bound
  using the reviewed STWO `split_at_mid` identity only while quotient-component
  openings remain internal to the PCS/FRI path.
- `quotient_degree_bounds` in verifier-owned metadata are the exact public
  bounds for the ZK composition split or quotient columns committed in the ZK
  proof.
- Activation rejects if prover-computed bounds, proof metadata, verifier-owned
  metadata, and commitment-tree column log sizes disagree.

For the explicit PCS FRI batch mask, the first accepted STWO rule is
conservative:

```text
h_batch = 2^(fri_first_layer_log_size - log_blowup_factor)
deg(R) < h_batch
deg(H_batch) < h_batch
```

`R` is sampled uniformly over `SecureField` coefficients in that bound and is
committed over `CanonicCoset::new(fri_first_layer_log_size).circle_domain()`.
Activation rejects unless `h_batch` is at least the reviewed raw quotient bound
after witness randomization and at least the reviewed Protocol 2 mask budget.
This is intentionally allowed to over-approximate the paper's
`ceil((d + 1) / d * h)` until STWO has a tighter signed derivation.

## 7.1 Canonical public hash inputs

The public hashes in ZK metadata are not informal review placeholders after
activation. They are deterministic no-std byte encodings with these common
rules:

- Digest: Blake2s-256 with the exact 32-byte digest output.
- ASCII domain tag.
- `u32` version in little-endian.
- Length-prefixed vectors with `u64` lengths in little-endian.
- `usize` values encoded as checked `u64`.
- Entries sorted by the same canonical order used by the Rust types.
- No map/hash-map iteration order.
- `BaseField`/`M31` values encoded as their canonical representative
  `0..P` as `u32` little-endian.
- `CirclePoint<BaseField>` encoded as `x` then `y`.
- `Coset` encoded as:
  `log_size`, `initial_index`, `initial`, `step_size`, `step`, where point
  indices are checked `u64` little-endian and points use the rule above.

`private_column_scope_hash` input:

```text
"stwo.zk.private-column-scope.v1"
version
entry_count
for each sorted entry:
  tree_index, column_start, column_end, usage_tag
```

`randomizer_space_hash` input:

```text
"stwo.zk.randomizer-space.v1"
version
private_column_scope_hash
entry_count
for each sorted randomizer-space entry:
  range.tree_index, range.column_start, range.column_end
  trace_domain_coset_encoding
  randomized_log_degree
  h_i
  basis_id = "circle-fft-bit-reversed"
  construction_id = "eval-coset-vanishing-times-r-interpolate"
  rank_matrix_id = "base-field-functional-matrix-v1"
```

The first implementation uses one global `h_witness` in `ZkDegreeProfile` as a
minimum dimension policy, but `randomizer_space_hash` still binds a sorted
per-private-column vector. This avoids ambiguity when private columns later use
different trace domains or randomized degree bounds.

`split_derivation_hash` input:

```text
"stwo.zk.split-derivation.v1"
version
composition_log_split
split_identity = "p(z)=left(z)+pi^(L-2)(z.x)*right(z)"
quotient_opening_model = "internal-pcs-fri-only"
h_batch_rule = "fri-first-layer-degree-bound"
```

Zero hashes are valid only for the public-only Phase 1 profile. Witness
randomization activation rejects any zero hash.

## 8. Protocol 2 FRI batch mask

The independent extension-field polynomial `R` remains mandatory after witness
randomization.

Ordering:

1. Mix verifier-owned ZK metadata and degree/profile hashes before any affected
   challenge.
2. Commit randomized trace and quotient/split oracles.
3. Draw OODS sampled values from the metadata-bound Fiat-Shamir stream.
4. Mix OODS sampled values of randomized oracles.
5. Commit `R`.
6. Draw the FRI quotient-batching challenge.
7. Commit FRI layers for `H_batch = raw_quotient + R`.
8. Sample FRI query positions from the metadata- and `R`-bound transcript.
9. Verify first-layer query answers after adding `R(query)` to raw quotient
   answers.

The verifier must never receive a value/mask pair that reveals a private
witness value. `R` is an independently committed oracle, not a per-value mask
that the verifier subtracts.

## 9. Current implementation status

Implemented:

- explicit ZK proof/config types;
- verifier-owned privacy/degree config;
- verifier-owned metadata comparison helper;
- deterministic public OODS rejection helper;
- separate FRI `R` oracle commit/decommit/authentication helpers;
- ZK-only PCS `R` ordering and `H_batch` first-layer answer handling;
- Phase 2 witness-randomization profile metadata, including
  `randomizer_space_hash`, `private_column_scope_hash`, `h_witness`, and
  private-column degree bounds;
- Phase 3 quotient-integration profile metadata, including
  `split_derivation_hash`, `h_batch`, FRI first-layer log size, and quotient
  degree bounds;
- fail-closed prover validation that rejects Phase 2/3 activation even when all
  review evidence is present.

Not implemented:

- witness polynomial randomization;
- query-closure construction from real component metadata;
- rank check for `M_i`;
- randomized composition degree metadata;
- full `prove_zk_ex` / `verify_zk_ex`;
- private lookup/permutation Appendix A treatment.

## 10. Candidate activation rule

Semantic witness randomization may be activated only when
`ZkProvingConfig::validate_witness_randomization_activation()` can succeed
without the terminal activation block. Function names in code must not contain
`phase`.

That requires:

- review evidence for all derivation gates;
- non-zero `randomizer_space_hash`;
- non-zero `private_column_scope_hash`;
- non-zero `split_derivation_hash`;
- non-empty private-column degree bounds;
- non-empty quotient degree bounds;
- verifier-derived `q_i` for every private column;
- `h_i >= q_i` for every private column;
- `rank(M_i) = q_i` for every private column;
- equality between `ZkDegreeProfile.h_witness` and
  `ZkWitnessRandomizationProfile.h_witness`;
- equality between `ZkDegreeProfile.h_batch` and
  `ZkQuotientIntegrationProfile.h_batch`;
- equality between `ZkDegreeProfile.fri_first_layer_log_size` and
  `ZkQuotientIntegrationProfile.fri_first_layer_log_size`.
- metadata and degree-profile hashes mixed into Fiat-Shamir before OODS,
  batching, FRI commitments, and query sampling;
- private-column commitment/FRI domains proven disjoint from `H`, or
  deterministic pre-decommit rejection/resampling for every private opening
  where `v_H = 0`;
- no secret randomizer coefficients, seeds, `v_H * r_i` coefficients, raw
  private openings, or value/mask pairs in proof data, aux data, logs, debug
  output, benchmark output, or serialization.

## 11. Secrecy and instrumentation

Secret witness-randomization material must remain prover-local:

- `r_i` coefficients, RNG state, seeds, `v_H * r_i` coefficients, and
  intermediate randomized deltas do not implement public `Serialize`,
  `Deserialize`, or `Debug`.
- Benchmarks and tracing may report counts, durations, byte sizes, domain
  sizes, and aggregate allocation sizes only.
- Benchmarks and tracing must not emit field elements from private witnesses,
  randomizers, masks, sampled private openings, or randomized deltas.
- Aux data may contain Merkle/FRI diagnostics required for local checking, but
  never secret coefficients or enough unmasked value/mask material to recover a
  private opening.

## 12. Phase 2/3 performance gate

Witness randomization must be measured only through explicit ZK entry points.
Default STWO benchmark baselines remain unchanged and are rerun as a control.

Required Phase 2/3 measurements:

- query-closure construction time and entry counts;
- rank-matrix construction time and dimensions;
- rank-check time;
- witness randomizer coefficient sampling time;
- `v_H * r_i` evaluation/interpolation/addition time;
- randomized private commitment evaluation and Merkle time;
- randomized composition generation time;
- quotient/split generation time under randomized degree metadata;
- `R` sampling/evaluation/commit/opening/verification costs;
- proof byte delta, aux byte delta, and allocation-count proxy.

Regression gates from the plan remain active. A ZK-path regression above the
threshold requires Performance Reviewer approval; a default-path regression is
an implementation blocker.
