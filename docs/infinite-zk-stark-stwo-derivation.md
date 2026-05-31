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

The selected randomizer space is the first `h_i` basis elements of STWO's
base-field circle-polynomial basis for the configured randomized column degree.
The prover samples the `h_i` coefficients uniformly from `M31` using
rejection sampling from a prover-private `CryptoRng`.

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

## 6. OODS exclusion

Every Phase 2 OODS point is drawn by public deterministic rejection sampling
from the Fiat-Shamir channel. The exclusion set includes:

- `H`;
- every committed evaluation domain `D`;
- transition/lookup translated domains;
- denominator-degenerate line cases;
- future quotient-component preimage sets.

Prover and verifier consume the same rejected candidates in the same order. A
bounded retry limit fails closed.

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

## 8. Protocol 2 FRI batch mask

The independent extension-field polynomial `R` remains mandatory after witness
randomization.

Ordering:

1. Commit randomized trace and quotient/split oracles.
2. Draw and mix OODS sampled values of randomized oracles.
3. Commit `R`.
4. Draw the FRI quotient-batching challenge.
5. Commit FRI layers for `H_batch = raw_quotient + R`.
6. Verify first-layer query answers after adding `R(query)` to raw quotient
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
`ZkProvingConfig::validate_for_phase_2_and_3()` can succeed without the
terminal activation block.

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
