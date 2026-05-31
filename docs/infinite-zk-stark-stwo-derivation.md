# STWO-Specific ZK STARK Derivation Work Item

This document tracks the derivation required before implementing witness
randomization and ZK degree-bound changes.

Status: `REVIEW_READY`

## 1. Names

- `H`: trace domain for the target private witness column.
- `D`: commitment/evaluation domain used for the committed oracle.
- `h_witness`: witness-randomizer degree budget.
- `h_batch`: Protocol 2 FRI batch-mask degree budget.
- `R(X)`: extension-field-uniform FRI batch mask polynomial.
- `H_batch(X)`: FRI input polynomial `raw_quotient(X) + R(X)`.

## 2. Non-assumptions

- Do not identify paper Protocol 3 `d` with `COMPOSITION_LOG_SPLIT`.
- Do not identify paper Protocol 3 `d` with the number of halves returned by
  `split_at_mid`.
- Do not assume `h_witness = |H|` is sufficient until the STWO query-expansion
  factor is reviewed.

## 3. STWO split fact to derive against

STWO's current composition split is an FFT-basis split with:

```text
p(z) = p_left(z) + pi^{L-2}(z.x) * p_right(z)
```

The paper's Protocol 3 split is stated as:

```text
q(X) = sum_j X^(j-1) * q_j(X^d)
```

The derivation must show how many independent query constraints are induced by
STWO's `split_at_mid` representation, instead of importing the paper's
`d`-root implicit-query argument unchanged.

## 4. Randomizer-space obligation

Before code lands, define:

- the STWO circle-polynomial basis used for `r_i`;
- the coefficient-level representation of `v_H`;
- the coefficient-level construction of `v_H * r_i`;
- the rank argument for OODS and FRI query closure evaluations;
- the reviewed bound that maps those query counts to `h_witness`.

## 5. Degree metadata obligation

The implementation must make these paths ZK-aware:

- `Component::trace_log_degree_bounds`;
- `FrameworkComponent`;
- `FrameworkEval::max_constraint_log_degree_bound`;
- `EvaluationMode::infer`;
- composition accumulator sizing;
- PCS column log sizes;
- verifier commitment log sizes.

## 6. Protocol 2 handoff

After the STWO split derivation determines the quotient-component degree
profile, define:

- the STWO equivalent of the paper's `h_batch`;
- the exact degree/domain for `R`;
- the exact FRI first-layer bound for `H_batch`;
- the proof that `R` is committed before the quotient batching challenge and is
  never included in raw quotient batching.

## 7. Current implementation status

Implemented:

- explicit ZK proof/config types;
- verifier-owned privacy/degree config;
- deterministic public OODS rejection helper;
- separate FRI `R` oracle commit/decommit/authentication helpers;
- ZK-only PCS `R` ordering and `H_batch` first-layer answer handling.
- Phase 2 witness-randomization profile metadata, including
  `randomizer_space_hash`, `h_witness`, and private-column degree bounds.
- Phase 3 quotient-integration profile metadata, including
  `split_derivation_hash`, `h_batch`, FRI first-layer log size, and quotient
  degree bounds.
- fail-closed prover validation that rejects Phase 2/3 activation unless all
  derivation reviews and degree/profile commitments are present and consistent.

Blocked:

- witness polynomial randomization;
- randomized composition degree metadata;
- full `prove_zk_ex` / `verify_zk_ex`;
- claiming complete paper-level witness zero-knowledge.

## 8. Candidate Phase 2/3 activation rule

Semantic witness randomization may be implemented only when
`ZkProvingConfig::validate_for_phase_2_and_3()` succeeds.

That requires:

- review evidence for all five derivation gates;
- non-zero `randomizer_space_hash`;
- non-zero `split_derivation_hash`;
- non-empty private-column degree bounds;
- non-empty quotient degree bounds;
- equality between `ZkDegreeProfile.h_witness` and
  `ZkWitnessRandomizationProfile.h_witness`;
- equality between `ZkDegreeProfile.h_batch` and
  `ZkQuotientIntegrationProfile.h_batch`;
- equality between `ZkDegreeProfile.fri_first_layer_log_size` and
  `ZkQuotientIntegrationProfile.fri_first_layer_log_size`.

This is an activation rule, not a substitute for the missing derivation.
