# ZK Soundness Mitigation Plan

Date: 2026-06-03
Branch: `infinite/zk-soundness-mitigations`
Status: Accepted by staff review and implemented

## Review Decision

The plan is accepted after review by the required staff roles.

- Math Reviewer: AGREE. No `SOUNDNESS-ESCALATION` required.
- Crypto Specialist: AGREE after the direct PCS LogUp statistical activation bypass was closed.
- Implementer: AGREE after the state-machine example was changed to assert fail-closed behavior and enum additions were accepted as intentional security failure states.

## Scope

This plan mitigates three soundness issues in the adjusted ZK STWO implementation:

1. FRI query soundness.
2. ZK witness-randomization audit geometry.
3. Private LogUp statistical aggregate activation under the current API.

## Mitigation 1: FRI Query Soundness

Accepted plan:

- Draw exactly `n_queries` unique Fiat-Shamir FRI query positions.
- Deterministically resample from the channel until enough distinct positions are collected.
- Reject impossible query configurations where `n_queries > 2^first_layer_log_size`.
- Reject externally supplied FRI query positions unless:
  - `query_positions.len() == config.n_queries`
  - all positions are unique
  - all positions are in the first-layer domain

Implementation:

- `draw_queries` now returns unique query positions and rejects impossible counts.
- `FriVerifier::commit` rejects impossible verifier configurations.
- `FriVerifier::decommit_on_query_positions` validates exact count, uniqueness, and domain membership before building `Queries`.
- Added `FriVerificationError::InvalidQueryPositions` as an intentional new rejection state.

## Mitigation 2: ZK Witness-Randomization Geometry Audit

Accepted plan:

- Add a shared geometry-aware sampled-metadata builder for private PCS audit paths.
- Use actual committed tree height and committed column log size in prover and verifier PCS paths.
- For OODS openings, use:
  - `raw_pcs_point.repeated_double(lifting_log_size - committed_column_log_size)`
- For FRI/Merkle openings, mirror `MerkleProverLifted::decommit`:
  - start from the first-layer FRI query position
  - preprocess query positions when tree height is smaller than lifting height
  - project with `(pos >> (shift + 1) << 1) + (pos & 1)`
  - convert the projected index into the committed column-domain point
- Preserve `ZkQueryClosureEntry` shape.
- Validate private degree metadata against actual opening geometry:
  - reject missing or duplicate geometry
  - reject missing or duplicate private degree bounds
  - reject non-singleton or non-private bounds
  - compute required randomized degree with `zk_private_column_randomized_log_degree`
  - require `bound.log_degree_bound >= required`
  - require `bound.log_degree_bound + log_blowup_factor == committed_column_log_size`

Implementation:

- Added `ZkPrivateColumnOpeningGeometry`.
- Added geometry-aware sampled-metadata construction in `core::zk`.
- Prover PCS derives audit metadata using actual committed tree and polynomial domains.
- Verifier PCS derives audit metadata using verifier-owned tree height and column log sizes.
- The legacy public wrappers remain for existing non-PCS helper tests; private PCS proof paths use the geometry-aware helper.

## Mitigation 3: Private LogUp Statistical Aggregates

Accepted plan:

- Fail closed under the current API.
- Reject activation if either:
  - `logup_statistical_security_budgets` is nonempty, or
  - `metadata.logup_statistical_security_budget_hash` is not the default empty hash.
- Apply before ZK metadata mixing or challenge derivation.
- Apply at both top-level STARK ZK and direct ZK PCS entrypoints.
- Preserve public/non-ZK LogUp and ordinary public ZK.

Implementation:

- Added `reject_zk_logup_statistical_security_budget_activation`.
- Top-level `prove_zk_ex` rejects activated statistical LogUp before transcript mixing.
- Top-level `verify_zk_ex` rejects both verifier-config activation and proof-echoed metadata activation before transcript mixing.
- Direct PCS `prove_values_zk_with_fri_batch_mask` rejects activated statistical LogUp before PCS proving.
- Direct PCS verifier rejects both verifier-config activation and proof-echoed metadata activation before metadata equality checks or transcript work.
- State-machine statistical private LogUp examples now assert the fail-closed behavior. Future activation tests remain ignored and return early while the mode is unsupported.

Future reactivation requires a reviewed additive sidecar carrying:

- `interaction_index`
- `claim_index`
- `aggregate_id`
- `correction_ref`

## Verification

Passed locally:

- `cargo test -p stwo --features prover --lib`
- `cargo test -p stwo --features prover --test zk_logup_budget`
- `cargo test -p stwo-examples`
- `cargo test -p stwo-examples state_machine_statistical_logup -- --include-ignored`
- `cargo test -p stwo-constraint-framework`
- `cargo check -p stwo --no-default-features`
- `git diff --check`

Targeted tests added or updated:

- FRI duplicate, wrong-count, out-of-domain, impossible-count, and resampling tests.
- Geometry-aware sampled metadata tests for committed-domain projection, legacy raw-point divergence, too-small degree bounds, missing bounds, duplicate bounds, and committed log-size mismatch.
- Verifier audit geometry acceptance test for same-height openings.
- Top-level and direct PCS LogUp statistical fail-closed tests, including coherent verifier budget activation and proof-only metadata hash activation.
- State-machine example fail-closed test for unsupported private LogUp statistical aggregates.

## Acceptance Notes

The public error enums gained intentional new failure states for this security fix. These are accepted as semver-relevant but necessary to make invalid proof/config states explicit:

- `FriVerificationError::InvalidQueryPositions`
- `ProvingError::ZkLogupStatisticalAggregatePolicy`
- `VerificationError::ZkLogupStatisticalAggregatePolicy`
- `ZkProvingConfigError::LogupStatisticalAggregatePolicy`
- additional `ZkSampleMetadataBuildError` variants for geometry and degree-bound validation

No reviewer requested a `SOUNDNESS-ESCALATION`.
