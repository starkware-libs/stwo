# Infinite/STWO statistical-ZK LogUp integration plan

## Scope

This plan covers option 1 for making STWO LogUp-heavy AIRs usable in Infinite with private witnesses:

- Keep the paper-style STARK witness, quotient, and FRI randomization path for ordinary private trace data.
- Replace public per-component private LogUp `claimed_sum` leakage with a statistical-ZK aggregate LogUp protocol.
- Build the implementation from AIR privacy declarations and canonical STWO metadata, not example-specific constants.
- Keep the public STWO prover/verifier path unchanged and additive.

The plan is grounded in:

- `/Users/ehjc/workspace/projects/firecrawl-py/zk-stark/scraped_document.md`
- IACR ePrint 2024/1037, "A note on adding zero-knowledge to STARKs"
- IACR ePrint 2022/1530, "Multivariate lookups based on logarithmic derivatives"
- IACR ePrint 2023/1284, "Improving logarithmic derivative lookups using GKR"
- IACR ePrint 2024/1075, "TaSSLE: Lasso for the commitment-phobic"

## Staff conclusion

Statistical ZK is good enough for Infinite if we enforce a quantified leakage budget and fail closed when the budget is not met.

This is not the same as the paper's perfect-ZK simple AIR construction. The paper supports perfect ZK for ordinary AIR polynomial randomization, but combined lookup, permutation, and fractional-decomposition arguments require extra care and can make perfect ZK too costly. For STWO LogUp, the correct practical direction is statistical ZK with explicit leakage accounting.

Minimum target:

- `>= 100` bits aggregate statistical privacy for production Infinite proofs.
- Prefer `>= 110` bits when proof volume or private lookup volume is high.
- Reject configurations below the configured threshold.
- Do not claim `128` bits over QM31 unless the computed budget actually supports it.

## Soundness escalation

```text
SOUNDNESS-ESCALATION:
  File: planned LogUp/private interaction integration
  Change: hide private LogUp claimed_sum scalars using aggregate statistical-ZK accumulator
  Invariant at risk: lookup multiset consistency and witness privacy
  Paper reference: /Users/ehjc/workspace/projects/firecrawl-py/zk-stark/scraped_document.md, Section 3 and Appendix A
  Code location: crates/constraint-framework/src/logup.rs, crates/constraint-framework/src/component.rs, crates/examples/src/poseidon/zk.rs
  Confidence: 88%
  Reason: the paper supports statistical ZK for combined lookup/permutation/fractional arguments but does not give a complete STWO LogUp protocol; we need explicit lemmas and verifier gates before calling this sound.
```

Human approval is required before implementation is called production-sound.

## Core protocol design

### Problem

Current STWO LogUp exposes per-component `claimed_sum` values. For private LogUp relations, this is a challenge-dependent fingerprint of private witness and interaction data.

That is not zero knowledge.

### Required replacement

Private LogUp components must expose only a public aggregate target:

- `0` for a signed send/receive relation where all private lookup mass cancels.
- A public boundary expression derived only from public statement data.

The verifier must never receive raw private per-component `claimed_sum` values.

### Statistical aggregate masking

For each private LogUp aggregate group:

- Let each private component have an actual LogUp boundary claim `s_i`.
- Let the public target be `T`, where `sum(s_i) = T`.
- Prover samples additive masks `b_i` with `sum(b_i) = 0`.
- Prover exposes only masked shares `m_i = s_i + b_i`, or an equivalent proof-format representation that reveals no raw `s_i`.
- Private correction columns connect the masked share to the real LogUp running-sum boundary.
- Aggregate constraints enforce `sum(m_i) = T`.

Random masks alone are not sufficient. The correction/accumulator columns must be committed, randomized, and constrained inside the AIR/quotient path.

## Required lemmas before production signoff

- `Aggregate hiding`: masked per-component LogUp shares are statistically independent of private component claims given only the public aggregate target.
- `Correction soundness`: private correction columns enforce that every masked share corresponds to the actual LogUp running-sum boundary plus mask.
- `Aggregate lookup soundness`: if all local LogUp constraints hold and the aggregate target is public-zero/public-derived, the private send/receive multisets are consistent except with bounded lookup challenge failure probability.
- `Bad challenge bound`: denominator-zero, tuple-collision, and rational-identity failure probability is bounded by metadata-derived term count over the extension field.
- `ZK composition`: original trace, interaction trace, correction columns, accumulator columns, quotient splits, and FRI openings are all covered by paper-style polynomial randomization.
- `Transcript binding`: privacy metadata, aggregate policy, relation IDs, column scopes, degree bounds, and public aggregate targets are bound before lookup challenges.

## General AIR privacy provider

Each AIR declares privacy intent. It must not implement masking manually.

Required shape:

```rust
trait AirPrivacyProvider {
    fn privacy_map(&self, layout: &TraceLayout) -> PrivacyMap;
    fn logup_privacy(&self, layout: &TraceLayout) -> Vec<LogupPrivacyGroup>;
    fn public_statement(&self) -> PublicStatementSpec;
}
```

Provider responsibilities:

- Mark private original trace columns.
- Mark private interaction trace columns.
- Mark private correction and accumulator columns.
- Mark public preprocessed columns.
- Mark public I/O and public table material.
- Declare LogUp aggregate groups.
- Declare public aggregate targets.
- Declare allowed shape leakage.

The provider may know AIR semantics, but sizes, ranges, and bounds must be derived from actual STWO layout/component metadata whenever possible.

## Canonical metadata builder

There must be one central builder for ZK metadata.

Required output:

```rust
struct ZkAirMetadata {
    trace_domains: Vec<DomainMetadata>,
    column_scopes: Vec<ColumnScope>,
    privacy_map_hash: Digest,
    logup_groups: Vec<CanonicalLogupGroup>,
    degree_bounds: ZkDegreeBounds,
    quotient_mask_profile: QuotientMaskProfile,
    fri_binding: FriBindingMetadata,
    statistical_security: StatisticalSecurityBudget,
}
```

The builder must bind:

- `tree 0`: preprocessed columns.
- `tree 1`: original trace columns.
- `tree 2`: interaction trace columns.
- `tree 3`: composition/split columns.
- Private ranges.
- Public ranges.
- Column degree bounds.
- Trace domain.
- Randomized witness domain.
- Quotient split mask profile.
- FRI first layer.
- Lookup challenge timing.
- Relation IDs.
- Aggregate group membership and ordering.
- Public aggregate target.

No Poseidon-specific, StateMachine-specific, or test-specific constants may be used to make examples pass.

## Dynamic degree and leakage allocator

All ZK bounds must be formula-driven.

Required input:

```rust
struct ZkBoundInput {
    trace_log_degree: u32,
    randomized_private_column_log_degree: u32,
    air_constraint_degree: u32,
    logup_term_count_bound: usize,
    composition_split_count: usize,
    fri_blowup_log: u32,
    lookup_challenge_count: usize,
    expected_proof_volume: usize,
    min_statistical_security_bits: u32,
}
```

Required output:

```rust
struct ZkBoundOutput {
    randomized_trace_log_degree: u32,
    quotient_degree_bound: u32,
    composition_split_log_degree: u32,
    fri_first_layer_log_degree: u32,
    statistical_security_bits: u32,
}
```

Statistical budget rule:

```text
security_bits =
  extension_field_bits
  - ceil_log2(private_lookup_terms * lookup_challenges * expected_proof_volume)
  - safety_margin_bits
```

For QM31, use roughly `124` extension-field bits unless the implementation proves a tighter effective challenge space.

Reject if:

- Computed budget is below policy.
- Any multiplication overflows.
- Term counts are missing or manually under-declared.
- Degree requirements exceed configured FRI/proof bounds.

## Transcript binding

The verifier must bind ZK metadata before every affected challenge:

- Commitments.
- Privacy map hash.
- LogUp aggregate policy.
- Lookup challenge.
- Composition challenge.
- OODS point.
- Sampled values.
- FRI commitments.
- FRI query positions.

For LogUp/Poseidon, binding before lookup challenge is mandatory because lookup challenges define both the soundness and leakage surface.

## Verifier API

ZK verification must use a dedicated fail-closed API:

```rust
fn verify_zk_stark(
    public_statement: PublicStatementSpec,
    zk_metadata: ZkAirMetadata,
    proof: ZkProof,
    verifier_config: ZkVerifierConfig,
) -> Result<(), VerificationError>;
```

Verifier must reject if:

- Metadata hash does not match transcript binding.
- Metadata is missing, malformed, stale, or bound too late.
- Private LogUp appears without an aggregate group.
- Any private per-component `claimed_sum` is public.
- Aggregate target is not public-zero or public-derived.
- Private interaction/correction/accumulator columns are not randomized.
- Masked shares do not sum to the public aggregate target.
- Computed statistical security is below threshold.
- Degree allocator says larger bounds are required than the proof supplies.
- Public proof surfaces contain raw private values.

## Proof-visible privacy surfaces

No raw private value may appear in:

- `sampled_values`
- `queried_values`
- FRI answers
- OODS openings
- Public metadata
- Commitment auxiliary/debug surfaces
- Benchmark reports
- Serialized proof/debug output

This must be enforced by tests using concrete private witness values.

## Implementation sequence

1. Add statistical-ZK LogUp plan gates and proof obligations to the existing ZK plan docs.
2. Add provider declarations for private LogUp aggregate groups and public aggregate targets.
3. Add canonical metadata builder support for aggregate groups and challenge timing.
4. Add dynamic degree/leakage allocator.
5. Add transcript binding before lookup challenge derivation.
6. Add private LogUp accumulator/correction columns.
7. Add aggregate constraints enforcing public target consistency.
8. Add masked sampled/opening path for private interaction, accumulator, correction, quotient, and FRI values.
9. Add fail-closed verifier API and relation checks.
10. Add positive tests for at least one small LogUp AIR, then Poseidon.
11. Add adversarial negative tests.
12. Add benchmark/report hooks.
13. Run staff rereview and human approval before production-sound signoff.

## Example rollout

### Pilot 1: minimal synthetic LogUp AIR

Purpose:

- Validate aggregate masking independent of Poseidon complexity.
- Exercise private interaction trace, correction columns, aggregate target, verifier checks, and leakage tests.

Required result:

- Two same-public-statement proofs with different private witnesses verify.
- Raw private trace/interactions/claims do not appear on proof-visible surfaces.
- Tampering after transcript fixation rejects.

### Pilot 2: StateMachine

Purpose:

- Validate signed send/receive aggregation on an existing STWO LogUp example.
- Replace per-component claim exposure with public aggregate target checks.

Required result:

- Public STWO path unchanged.
- ZK path accepts only provider-declared public aggregate targets.
- Private per-component claims remain hidden.

### Pilot 3: Poseidon

Purpose:

- Validate high-degree private LogUp path used by Infinite.
- Validate dynamic degree allocation and quotient masking under `LOG_EXPAND = 2` or higher.

Required result:

- No hardcoded Poseidon-specific ZK degree budget.
- Public Poseidon still passes original STWO tests.
- Private Poseidon ZK proof verifies end-to-end.
- Negative tests reject wrong privacy map, wrong aggregate target, tampered masked opening, wrong quotient mask profile, insufficient leakage budget, and replay under different metadata.

## Benchmark requirements

Benchmark report must include:

- Public Poseidon prove.
- ZK witness randomization.
- ZK LogUp accumulator/correction generation.
- ZK quotient/composition masking.
- ZK FRI/proof generation.
- Public verify.
- ZK verify.
- Proof size delta.
- Opened value count delta.
- Metadata size delta.

Benchmarks are not a soundness gate, but they are required before calling the integration production-ready.

## Staff review result

### Math/Crypto

Conditional design signoff.

Option 1 is the right direction and paper-compatible as a statistical-ZK extension. Implementation soundness is not signed off until the required lemmas, aggregate constraints, transcript gates, and leakage-budget checks are implemented and tested.

### Security/Pentest

Conditional design signoff.

Statistical ZK is acceptable if public proof surfaces do not leak raw private data, metadata is bound before challenges, tampering/replay tests exist, and the leakage budget fails closed.

### Rust/API/Performance

Conditional design signoff.

Implementation is feasible if the provider-driven metadata builder is canonical, bounds are formula-driven, the verifier fails closed, and benchmark hooks are added.

## Current decision

Proceed with option 1.

Do not claim final soundness yet. The next implementation increment should be the provider-driven aggregate metadata and dynamic leakage/bound allocator, because those are prerequisites for a sound prover/verifier path.
