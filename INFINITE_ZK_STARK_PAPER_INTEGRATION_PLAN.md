# Infinite STWO Paper-Level ZK STARK Integration Plan

Branch: `infinite/zk-stark-paper-integration-v1`

Source baseline: local `dev`

Primary paper: `/Users/ehjc/workspace/projects/firecrawl-py/zk-stark/scraped_document.md`

Objective: integrate the Habock-Kindi STARK zero-knowledge construction into
STWO in a paper-grounded way, with prover-private polynomial randomization and
FRI batch masking. This plan intentionally excludes additive sampled-value
mask/unmask designs.

## 1. Non-negotiable boundaries

- Do not send a value together with the mask/randomizer that recovers it.
- Do not derive production privacy from public transcript data alone.
- Do not implement verifier-side unmasking for public ZK verification.
- Do not expose witness randomizer coefficients, mask polynomial coefficients,
  random seeds, or per-value masks in proof data, aux data, logs, debug output,
  or verifier APIs.
- Do not modify the existing `prove` / `verify` behavior as the first
  production path. Original STWO proofs and vectors must remain runnable through
  the default APIs.
- Add an explicit ZK API/proof type first, then decide later whether it should
  become the default for Infinite.
- Any proof-system semantic change requires Math Reviewer signoff before code
  lands and another Math Reviewer pass after the implementation.

## 2. Paper requirements mapped to STWO

### 2.1 Protocol 1: randomized witness oracles

Paper requirement:

```text
w_hat_i(X) = w_i(X) + v_H(X) * r_i(X)
r_i(X) sampled uniformly with deg(r_i) < h
```

STWO mapping:

- `H`: the trace domain for a committed trace column.
- `v_H`: STWO circle-domain vanishing polynomial for that trace domain, using
  `coset_vanishing` or a reviewed circle-domain equivalent.
- `w_i`: private witness trace column polynomial represented as
  `CircleCoefficients<B>`.
- `w_hat_i`: committed replacement polynomial for private witness columns.
- Public columns, preprocessed columns, and padding/root-only columns are not
  randomized unless a role-specific privacy map marks them private and the
  Math Reviewer signs off.

Required STWO work:

- Define a `ZkPrivacyMap` that identifies private witness columns before trace
  commitment.
- `ZkPrivacyMap` is verifier-owned configuration, not proof-trusted metadata.
  The proof may echo a privacy-map commitment/hash, but `verify_zk_ex` must
  compare it against `ZkVerificationConfig` and transcript-bind it before any
  affected Fiat-Shamir challenge.
- Add a prover-only randomizer generator that samples uniform base-field
  coefficients for `r_i`.
- Add a reviewed coefficient-level construction of the STWO circle equivalent
  of `v_H * r_i` and add it to the original column polynomial before
  `CommitmentTreeProver::new`.
- Ensure `w_hat_i(p) == w_i(p)` for every point `p in H`.
- Update degree accounting for private columns from `< |H|` to `< |H| + h`.

Primary insertion points:

- `crates/stwo/src/prover/pcs/mod.rs`: before `CommitmentTreeProver::new`
  evaluates and commits trace polynomials.
- `crates/stwo/src/prover/mod.rs`: composition polynomial generation must use
  the randomized trace already present in `commitment_scheme.trace()`.
- `crates/stwo/src/core/verifier.rs`: verifier column log-size and degree-bound
  expectations must match the randomized committed degree.

### 2.2 Protocol 3: randomized AIR / DEEP-ALI

Paper requirement:

```text
2 * d * (e * n_F + n_D) + n_D <= h <= |H|
```

where:

- `d`: constraint composition quotient split factor.
- `e`: extension-field degree.
- `n_F`: number of OODS/DEEP query points, not counting translated points.
- `n_D`: number of FRI query rounds.
- `h_witness`: witness-randomizer degree of freedom.

STWO-specific blocker:

- STWO uses circle polynomials and an FFT-basis split. The paper's Protocol 3
  presents a univariate `q_j(X^d)` split. We must write and review a
  STWO-specific derivation before implementing witness randomization.
- Do not instantiate `2 * d * (e * n_F + n_D) + n_D <= h` with
  `COMPOSITION_LOG_SPLIT`, `2`, or `2^COMPOSITION_LOG_SPLIT`. The paper's `d`
  is not currently mapped to STWO.

Reviewed blocker resolution:

- Do not satisfy quotient/split coverage by emitting placeholder
  `FutureQuotientComponent` witness-randomizer rows. STWO's `split_at_mid`
  openings are not currently proven equivalent to the paper's univariate
  FFT-fiber openings, so counting them as ordinary witness point evaluations
  can undercount leakage.
- Use independent quotient-split masking for the STWO split identity instead.
  For the committed composition split
  `p(z) = left(z) + Pi_L(z.x) * right(z)`, sample a prover-secret split mask
  `t` and commit/open only
  `left_hat = left + Pi_L * t` and `right_hat = right - t`.
- The verifier recombination is unchanged:
  `left_hat(z) + Pi_L(z.x) * right_hat(z) = left(z) + Pi_L(z.x) * right(z)`.
  The split mask hides the additional decomposition leakage from observing
  `left` and `right` separately. Privacy of the recombined value remains the
  responsibility of the witness-randomization rank audit, because the
  recombined value is deterministic post-processing of randomized witness
  openings and public challenges.
- `Pi_L` must be derived from the original STWO split identity bound, not from
  any inflated committed degree bound used for masked split columns.
- Raw `left`, raw `right`, and `t` must never be serialized, logged, opened, or
  passed to the verifier. Private STARK activation remains fail-closed until
  this independent split-mask path is implemented and reviewed end-to-end.

Required derivation deliverable:

- Document STWO's exact trace domain `H`, commitment domain `D`, translated
  query set, and composition split semantics.
- Derive the STWO query-expansion factor, if any, for `split_at_mid`, whose
  identity is `p(z) = p_left(z) + pi^{L-2}(z.x) * p_right(z)`.
- Prove or conservatively bound how many independent randomizer degrees are
  consumed by:
  - OODS trace queries.
  - translated/shifted trace queries.
  - composition quotient component queries.
- FRI query openings.
  - FRI last-layer publication.
- Define the initial accepted `h_witness` strategy after the STWO query-expansion
  review.
- Define a separate quotient-split mask profile:
  - split identity bound used to compute `Pi_L`;
  - masked left/right committed degree bounds;
  - split-mask entropy budget `h_split`;
  - commitment-tree position and side semantics;
  - transcript binding before OODS, FRI queries, and batch-FRI lambda.
- Prove that leaking the recombined composition value is safe under the
  witness-randomization rank audit, and that the independent split mask covers
  the extra information from opening split components separately.
  derivation is reviewed. The recommended first implementation candidate is
  conservative, `h = |H|`, but it is not accepted until the STWO-specific bound
  proves it sufficient.
- Specify the exact circle-polynomial randomizer space, its dimension `h`, its
  base-field sampling basis, and the coefficient-level construction of
  `v_H * r_i`.
- Prove that evaluations on the STWO OODS/FRI query closure are independent of
  the original private witness values under that randomizer space.

Primary insertion points:

- `crates/stwo/src/prover/mod.rs`: composition generation, split commitment,
  OODS sampling, and sampled-values production.
- `crates/stwo/src/core/verifier.rs`: composition split degree expectations,
  OODS consistency checks, and verifier-side sampled point construction.
- `crates/stwo/src/prover/poly/circle/ops.rs`: current `split_at_mid` basis
  semantics used by STWO.
- `crates/constraint-framework/src/component.rs`: ZK-aware
  `Component::trace_log_degree_bounds` / framework trace metadata.
- `crates/stwo/src/prover/air/accumulation.rs`: ZK-aware
  `EvaluationMode::infer` and composition accumulator sizing.

### 2.3 Protocol 2: FRI batch mask polynomial `R`

Paper requirement:

```text
R(X) sampled independently over F before batching challenge lambda
H_batch(X) = R(X) + batched DEEP quotient expression
FRI proves proximity of H_batch or its reviewed decomposition
```

STWO mapping:

- Current STWO computes batched OODS quotients in
  `compute_fri_quotients(...)`.
- Current STWO draws the batching challenge immediately before quotient
  computation in `CommitmentSchemeProver::prove_values`.
- Current STWO runs FRI on the unmasked quotient evaluation.
- Paper-level ZK requires a separate committed oracle for `R` before the
  batching challenge, and FRI must run on `h = quotient + R`.
- In STWO terms, `R` is sampled uniformly over the extension field
  `SecureField`, not only the base field. Its commitment is encoded as four
  canonical base-field columns for Merkle authentication.
- Use distinct degree symbols:
  - `h_witness`: degree of witness randomizers `r_i`.
  - `h_batch`: Protocol 2 FRI batch-mask degree budget. In the paper's
    Protocol 3 handoff this is `ceil((d + 1) / d * h_witness)`; in STWO it must
    be replaced by the reviewed STWO equivalent after the circle split
    derivation.
  - `H_batch(X)`: the FRI input polynomial `raw_quotient(X) + R(X)`.

Required STWO work:

- Add a prover-only `FriBatchMaskPolynomial` sampled uniformly in
  `SecureField[X]` and independently from witness randomizers.
- Commit an oracle for `R` after OODS sampled values are mixed into the channel
  and before the PCS quotient batching challenge is drawn.
- Draw the batching challenge after the `R` commitment is mixed.
- Compute the existing batched quotient.
- Add `R` on the FRI evaluation domain to produce `H_batch`.
- Run FRI on `H_batch`, not on the raw quotient.
- During FRI queries, open `R` at the same query positions using the pre-lambda
  `R` commitment.
- Verifier computes `fri_answers(...) + R(query)` and passes that to
  `FriVerifier::decommit`.
- Verifier rejects if the `R` openings do not authenticate against the
  pre-lambda `R` commitment.
- `R` is a separate oracle, not a normal `CommitmentSchemeVerifier.trees` entry.
  Adding it to the normal PCS tree list would corrupt raw quotient batching.
- The exact `R` degree bound and commitment domain must follow the reviewed
  STWO equivalent of the paper's `h_batch` and the reviewed STWO FRI
  first-layer bound. Until the STWO degree derivation is signed off, Phase 1 may
  only test the separate-oracle mechanics and must not claim full paper-level
  witness privacy.
- `R` must be excluded from raw quotient batching and included only through
  `H_batch = raw_quotient + R`.

Primary insertion points:

- `crates/stwo/src/prover/pcs/mod.rs`: between `channel.mix_felts(sampled_values)`
  and `channel.draw_secure_felt()` for quotient batching.
- `crates/stwo/src/core/pcs/verifier.rs`: mirror the same order after
  `channel.mix_felts(proof.sampled_values)` and before drawing the batching
  challenge.
- `crates/stwo/src/prover/pcs/quotient_ops.rs`: keep raw quotient computation
  intact, then add `R` in a reviewed wrapper.
- `crates/stwo/src/core/pcs/quotients.rs`: compute FRI first-layer query
  answers as `raw_quotient_answer + R_query_value`.
- `crates/stwo/src/prover/fri.rs` and `crates/stwo/src/core/fri.rs`: no folding
  semantic changes unless the STWO derivation requires a reviewed decomposition
  change.

## 3. Proposed API shape

Keep default STWO APIs unchanged:

```text
prove(...)
prove_ex(...)
verify(...)
verify_ex(...)
CommitmentSchemeProof
StarkProof
```

Add explicit ZK APIs:

```text
prove_zk(...)
prove_zk_ex(...)
verify_zk(...)
verify_zk_ex(...)
prove_values_zk(...)
verify_values_zk(...)
ZkStarkProof
ZkCommitmentSchemeProof
ZkFriBatchMaskProof
ZkFriBatchMaskQueryValues
ZkProvingConfig
ZkVerificationConfig
ZkPrivacyMap
```

Rules:

- `prove_values_zk` and `verify_values_zk` must be separate functions, not
  branches inside existing `prove_values` / `verify_values`.
- No feature flag may alter default `prove_values` / `verify_values`
  transcript order or proof bytes.
- `R` is committed and verified by a dedicated `ZkFriBatchMaskProof`, with its
  own `MerkleVerifierLifted`, decommitment, log size, and canonical query-value
  encoding.
- `ZkFriBatchMaskQueryValues` must define the exact public encoding of
  `R(query)` values using existing lifted Merkle verifier base-field query
  value representation.
- Canonical `R(query)` encoding:
  - `R` is sampled as an extension-field polynomial over `SecureField` and
    committed as four base-field columns in one dedicated lifted Merkle oracle,
    representing the canonical `QM31` coordinate order returned by
    `SecureField::to_m31_array()`.
  - `ZkFriBatchMaskQueryValues` stores `queries: Vec<[BaseField; 4]>` in the
    exact order of the deduplicated FRI `query_positions` returned by
    `FriVerifier::sample_query_positions` / `FriProver::decommit`.
  - Serialization order is outer query order, then coordinate index `0..4`,
    each `BaseField` encoded by the existing serde representation for `M31`.
  - Conversion to `SecureField` for `fri_answers + R(query)` uses
    `SecureField::from_m31_array([c0, c1, c2, c3])`.
  - The Merkle verifier receives the same values as lifted base-field query
    columns: four columns, each with one value per deduplicated FRI query
    position, in the same query order.
  - The proof does not store unsorted `R` query values separately. If aux data
    needs unsorted locations for diagnostics, it must be non-public aux and must
    not affect transcript or serde proof bytes.
- `ZkStarkProof` may contain randomized commitments, randomized sampled values,
  authenticated `R` commitment/openings, and FRI proof data for `H_batch`.
- `ZkStarkProof` must not contain original private witness openings.
- `ZkStarkProof` must not contain `r_i`, `R` coefficients, seeds, additive
  masks, or verifier-recoverable mask material.
- `ZkCommitmentSchemeProof` should wrap or mirror `CommitmentSchemeProof` only
  where fields remain semantically correct for the randomized proof.
- If `ZkCommitmentSchemeProof` embeds a `CommitmentSchemeProof`, that embedded
  proof is semantically incomplete without `ZkFriBatchMaskProof`. Do not
  implement `Deref<Target = CommitmentSchemeProof>`, `From<ZkStarkProof> for
  StarkProof`, or any conversion that lets callers accidentally use the normal
  verifier.
- Original proof structs should not gain optional ZK fields in the first
  implementation pass. A separate type avoids silently changing serialization
  and vector hashes.
- `ZkVerificationConfig` is verifier-owned and contains the expected version,
  degree profile, privacy-map commitment/hash, public AIR/domain identifiers,
  and accepted randomizer strategy. Proof metadata may echo these fields, but
  verification trusts only the verifier config.
- `ZkPrivacyMap` is owned, deterministic, and index-based. It may use stable
  tree/column identifiers or `TreeSubspan`-like ranges. It must not borrow
  component provers, trace columns, or temporary randomized polynomials.
- `prove_zk` / `prove_zk_ex` keep STWO's generic shape:
  `B: BackendForChannel<MC>, MC: MerkleChannel`. No CPU-only randomizer path and
  no backend-specific serialized proof types.

Minimal signed-off API shape target:

```rust
pub struct ZkStarkProof<H: MerkleHasherLifted>(pub ZkCommitmentSchemeProof<H>);

pub struct ZkCommitmentSchemeProof<H: MerkleHasherLifted> {
    pub version: ZkProofVersion,
    pub randomized_pcs_proof: CommitmentSchemeProof<H>,
    pub fri_batch_mask: ZkFriBatchMaskProof<H>,
    pub public_metadata: ZkPublicMetadata,
}

pub struct ZkFriBatchMaskProof<H: MerkleHasherLifted> {
    pub commitment: H::Hash,
    pub log_size: u32,
    pub decommitment: MerkleDecommitmentLifted<H>,
    pub queried_values: ZkFriBatchMaskQueryValues,
}

pub struct ZkFriBatchMaskQueryValues {
    /// Deduplicated FRI query order; each entry is one QM31 value encoded as
    /// four canonical M31 coordinates.
    pub queries: Vec<[BaseField; 4]>,
}

pub fn prove_zk_ex<B, MC, R>(
    components: &[&dyn ComponentProver<B>],
    channel: &mut MC::C,
    commitment_scheme: CommitmentSchemeProver<'_, B, MC>,
    zk_config: &ZkProvingConfig,
    rng: &mut R,
    include_all_preprocessed_columns: bool,
) -> Result<ExtendedZkStarkProof<MC::H>, ProvingError>
where
    B: BackendForChannel<MC>,
    MC: MerkleChannel,
    R: RngCore + CryptoRng;

pub fn verify_zk_ex<MC: MerkleChannel>(
    components: &[&dyn Component],
    channel: &mut MC::C,
    commitment_scheme: &mut CommitmentSchemeVerifier<MC>,
    proof: ZkStarkProof<MC::H>,
    zk_config: &ZkVerificationConfig,
    include_all_preprocessed_columns: bool,
) -> Result<(), VerificationError>;
```

## 3.1 no_std, serialization, and memory boundaries

- `ZkStarkProof`, `ZkCommitmentSchemeProof`, `ZkFriBatchMaskProof`,
  `ZkFriBatchMaskQueryValues`, `ZkVerificationConfig`, and verifier logic live
  under `core` and use `std_shims` / `alloc` only.
- RNGs, secret randomizer material, zeroization, and `ZkProvingConfig`
  internals live under `prover`.
- Do not change field order, serde shape, or derive behavior for `StarkProof`,
  `CommitmentSchemeProof`, `FriProof`, `PcsConfig`, or default proof structs.
- ZK metadata serialization must be deterministic and ordered. No raw hash-map
  iteration order may affect proof bytes or transcript bytes.
- Secret seed/randomizer/`R` coefficient types must not implement `Serialize`,
  `Deserialize`, `Debug`, or `Clone` unless a staff review documents why that is
  safe.
- ZK witness commitments should use owned trees through the existing commitment
  scheme or provide an explicit buffer-return path. Do not construct borrowed
  temporary randomized trees that bypass `BaseColumnPool` return.
- Initial implementation allocation policy:
  - randomized witness commitments use owned trees through
    `CommitmentSchemeProver` and return base-column buffers through the existing
    `BaseColumnPool` path.
  - `R` evaluation uses normal secure-column allocation in the first
    implementation, not a new pool.
  - `H_batch = raw_quotient + R` is formed into a normal `SecureEvaluation`
    passed to `FriProver::commit`; ownership remains local to `prove_values_zk`
    until FRI decommitment completes.
  - No borrowed temporary `R` or `H_batch` trees may be inserted into
    `CommitmentSchemeProver.trees`.
  - Phase -1/1 benchmarks must record `R` and `H_batch` allocation
    counts/bytes; a dedicated secure-column pool is an optimization only after
    performance evidence, not part of the first soundness implementation.

## 4. Randomness and sampling

Production randomness:

- Use `rng: &mut impl RngCore + CryptoRng` for explicit ZK proving APIs, or add
  a prover-only OS-CSPRNG feature/dependency before offering convenience
  constructors.
- Do not use `SmallRng` or user-supplied deterministic seeds for production
  proving.
- Sample fresh proof randomness for each proof.
- Derive independent streams for witness randomizers and FRI `R`.
- Use explicit domain separation:
  - `stwo.zk.witness-randomizer.v1`
  - `stwo.zk.fri-batch-mask.v1`
  - `stwo.zk.test-deterministic-rng.v1`
- Use exact uniform field sampling, not modulo reduction.
- Test-only deterministic RNGs must be typed as test-only and unavailable from
  production constructors.

Secret handling:

- Randomizer coefficients and seeds are prover-only.
- No `Serialize`, `Deserialize`, `Debug`, or public field exposure for secret
  material.
- Drop/zeroize where practical, and never log secret material.
- Treat sampled private-column openings as sensitive in logs unless Math Review
  explicitly proves a specific log point is ZK-safe. No `trace!` / `debug!`
  dumps of proof internals in ZK prover paths.

## 5. Degree and domain plan

Initial conservative strategy:

- Do not choose `h_witness` or `h_batch` in code until the STWO
  query-expansion factor and degree
  derivation are signed off.
- Candidate first strategy after derivation: choose `h_witness = |H|` for private
  witness columns when the reviewed STWO bound permits it.
- Under candidate `h_witness = |H|`, treat randomized private witness degree as
  `< 2 * |H|`.
- Accept the initial performance cost to reduce soundness ambiguity.
- Optimize to smaller `h` only after a Math Reviewer approves the
  STWO-specific non-twoadic/circle decomposition.

Degree-bound updates:

- Private trace columns: degree bound increases by one log unit under candidate
  `h_witness = |H|`.
- Composition quotient degree: must be recomputed from randomized witness
  degree and STWO constraint degree.
- PCS verifier log sizes: must reflect randomized column bounds for ZK proofs.
- FRI bound: must be set for `H_batch = raw_quotient + R` under the reviewed
  `h_batch` degree profile.

ZK-aware metadata updates required:

- `Component::trace_log_degree_bounds` must expose the randomized degree profile
  for private columns in the ZK path.
- `FrameworkComponent` and `FrameworkEval::max_constraint_log_degree_bound` must
  account for randomized witness degrees.
- `EvaluationMode::infer` must not reuse committed evaluations under stale
  non-ZK trace bounds.
- Composition accumulator sizing and split commitments must use the ZK degree
  profile.
- PCS column log sizes and verifier commitment log sizes must match the
  randomized commitments.

OODS and domain exclusion requirements:

- ZK prover and verifier must deterministically reject or resample OODS points
  that collide with any relevant trace domain `H`, commitment domain `D`,
  translated/shifted query domain, or line-denominator degeneracy.
- The exclusion rule is part of the public verifier algorithm and transcript
  definition, not a prover convention.
- Resampling, if used, must be deterministic from the same Fiat-Shamir stream
  and must not introduce prover choice.
- Tests must cover collisions or mocked collision paths for OODS points,
  translated mask points, and line-construction degeneracy.

Open derivation items:

- Whether STWO can avoid increasing commitment domain for all private columns by
  committing `w` on `H` and `w_hat` on `D` through a specialized oracle layer.
- Whether current `CirclePolyDegreeBound` can represent the needed bound
  without over-approximating too much.
- Whether the paper's `h0, h1` decomposition maps cleanly to STWO's
  `split_at_mid` / circle-to-line FRI path.

These are not implementation details. They are soundness gates.

## 6. Proof-format plan

Add `ZkCommitmentSchemeProof` fields:

- `randomized_pcs_proof`: the randomized analogue of the existing PCS proof.
- `fri_batch_mask`: separate `ZkFriBatchMaskProof` containing:
  - Merkle root for `R`, mixed before quotient batching challenge.
  - `R` log size and reviewed degree/domain metadata.
  - Merkle decommitment for `R` at FRI query positions.
  - canonical `ZkFriBatchMaskQueryValues` authenticated as `R(query)` values.
- `zk_metadata`: public parameters only, such as version, randomizer degree
  strategy, and privacy-map commitment/hash.

Do not include:

- `mask`.
- `masked_value` plus `mask`.
- witness randomizer coefficients.
- FRI `R` coefficients.
- proof seed or RNG state.

Verifier responsibilities:

- Reconstruct the same public sample point structure.
- Compare proof metadata against verifier-owned `ZkVerificationConfig`.
- Transcript-bind public ZK parameters before affected challenges: version,
  degree strategy, privacy-map hash, AIR/domain identifiers, randomized
  commitment roots, `R` commitment, and domain/log-size metadata.
- Mix randomized sampled values into Fiat-Shamir.
- Mix `fri_batch_mask_commitment` before drawing PCS quotient batching
  challenge.
- Verify `R` openings at FRI query positions.
- Compute raw quotient answers from randomized committed oracle openings.
- Add authenticated `R(query)` values.
- Feed the sum into the existing FRI verifier.
- Reject if `R` is omitted, reordered, duplicated, opened at wrong query
  positions, or opened against any commitment other than the pre-lambda
  commitment.

## 6.1 Baseline and benchmark gate before Phase 1

Performance work starts before implementation, not after the ZK API lands.

Required pre-Phase-1 baseline:

- Capture original `dev` behavior on this branch before any ZK code.
- Capture CPU and SIMD backends where supported.
- Capture small, medium, and large traces.
- Capture at least these private-column ratios for ZK benchmark fixtures:
  0%, 25%, 50%, 100%.
- Record machine metadata: CPU model, core count, memory, OS, Rust toolchain,
  `Cargo.lock` hash, commit SHA, target directory policy, power/thermal policy,
  CPU governor if available, and `RAYON_NUM_THREADS`.
- Record exact commands, cargo features, compiler flags, benchmark corpus, and
  repeated-run statistics.
- Use repeated runs; treat deltas within 3% as noise unless repeated in the
  same direction across at least three runs.

Required phase-by-phase benchmark reruns:

- Re-run after Phase 1, Phase 2, Phase 3, and Phase 4.
- Report absolute time, relative delta, proof size delta, memory peak delta,
  allocation-count proxy where practical, and stage-level deltas.
- A regression beyond the threshold below requires explicit Performance
  Reviewer approval before continuing:
  - prover time: +20%
  - verifier time: +15%
  - proof bytes: +10% before Phase 4, then phase-specific approval
  - peak memory: +20%

Required non-overlapping stage metrics:

- witness randomizer generation.
- `v_H * r_i` construction/addition.
- enlarged private commitment evaluation, tree construction, and hash cost.
- composition constraint evaluation.
- composition quotient/split generation.
- OODS value production and randomized trace access.
- `R` sampling.
- `R` evaluation.
- `R` commitment.
- `R` prover opening/decommitment construction.
- verifier `R` opening authentication.
- verifier `R(query)` addition to raw answers.
- raw FRI quotient computation.
- `R` addition to the FRI first layer.
- FRI commit/fold/query/prove/decommit/verify.

Required per-column/aggregate randomization metrics:

- private column count.
- trace domain size `|H|`.
- chosen `h`.
- randomizer degree.
- RNG time.
- polynomial construction time.
- extra allocations.
- bytes touched.

Required commitment/FRI attribution:

- original vs randomized degree bound.
- commitment domain log size and domain size.
- Merkle leaf count.
- commitment bytes.
- proof bytes.
- decommitment size.
- FRI first-layer size.
- FRI query count.
- FRI verifier decommit time.
- whether observed FRI overhead comes from larger degree, added `R`, or extra
  openings.

## 7. Implementation phases

### Phase -1: Baseline and instrumentation

Deliverables:

- Original STWO baseline captured with the methodology in Section 6.1.
- Stage timer/counter framework added without changing default proof bytes.
- Benchmark fixtures for small/medium/large traces and private-column ratios.

Exit criteria:

- Performance Reviewer signs off the baseline artifact and metric boundaries.
- Original APIs and vectors remain byte-identical.

### Phase 0: Derivation and guardrails

Deliverables:

- This plan.
- A derivation note for STWO circle-domain witness randomization and quotient
  split bounds.
- Proof-boundary denylist: no secret/randomizer names or bytes appear in proof
  serialization.
- Compile-time separation between test deterministic RNG and production CSPRNG.
- Verifier-owned `ZkVerificationConfig` and deterministic `ZkPrivacyMap`
  serialization rules.
- OODS/domain exclusion algorithm and tests.

Exit criteria:

- Math Reviewer signs off on the STWO derivation.
- Security/Pentest signs off that proof data cannot recover private witness
  openings.
- Rust reviewer signs off on API split and no_std verifier impact.

### Phase 1: FRI `R` infrastructure without ZK claim

Deliverables:

- Internal prover representation for `R`.
- Public proof fields for `R` commitment and authenticated query openings.
- `prove_values_zk` / `verify_values_zk`; existing `prove_values` /
  `verify_values` remain unchanged.
- Separate `ZkFriBatchMaskProof` oracle; `R` is not inserted into normal PCS
  trees.
- Canonical `ZkFriBatchMaskQueryValues` encoding.
- Prover/verifier transcript ordering changed only in explicit ZK APIs.
- FRI first-layer answers verified as `raw_answer + R(query)`.

Exit criteria:

- Existing non-ZK tests/vectors remain on the original APIs.
- ZK API verifies on existing PCS/STARK smoke cases.
- Tampering `R` commitment/openings/query values fails.
- Staff review confirms this is still not claimed as complete witness privacy
  until Phase 2/3 land.

### Phase 2: Witness polynomial randomization

Deliverables:

- `ZkPrivacyMap` for private witness columns.
- Prover-only randomizer polynomial generation.
- Reviewed STWO `v_H * r_i` implementation.
- Randomized private witness commitments.
- Composition generated from randomized trace polynomials.

Exit criteria:

- On-domain equality tests: `w_hat == w` over `H`.
- Off-domain difference tests: repeated proofs of the same witness produce
  different private commitments and OODS sampled values.
- Public/preprocessed columns remain unchanged unless marked private.
- No secret material appears in proof serialization or debug output.

### Phase 3: Degree-bound and quotient integration

Deliverables:

- ZK degree profile in prover and verifier.
- Updated composition split degree handling.
- Updated PCS/Fri degree bounds for randomized witnesses and `R`.
- OODS rejection/exclusion checks for trace and commitment domains.
- Independent STWO quotient-split masking:
  - sample prover-secret split masks from a CSPRNG with domain separation;
  - commit/open only `left_hat = left + Pi_L * t` and
    `right_hat = right - t`;
  - bind split-mask metadata before OODS, FRI query, and batch-FRI challenges;
  - reject mixed masked/unmasked quotient split modes;
  - keep raw split components and mask coefficients out of proof, logs, audit
    data, and verifier APIs.

Exit criteria:

- Verifier rejects inconsistent randomized OODS values.
- Verifier rejects wrong degree profile / wrong privacy map hash.
- Math Reviewer signs off that STWO's circle split preserves the paper
  invariant or that independent quotient-split masking replaces the blocked
  FFT-fiber dependency.
- Security reviewer signs off that recombined composition values are protected
  by witness randomization and that split masks cover only the additional
  decomposition leakage.

### Phase 4: End-to-end ZK STARK API

Deliverables:

- `prove_zk_ex` and `verify_zk_ex`.
- `prove_zk` and `verify_zk` convenience wrappers.
- Example/fixture coverage using the same AIRs as original STWO tests.
- Infinite-facing API notes.

Exit criteria:

- Original APIs continue to pass original vectors.
- ZK APIs verify randomized proofs.
- Same witness + different proof randomness yields different private
  commitments/proof bytes.
- Public outputs remain stable.

### Phase 5: Performance baseline and optimization

Deliverables:

- ZK baseline after Phase 4.
- Stage-level attribution:
  - witness randomizer generation.
  - enlarged private commitments.
  - composition overhead.
  - `R` commitment/openings.
  - FRI overhead.
- Optimization backlog based on measured deltas only.

Exit criteria:

- Performance reviewer signs off measured deltas and optimization priorities.
- No optimization removes `R`, reuses randomness, weakens degree bounds, or
  changes transcript order without Math Reviewer signoff.

## 8. Tests and adversarial coverage

Required functional tests:

- Original `prove` / `verify` unchanged.
- ZK `prove_zk` / `verify_zk` succeeds for CPU backend.
- ZK `prove_zk` / `verify_zk` succeeds for SIMD backend where existing tests
  cover SIMD.
- Randomized witness equals original witness on trace domain.
- Randomized witness differs off-domain with overwhelming probability.
- FRI verifies `raw_quotient_answer + R(query)`.

Required negative tests:

- Tamper `fri_batch_mask_commitment`: reject.
- Tamper `fri_batch_mask_queried_values`: reject.
- Tamper `fri_batch_mask_decommitment`: reject.
- Move `R` commitment after batching challenge in a test harness: transcript
  mismatch / reject.
- Use wrong privacy map hash: reject.
- Use wrong randomized degree profile: reject.
- Reuse deterministic test RNG in production constructor: reject or impossible
  by type.
- Serialize proof and scan for known test randomizer seed/material: absent.
- Replay a valid ZK proof under mismatched public inputs: reject.
- Replay under mismatched AIR identifiers/config: reject.
- Replay under mismatched domain sizes: reject.
- Replay under mismatched verifier-owned privacy map: reject.
- Replay under mismatched verifier-owned degree profile: reject.

Required vector tests:

- Original proof bytes unchanged for original APIs.
- Original sampled values unchanged for original APIs.
- Original FRI query indices unchanged for original APIs.
- ZK proof bytes intentionally differ across repeated proofs.
- Masked quotient split commitments differ across repeated private ZK proofs
  while verifier recombination remains unchanged.
- Tampering either masked split side rejects unless all PCS openings,
  recombination checks, and degree bounds remain valid.
- Private STARK verifier rejects missing split-mask metadata, wrong `h_split`,
  wrong masked split degree bounds, and mixed masked/unmasked split modes.

## 9. Staff review matrix

Plan review status applies only to this plan, not to future code.

| Role | Status | Required signoff scope |
|---|---|---|
| Math Reviewer | PASS | Plan requires STWO-specific query-expansion, randomizer-space, degree-bound, OODS/domain-exclusion, and FRI `R` derivation gates before implementation |
| Crypto Specialist | PASS | Plan is paper-faithful for Protocols 1/2/3, requires extension-field-uniform `R`, separates `h_witness` / `h_batch` / `H_batch`, and forbids verifier-unmask designs |
| Security/Pentest | PASS | Plan keeps verifier-owned privacy/config, transcript-bound public ZK metadata, no verifier-recoverable masks, CSPRNG requirements, serialization/logging restrictions, and tamper/replay coverage |
| Staff Rust Engineer | PASS | Plan isolates ZK APIs/proof types, uses a separate `R` oracle proof, specifies canonical `R(query)` encoding, preserves no_std verifier boundaries and original serialization/API behavior |
| Performance Reviewer | PASS | Plan requires baseline before Phase 1, phase-by-phase reruns, CPU/SIMD matrix, trace-size/private-ratio matrix, non-overlapping stage metrics, and regression gates |

Implementation signoff must be collected separately for each code phase.

## 10. Current staff-level risks

These are expected plan risks, not excuses to implement an unsound shortcut.

- STWO's circle FFT-basis composition split is not identical to the paper's
  univariate split. This is the main math blocker before witness
  randomization code. The accepted implementation path is independent
  quotient-split masking for STWO's `left + Pi_L * right` identity, not fake
  witness-randomizer query rows.
- OODS sampling currently uses random circle points; the ZK path must document
  and enforce exclusion from trace and commitment domains.
- Choosing `h = |H|` is simpler but can increase private commitment degree and
  cost. Performance optimization comes after correctness.
- FRI `R` requires a new pre-batching commitment and query-opening path. Adding
  `R` only to the first FRI layer without authenticating `R` before lambda is
  not paper-equivalent.

## 11. Forbidden designs

- Additive sampled-value masking where verifier receives both masked value and
  mask.
- Shadow proof wrappers that carry original proof plus diagnostic masks.
- Transcript-derived public masks treated as zero-knowledge.
- Any benchmark or feature flag that bypasses randomized witness commitment,
  OODS checks, `R` commitment/openings, or FRI verification.
- Claiming Phase 1 as full ZK before witness randomization and degree review.
