# Infinite ZK STARK Phase 2 readiness pack

Status: `REVIEW_READY`

This is the review artifact for starting, not activating, Phase 2 witness
randomization. Code remains fail-closed until Math, Crypto, Security, Rust, and
Performance reviewers sign off. The activation guard
`Phase2And3ActivationBlocked` must stay in place until the semantic Phase 2/3
implementation and its evidence pass review.

## Paper anchor

The paper randomizes each private witness polynomial as:

```text
w_hat_i(X) = w_i(X) + v_H(X) * r_i(X)
```

The randomizer `r_i` is prover-private. The verifier must see only commitments
and ordinary openings of randomized oracles. The verifier must never receive
randomizer coefficients, RNG seeds, additive masks, or enough mask material to
recover the original witness value.

The paper's Protocol 3 bound counts base-field linear constraints on the
randomizer after expanding extension-field OODS queries, translated transition
queries, quotient-component implicit queries, and FRI queries. STWO must not
copy the paper's symbolic `d`; it must instantiate the same rank/counting
argument from STWO's actual query closure.

## Phase 2 scope restriction

Phase 2 covers ordinary committed witness columns only.

Private columns are ineligible if they participate in private lookup,
permutation, fractional, LogUp, memory, grand-product, or other multiset
arguments whose success/failure or denominator behavior can leak private
values. This follows the paper's Appendix A warning that polynomial
randomization alone does not complete zero knowledge for permutation-style
arguments.

The verifier-owned `private_column_scope_hash` binds this eligibility decision.
Activation must reject if:

- the hash is missing or zero;
- the proof echoes a different scope hash than verifier-owned metadata;
- a private column is used by a lookup/permutation/fractional argument;
- the application cannot produce canonical column ownership metadata.

Appendix A support is a separate future gate and is not part of Phase 2.

## Gate 1: STWO query closure

Status: `REVIEW_READY`

For each private column range, the verifier constructs a canonical set of
base-field linear functionals. The prover cannot define this set.

Inputs:

- verifier-owned privacy map and private column ownership;
- verifier-owned sampled-point metadata for OODS and translated constraints;
- verifier-owned trace/commitment domain metadata;
- public FRI query positions sampled from the Fiat-Shamir channel;
- future quotient-component preimage metadata, if STWO ever exposes explicit
  quotient-component openings.

Canonical algorithm:

1. Reject the private column if the Phase 2 scope restriction marks it
   ineligible.
2. For every extension-field sampled point involving the private column, add
   four base-field coordinate functionals because `QM31 / M31` has degree four.
3. For every translated base-domain sampled point involving the private column,
   add one base-field functional.
4. For every FRI query opening of the private column, add one base-field
   functional at the queried commitment-domain position.
5. For every future explicit quotient-component opening involving the private
   column, add the corresponding preimage functionals before deduplication.
6. Deduplicate by canonical tuple:

```text
(column_range, query_kind, domain_id, point_or_position_encoding, coordinate_index)
```

7. Let `q_i` be the number of remaining functionals for private column `i`.

For the current STWO PCS shape, `future_implicit_component_points_i = 0`
because `split_at_mid` quotient pieces are not verifier-opened as independent
witness-column oracles. If that proof format changes, this gate reopens.

Activation must compute `q_i` from actual STWO metadata and reject if any
configured witness-randomizer dimension is smaller than `q_i`.

## Gate 2: circle randomizer space

Status: `REVIEW_READY`

For each private base-field column `w_i` over trace domain `H`, the prover
samples a base-field circle polynomial `r_i` from the verifier-approved STWO
circle basis.

The randomized committed column is:

```text
w_hat_i(P) = w_i(P) + v_H(P) * r_i(P)
```

where `v_H` is STWO's circle-domain vanishing polynomial for the original trace
domain. Since `v_H(P) = 0` for `P in H`, trace constraints over `H` are
unchanged. Since Phase 2 rejects query points in `H`, multiplication by
`v_H(P)` is invertible on every accepted query point.

The implementation must form the public evaluation matrix for the private
column's query-closure functionals against the selected randomizer basis and
reject unless its row rank is exactly `q_i`. The inequality `h_i >= q_i` is
necessary but not sufficient; the rank check is the activation condition that
binds the STWO circle basis to the paper's linear-independence proof.

Randomizer coefficients must be sampled uniformly from a prover-private
`CryptoRng` using rejection sampling. Coefficients, RNG seeds, derived masks,
and debug representations of randomness must not enter proofs, aux data, logs,
or verifier APIs.

## Gate 3: OODS and domain exclusion

Status: `REVIEW_READY`

Phase 2 must use deterministic public rejection sampling for OODS points.

The exclusion set must reject:

- the trace domain `H`;
- every committed evaluation domain `D`;
- translated or shifted query domains used by the AIR;
- denominator-degenerate line cases;
- any future implicit quotient-component preimage set.

Prover and verifier run the same rejection loop from the same Fiat-Shamir
state. Rejected candidates are consumed identically by both sides. A bounded
retry limit fails closed.

## Gate 4: Fiat-Shamir schedule

Status: `REVIEW_READY`

Phase 2 must use this ordering:

1. Mix protocol version, PCS config, public statement hash, privacy-map hash,
   private-column scope hash, degree profile, and column degree metadata.
2. Commit randomized private witness columns.
3. Draw composition challenges.
4. Commit randomized composition quotient/split oracles.
5. Draw the OODS point with the public rejection loop.
6. Mix OODS sampled values of randomized oracles.
7. Commit the independent FRI batch-mask oracle `R`.
8. Mix the `R` commitment and domain metadata.
9. Draw the FRI quotient-batching challenge.
10. Commit FRI layers for `H_batch = raw_quotient + R`.
11. Mix the last FRI layer and proof-of-work state.
12. Draw FRI query positions.
13. Open randomized trace/quotient commitments and `R` at query positions.
14. Verify FRI first-layer answers with `R(query)` added to raw quotient
    answers.

Any implementation that draws batching randomness before binding `R`, or sends
both original and randomized private witness values, is out of scope.

## Gate 5: ZK-aware degree metadata

Status: `REVIEW_READY`

The verifier owns all degree metadata used to activate Phase 2. Proof metadata
may echo it but cannot define it.

For each private column:

```text
randomized_log_degree_i >= max(original_log_degree_i, log2_ceil(|H| + h_i))
```

PCS commitment domains, sampled-point metadata, quotient bounds, and verifier
FRI bounds must be derived from verifier-owned metadata.

Activation must reject if:

- proof metadata differs from verifier metadata;
- privacy-map hashes do not match canonical column metadata;
- private-column scope hash is missing or mismatched;
- private-column degree bounds are missing;
- randomized degree bounds are smaller than original bounds;
- any query closure cardinality exceeds the configured `h_i`;
- the randomizer evaluation matrix rank is below `q_i`.

## Gate 6: FRI batch-mask degree

Status: `REVIEW_READY`

Protocol 2 `R` remains mandatory after witness randomization. The Phase 1 `R`
oracle samples prover-private coefficients, commits before the batching
challenge, and authenticates `R(query)` values.

For Phase 2, `h_batch` must be derived from the first-layer FRI domain and the
reviewed quotient degree bound after witness randomization. The verifier must
reject any proof where the `R` domain, first FRI layer, commitment, query
positions, query values, or metadata disagree.

## Tests required before activation

- Phase 2 config remains fail-closed without all review hashes.
- Every public Phase 2 validation API remains fail-closed before activation.
- Phase 2 config remains fail-closed with inconsistent privacy-map hash.
- Phase 2 config remains fail-closed with missing private-column scope hash.
- Phase 2 config remains fail-closed with insufficient `h_i`.
- Phase 2 rejects private lookup/permutation/fractional columns.
- Phase 2 OODS sampling rejects points in every forbidden domain.
- Phase 1 verifier rejects unsupported versions and Phase 2 metadata.
- Verifier rejects proof metadata that differs from verifier-owned metadata.
- Tampered `R` log size, query count, query position, commitment, and queried
  value all fail verification.

## Performance evidence required before activation

- Three matched baseline/current runs.
- Default-path PCS/FRI regression closure.
- Small/medium/large end-to-end fixtures.
- `0%`, `25%`, `50%`, and `100%` private-column ratios.
- Proof-byte attribution for `R`, randomized commitments, decommitments, and
  query values.
- Peak-memory or allocation proxy for randomizer generation, `v_H * r_i`,
  commitment evaluation, tree construction, quotient generation, and FRI.

## Activation rule

Only after the staff team signs this pack may code remove
`Phase2And3ActivationBlocked` and activate semantic witness randomization.
