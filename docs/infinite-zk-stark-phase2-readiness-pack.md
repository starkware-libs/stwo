# Infinite ZK STARK Phase 2 readiness pack

Status: `REVIEW_READY`

This document is the proposed review artifact for moving from Phase 1 FRI
batch masking to Phase 2 witness randomization. It does not enable Phase 2 by
itself. Code remains fail-closed until Math, Crypto, Security, Rust, and
Performance reviewers sign off.

## Paper anchor

The paper randomizes private witness polynomials as:

```text
w_hat_i(X) = w_i(X) + v_H(X) * r_i(X)
```

The randomizer `r_i` is prover-private and verifier-unrecoverable. The verifier
must see only commitments and ordinary openings of randomized oracles.

The paper's Protocol 3 bound counts the number of base-field linear constraints
on the randomizer after expanding extension-field OODS queries, transition
translates, quotient-component implicit queries, and FRI queries. STWO must not
copy the paper's symbolic `d` blindly; it must instantiate the same counting
argument from the actual STWO query closure.

## Gate 1: STWO query expansion

Status: `REVIEW_READY`

STWO's PCS path does not expose separate FFT quotient component oracles to the
verifier. The verifier checks raw quotient answers through committed trace
openings and sampled values, then FRI checks the first-layer quotient oracle.

For witness randomization, the randomizer must hide every private-column
evaluation that contributes to:

- OODS sampled points.
- transition or lookup translated points included in `sampled_points`.
- trace openings at FRI query positions.
- any future explicit quotient-component opening, if one is added later.

The proposed STWO replacement for the paper's `d` is therefore not a fixed
constant. It is a verifier-owned query-closure cardinality:

```text
q_i = |GaloisClosure(extension_points_i)|
    + |base_domain_points_i|
    + |future_implicit_component_points_i|
```

For the current PCS shape, `future_implicit_component_points_i = 0`.

Each `QM31` OODS point contributes at most four base-field constraints because
`QM31 / M31` has extension degree four. Each committed base-domain FRI query
position contributes one base-field constraint.

Phase 2 must compute `q_i` from actual STWO column metadata and sampled point
metadata, not from proof-supplied metadata.

## Gate 2: circle randomizer space

Status: `REVIEW_READY`

For each private column `w_i` over trace domain `H`, sample a base-field circle
polynomial `r_i` from a verifier-approved randomizer space of dimension
`h_i >= q_i`.

The randomized committed column is:

```text
w_hat_i(P) = w_i(P) + v_H(P) * r_i(P)
```

where `v_H` is the circle-domain vanishing polynomial for the original trace
domain. Since `v_H(P) = 0` for `P in H`, original constraints over `H` remain
unchanged. Since all Phase 2 query points must be outside `H`, multiplication by
`v_H(P)` is invertible on the query set, so the image of the randomizer
evaluation map has rank `q_i` when `h_i >= q_i`.

The implementation must sample randomizer coefficients from prover-private
`CryptoRng`; coefficients, RNG seeds, and derived masks must never enter the
proof.

## Gate 3: OODS and domain exclusion

Status: `REVIEW_READY`

Phase 2 must replace raw OODS sampling with deterministic rejection sampling
from verifier-owned exclusion sets.

The exclusion set must reject:

- the trace domain `H`;
- every committed evaluation domain `D`;
- transition/lookup translated domains used by the AIR;
- denominator-degenerate line cases;
- any future implicit quotient-component preimage set.

Prover and verifier must run the same public rejection loop from the same
Fiat-Shamir transcript state. A bounded retry limit must fail closed.

## Gate 4: ZK-aware degree metadata

Status: `REVIEW_READY`

The verifier must own all degree metadata used to activate Phase 2. Proof
metadata may echo it but cannot define it.

For each private column:

```text
randomized_log_degree_i >= max(original_log_degree_i, log2_ceil(|H| + h_i))
```

PCS commitment domains, sampled point metadata, quotient bounds, and verifier
FRI bounds must be derived from this verifier-owned metadata.

Phase 2 must reject if:

- proof metadata differs from verifier metadata;
- privacy-map hashes do not match canonical column metadata;
- private-column degree bounds are missing;
- randomized degree bounds are smaller than original bounds;
- any query closure cardinality exceeds the configured `h_i`.

## Gate 5: FRI batch-mask degree

Status: `REVIEW_READY`

Protocol 2 `R` remains mandatory after witness randomization. The Phase 1 `R`
oracle already samples prover-private coefficients, commits before the batching
challenge, and authenticates `R(query)` values.

For Phase 2, `h_batch` must be derived from the first-layer FRI domain and the
reviewed quotient degree bound after witness randomization. The verifier must
continue to reject any proof where the `R` domain, first FRI layer, or metadata
disagree.

## Tests required before activation

- Phase 2 config remains fail-closed without all review hashes.
- Phase 2 config remains fail-closed with inconsistent privacy-map hash.
- Phase 2 config remains fail-closed with insufficient `h_i`.
- Phase 2 OODS sampling rejects points in every forbidden domain.
- Phase 1 verifier rejects Phase 2 metadata.
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
`Phase2And3ActivationBlocked` and implement semantic witness randomization.
