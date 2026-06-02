# Infinite STWO ZK AIR privacy-provider integration plan

This plan records the staff-reviewed direction for replacing example/test-specific
ZK builders with a general AIR privacy annotation and dependency-closure layer.

## Staff committee decision

The committee conditionally approves the architecture:

- AIR authors declare private/public roots and dependency metadata.
- `core::zk` derives the privacy closure, canonical metadata, degree geometry,
  quotient mask profile, verification config, and proof-public hashes.
- Unknown dependencies must reject or become derived-private. They must never
  silently remain public.
- Verifier APIs must not receive mask seeds, mask coefficients, or enough mask
  material to recover private witness openings.

This is design sign-off, not final production sign-off. Production sign-off
requires the gates below.

## Non-negotiable invariants

- Private roots are explicit AIR/application policy.
- Any value depending on a private root is derived-private.
- Private LogUp inputs imply private LogUp interaction/running-sum columns.
- Claimed sums are public only when semantically proven public.
- Composition/quotient split masking must match STWO's actual split semantics.
- FRI masking must cover OODS/DEEP queries, FRI queries, and last-layer exposure.
- ZK metadata is canonically transcript-bound before affected challenges.
- The verifier checks randomized-oracle consistency without reconstructing the
  private witness.
- Original public STWO `prove` / `verify` paths remain additive and unchanged.

## Target architecture

```text
AIR privacy provider
  -> private roots
  -> public roots
  -> dependency edges
  -> component/tree metadata
  -> generic closure builder
  -> canonical ZK metadata/config artifacts
  -> prover/verifier explicit ZK APIs
```

AIR providers may declare:

- AIR/component identity.
- Original private witness roots.
- Public roots and public statement bindings.
- Dependency edges from private roots to derived columns.
- Whether dependency metadata is complete.
- LogUp scalar claim manifests and reviewed visibility policies when private
  LogUp interaction columns are present.

AIR providers must not manually provide:

- Privacy map hashes.
- Private-column scope hashes.
- Randomized witness domain sizes.
- FRI first-layer sizes.
- Quotient split mask profiles.
- Ad hoc degree slack.

Those are derived by `core::zk`.

## Current integration status

- `core::zk` provider construction exists and derives provider-owned metadata
  through the generic builder.
- Dependency closure is fail-closed for private roots and private LogUp
  interaction columns.
- Blake public proof OODS planning is unblocked by dynamic lift geometry.
- Private LogUp scalar claims are now fail-closed:
  - providers must declare exact per-interaction scalar claim manifests;
  - every claim index in the manifest must have a reviewed policy;
  - stale, extra, wrong-index, unsupported, or incomplete policies reject;
  - manifest and policy data are bound into the canonical public statement hash.
- Poseidon, Blake, StateMachine, and Plonk private LogUp remain intentionally
  blocked because their public `claimed_sum` surfaces are witness-derived and no
  reviewed private-claim protocol exists yet. Blake and StateMachine provider
  coverage is metadata-only/fail-closed and must not be treated as complete
  private LogUp ZK. Plonk coverage is likewise metadata-only/fail-closed.
- WideFib positive private-witness ZK uses provider-derived metadata/config.
- XOR/GKR lookup-subprotocol examples do not currently expose a reviewed
  top-level AIR privacy provider path in this plan. They remain public
  subprotocol coverage until a concrete private-witness statement and public
  oracle/claim binding are specified.
- Review caveat remediation is explicit:
  - fail-closed provider statements bind available public statement and shape
    metadata, not only a blocked marker;
  - AIR-specific tree roles remain explicit semantic provider policy and are
    validated against actual statement/component geometry;
  - `SemanticallyPublic` LogUp claim policy is documented as a reviewed
    assertion only, never as cryptographic hiding;
  - Blake fail-closed provider coverage is test-only and must not be treated as
    production private LogUp support.

## Implementation phases

1. Add production provider API in `core::zk`.
2. Add dependency-closure construction from private roots and dependency edges.
3. Add a generic builder wrapper around the existing canonical metadata builder.
4. Migrate Poseidon from bespoke metadata construction to the provider path.
5. Migrate WideFib away from hand-built ZK configs.
6. Fix the ignored Blake public proof blocker before using Blake as a stress
   test.
7. Add private LogUp scalar claim guardrails for provider-derived ZK metadata.
8. Add Blake provider metadata coverage, but do not call Blake private LogUp ZK
   complete until its scalar claim manifest and private-claim protocol are
   reviewed.
9. Add StateMachine provider coverage.
10. Add Plonk/XOR/GKR providers only where private witnesses are required.
11. Deprecate/remove test-only builders after production providers cover the
    examples.

## Required test gates

- Public STWO compatibility still passes.
- WideFib positive ZK tests pass through provider-derived config.
- Poseidon private LogUp metadata/config rejects until a reviewed private LogUp
  scalar-claim protocol exists.
- Blake public proof passes before Blake ZK work.
- Missing dependency metadata rejects.
- Missing private LogUp dependency rejects or becomes derived-private.
- Missing, stale, wrong-index, unsupported, or incomplete private LogUp scalar
  claim policies reject.
- Tampered privacy map rejects.
- Tampered AIR/component identity rejects.
- Swapped tree/column metadata rejects.
- Tampered masked sampled value rejects.
- Tampered FRI answer rejects.
- Tampered quotient mask profile rejects.
- Transcript-ordering tamper rejects.
- Raw private values are absent from sampled values, queried values, FRI
  answers, public metadata, aux/debug surfaces, logs, and benchmark reports.

## Staff review gates

- Math/Crypto review is required before proof-protocol semantic changes.
- Security/Pentest review is required before accepting verifier/proof-format
  changes.
- Rust/API review is required before stabilizing provider traits.
- Performance review is required after Poseidon and Blake benchmarks.

## Current known blocker

Private LogUp scalar claims are the active blocker for Poseidon, Blake, Plonk,
and StateMachine-style private lookup/permutation examples.

`LogupTraceGenerator::finalize_last()` returns `claimed_sum`, and
`LogupAtRow::new()` uses that scalar as a public constraint parameter. When the
lookup multiset depends on private witness data, the scalar is a
challenge-dependent fingerprint of private data.

The provider guardrail prevents accidental sign-off but is not itself a privacy
fix. A complete implementation still needs either:

- a Math/Crypto-reviewed proof that each scalar is semantically public; or
- a reviewed private LogUp claim protocol that avoids exposing witness-derived
  scalar claims.
