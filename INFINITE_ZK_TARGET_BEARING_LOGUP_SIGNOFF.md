# Target-Bearing Private LogUp ZK Signoff

## Decision

Private LogUp ZK is accepted only when every private LogUp claim has a verifier-checkable algebraic target.

Recognized claim policy variants:

- `SemanticallyPublic`: the claim is public and bound by nonempty semantic metadata.
- `StatisticalAggregate { aggregate_id }`: the claim is private and belongs to an aggregate with target `Zero` or `PublicExpression`.
- `PrivateUnsupported`: rejected on active private LogUp proof paths.

Rejected claim class:

- Private local masked correction without `Zero`, `PublicExpression`, or aggregate target.

## Staff review

- Math/Crypto: PASS. Target-bearing claims preserve the algebraic statement and match the paper warning that lookup/permutation ZK cannot simply hide challenge-dependent claims.
- Security/Pentest: PASS. Metadata alone is not enough; target policy must be verifier-owned, transcript-bound, and algebraically checked.
- Rust/API: PASS. The API remains fail-closed and avoids a permissive no-target private claim path.
- Performance: PASS for scoped implementation. Aggregate components are only added when a real aggregate target exists.

## Shipping rule

The generic framework may support arbitrary AIRs through provider-declared privacy metadata, but it must not infer private LogUp targets.

Each AIR privacy provider must declare:

- private roots and dependency closure,
- private interaction columns,
- LogUp claim manifest,
- LogUp claim policy,
- aggregate target metadata when claims are private,
- statistical security budget inputs,
- semantic trace domains for private columns.

The framework must reject:

- missing private LogUp manifest,
- missing private LogUp policy,
- `PrivateUnsupported` on active private LogUp paths,
- private LogUp without a matching aggregate group,
- duplicate or unused aggregate groups,
- insufficient statistical security budget,
- metadata or budget hash mismatch,
- missing pre-lookup metadata binding.

## Poseidon status

Poseidon private LogUp remains fail-closed unless its `claimed_sum` is formally tied to a verifier-checkable `Zero`, `PublicExpression`, or aggregate target.

Do not enable Poseidon private LogUp with only a masked local correction. That hides the scalar but does not preserve lookup soundness.
