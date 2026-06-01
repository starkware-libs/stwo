# Infinite ZK Privacy Demonstration

## Scope

This report documents the concrete privacy regression added for the first
private-witness Wide Fibonacci ZK-STARK example.

The demonstration is not a standalone proof of zero knowledge. It is executable
evidence for the implementation-level privacy invariant required by the paper
integration: for the same public statement and same underlying witness, prover
randomness changes the private committed/opened material while verifier-owned
checks still accept both proofs.

## Test

`crates/examples/src/wide_fibonacci/mod.rs`

```text
test_wide_fib_zk_private_witness_repeated_proofs_hide_private_material
```

The test constructs two proofs with:

- the same public metadata
- the same Wide Fibonacci witness
- the same verifier-owned privacy map and audit
- different witness-randomization and proof-mask RNG seeds

It asserts:

- public metadata is identical
- the empty preprocessed commitment is identical
- the private witness commitment differs
- the masked composition commitment differs
- private-tree sampled values differ
- the FRI batch-mask commitment differs
- both proofs verify with the same verifier-owned audit

## Security interpretation

The verifier does not receive the prover's witness randomizer coefficients.
The verifier receives randomized commitments, randomized sampled values, and
public audit metadata. The audit reconstructs the query closure and checks that
the declared randomizer space has enough independent rank to cover verifier
observations.

The privacy claim supported by this test is:

```text
same witness + same public statement + different prover randomness
  => different private proof material
  => both proofs still verify under the same public verifier policy
```

The stronger zero-knowledge claim depends on the paper argument plus the
implementation's rank/query-closure checks. This test is intended to catch
implementation regressions that accidentally make private proof material
deterministic or verifier-recoverable.

