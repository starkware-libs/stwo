# Infinite STWO ZK STARK generalization status

This status records the current generalization layer for using the paper-style
STARK ZK construction in STWO and later in Infinite.

## Implemented in STWO

- Generic AIR metadata builder:
  `stwo::core::zk::build_stwo_zk_air_metadata`.
- Generic config artifact builder:
  `stwo::core::zk::build_stwo_zk_air_config_artifacts`.
- Generic public-statement metadata hash:
  `stwo::core::zk::canonical_zk_air_public_statement_hash`.
- Checked STWO ZK degree geometry helper:
  `stwo::core::zk::derive_stwo_zk_air_degree_bounds`.
- Poseidon now consumes the generic builder and only supplies AIR-specific
  policy:
  original trace columns are `OrdinaryWitness`, interaction trace columns are
  `LogUp`.
- Generic builder fail-closed rules:
  one scope binding per committed trace tree, no duplicate tree scopes, no
  public/private overlap, singleton private ranges, explicit eligible private
  usage, and full `LogUp` coverage for every `InteractionTrace` tree.
- Poseidon ZK proof path still verifies original STWO public Poseidon and
  private Poseidon ZK tests through explicit ZK APIs.

## Generality boundary

The core builder is general for STWO AIRs that use the current two-way
composition split geometry. It does not infer soundness-critical private
degree growth from AIR syntax. Every future AIR must provide a reviewed private
degree expansion or a reviewed stronger degree plan.

The public metadata intentionally leaks public circuit metadata: tree roles,
column counts, private/public column positions, degree bounds, and stable
metadata hashes. Do not use this builder when the private-column policy itself
is confidential.

## Future AIR integration checklist

- Build `component.trace_log_degree_bounds()` from the actual component.
- Provide exactly one `ZkTraceTreeScopeBinding` for every committed trace tree.
- Classify every private singleton range with an explicit
  `ZkPrivateColumnUsage`.
- Mark every private LogUp interaction column as `LogUp`; the generic builder
  rejects omitted interaction-tree columns.
- Compute or supply a reviewed private constraint expansion for the AIR.
- Call `derive_stwo_zk_air_degree_bounds` with the reviewed private expansion.
- Call `build_stwo_zk_air_metadata` and
  `build_stwo_zk_air_config_artifacts`.
- Mix verifier-owned ZK metadata before any affected Fiat-Shamir challenge,
  especially lookup and LogUp challenges.
- Keep default public `prove` / `verify` paths unchanged.

## Poseidon caveat

The existing Poseidon example still uses placeholder round constants. This is
a pre-existing example-correctness caveat, not a ZK masking soundness change.
Production Poseidon constants should be integrated from a primary source before
using the example as a production Poseidon2 hash.

## Benchmarks

Benchmark/report wiring exists in `crates/examples/benches/poseidon.rs`.
Benchmarks are opt-in:

```text
STWO_RUN_POSEIDON_PROOF_BENCH=1 cargo bench -p stwo-examples --bench poseidon
```

Markdown report generation is opt-in:

```text
STWO_POSEIDON_ZK_BENCH_REPORT=/path/to/report.md cargo bench -p stwo-examples --bench poseidon
```

No benchmarks were run while creating this status.

## Infinite handoff

Use this STWO branch as the dependency source for Infinite once formatting,
tests, and benchmarks are accepted. Remove the toy masking integration in
Infinite and call the explicit STWO ZK APIs with an Infinite-specific metadata
builder that follows the checklist above.
