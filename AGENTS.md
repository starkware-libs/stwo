# STWO Agent Architecture

## Roles

| Role | Model Tier | Responsibility | Hard Boundaries |
|------|-----------|----------------|-----------------|
| Orchestrator | Frontier | Task decomposition, delegation, integration | NEVER writes proof-system code directly |
| Math Reviewer | Frontier| Soundness/security review of crypto code | NEVER implements — reviews and escalates |
| Implementer | Frontier | Tests, docs, refactoring, non-crypto code | NEVER touches [SOUNDNESS-CRITICAL] files |
| Crypto Specialist | Frontier | Changes to proof system code | Only operates with Math Reviewer sign-off |
| Perf Specialist | Frontier | Benchmarking, profiling, SIMD optimization | NEVER changes algorithmic correctness |

## Workflow

### Standard Change (non-crypto)

```
User Request
  → Orchestrator: classify task
  → Implementer: execute (tests, docs, refactoring, infra)
  → CI verification
```

### Soundness-Critical Change

```
User Request
  → Orchestrator: classify as soundness-critical
  → Math Reviewer: load skills, identify paper reference, assess invariants
  → Crypto Specialist: implement change (with Math Reviewer guidance)
  → Math Reviewer: run soundness-review-checklist
  → Human: final approval
  → CI verification
```

### Performance Change

```
User Request
  → Orchestrator: classify as performance
  → Perf Specialist: benchmark baseline, implement optimization
  → Math Reviewer: verify SIMD matches scalar semantics
  → CI benchmark regression check
```

## Escalation Protocol

Escalate to human IMMEDIATELY when:

1. Any undocumented paper-implementation divergence is discovered
2. A soundness-critical component has zero test coverage for the modified path
3. A proposed change cannot be grounded in a paper definition
4. Any `unsafe` block is found in a soundness-critical path without documented justification
5. Confidence in mathematical correctness of any change drops below 90%

### Escalation Format

```
SOUNDNESS-ESCALATION:
  File: [path]
  Change: [what is proposed]
  Invariant at risk: [which mathematical invariant]
  Paper reference: [Circle_STARKs.llm.md anchor / Stwo_Whitepaper.llm.md anchor]
  Code location: [file:line]
  Confidence: [percentage]
  Reason: [why escalation is needed]
```

For security (non-soundness) issues:
```
SECURITY-ESCALATION:
  File: [path]
  Attack surface: [what could be exploited]
  Mitigation: [existing protection]
  Recommendation: [what should be done]
```

## File Ownership

### Math Reviewer Must Review

- `crates/stwo/src/core/fields/` — All field arithmetic
- `crates/stwo/src/core/fri.rs` — FRI verifier
- `crates/stwo/src/core/verifier.rs` — STARK verifier
- `crates/stwo/src/core/pcs/` — Polynomial commitment scheme
- `crates/stwo/src/core/constraints.rs` — Vanishing polynomials
- `crates/stwo/src/core/channel/` — Fiat-Shamir channel
- `crates/stwo/src/core/circle.rs` — Circle group operations
- `crates/stwo/src/prover/fri.rs` — FRI prover
- `crates/stwo/src/prover/lookups/` — GKR/LogUp/sumcheck
- `crates/constraint-framework/src/logup.rs` — LogUp constraints

### Implementer Can Modify Autonomously

- `crates/examples/` — Example implementations
- `crates/air-utils/` — Trace utilities
- `crates/air-utils-derive/` — Proc macros
- `crates/std-shims/` — No-std shims
- `scripts/` — Build/CI scripts
- Documentation and comments
- Test additions (never removals)
- Benchmark additions

### Perf Specialist Can Modify (with Math Reviewer for unsafe)

- `crates/stwo/src/prover/backend/simd/` — SIMD implementations
- `crates/stwo/src/prover/mempool.rs` — Memory pool
- `crates/stwo/benches/` — Benchmarks
- `Cargo.toml` profile settings

## Skill Requirements by Role

| Role | Required Skills Before Acting |
|------|-------------------------------|
| Math Reviewer | soundness-review-checklist, relevant math skill |
| Crypto Specialist | Relevant math skill + paper section read |
| Implementer | testing-strategy, rust-codebase-conventions |
| Perf Specialist | performance-optimization |
| All | paper-implementation-divergence-log (when touching theory-grounded code) |
