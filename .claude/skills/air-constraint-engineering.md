---
name: air-constraint-engineering
description: >
  Defines how to audit and modify AIR constraints in STWO. Covers Flat AIR
  model, constraint degree bounds, EvalAtRow trait, LogUp interactions,
  constraint debugging, and soundness criteria. Use when working on constraint
  definitions, the constraint framework, logup interactions, or any
  component's constraints.
---

# AIR Constraint Engineering

## Canonical Theory Sources

- `.agents/papers/llm/INDEX.llm.md` — unified notation and anchor map
- `.agents/papers/llm/Circle_STARKs.llm.md` — AIR quotient model and degree constraints
- `.agents/papers/llm/Stwo_Whitepaper.llm.md` — Flat AIR model and logUp protocol framing

## Flat AIR Model

STWO implements Flat AIRs as captured in the STWO distillation:
`Stwo_Whitepaper.llm.md` (`def:flat:AIR`, `def:flat:AIR:solution`,
`e:flat:AIR:consistency`).

### Structure

A Flat AIR consists of:
- **Columns**: Each column is a polynomial evaluated on the trace domain
- **Constraints**: Polynomial identities that must hold at every row
- **Interactions**: LogUp-based lookup arguments between columns

### Trace Layout

| Index | Name | Description |
|-------|------|-------------|
| 0 | PREPROCESSED_TRACE_IDX | Fixed columns (selectors, preprocessed data) |
| 1 | ORIGINAL_TRACE_IDX | Witness columns (prover-generated) |
| 2 | INTERACTION_TRACE_IDX | LogUp interaction columns |

**Implementation**: `crates/constraint-framework/src/lib.rs` — `PREPROCESSED_TRACE_IDX`, `ORIGINAL_TRACE_IDX`, `INTERACTION_TRACE_IDX`

### Constraint Degree

The AIR degree d = max_i deg(P_i) determines:
- The composition polynomial degree: (d-1)*N
- The number of composition polynomial splits: controlled by COMPOSITION_LOG_SPLIT
- The evaluation domain size: >= d * N * blowup_factor

**Source**: `.agents/papers/llm/Circle_STARKs.llm.md` — AIR quotient identity and
parameter rules (`e:overall:identity`, Section "6. Parameter Rules")

**Implementation**: Degree tracking in `crates/constraint-framework/src/expr/degree.rs`

## The EvalAtRow Trait

**File**: `crates/constraint-framework/src/lib.rs` — `EvalAtRow` trait

This is the core abstraction for constraint evaluation. A component
implements `FrameworkEval::evaluate()` which calls methods on an
`EvalAtRow` implementor.

Key associated types:
- `F` — Column value type (BaseField for CPU, PackedBaseField for SIMD,
  SecureField for out-of-domain evaluation)
- `EF` — Extended field type for constraint accumulation

### Implementations

| Evaluator | Purpose | File |
|-----------|---------|------|
| `PointEvaluator` | Verifier: evaluates at OODS point | `point.rs` |
| `InfoEvaluator` | Extracts constraint metadata | `info.rs` |
| `SimdDomainEvaluator` | Prover: SIMD evaluation on domain | `prover/simd_domain.rs` |
| `CpuDomainEvaluator` | Prover: CPU evaluation on domain | `prover/cpu_domain.rs` |
| `AssertEvaluator` | Testing: asserts constraints hold | `prover/assert.rs` |

## LogUp Interactions

LogUp is the lookup argument protocol used in STWO for cross-component
communication (e.g., range checks, memory lookups).

**Source**: `.agents/papers/llm/Stwo_Whitepaper.llm.md` — IOPP logUp constraints
(`prot:STARK:IOPP`, `e:constraint:uses`, `e:constraint:yields`,
`e:constraint:sum:increment`)

### Protocol

For a lookup table T and access columns (a_1, ..., a_k):

```
sum_i 1/(alpha - a_i) = sum_j mult_j/(alpha - t_j)
```

Where alpha is a random challenge from the Fiat-Shamir transcript.

### Implementation

- Constraint definitions: `crates/constraint-framework/src/logup.rs`
- Prover logup: `crates/constraint-framework/src/prover/logup.rs`
- `LogupTraceGenerator` — generates interaction trace columns
- `LogupColGenerator` — generates individual logup columns
- `FractionWriter` — writes logup fractions to the trace

### Batching

Entries can be grouped into batches via the `Batching` type.
Pairwise batching (two entries per batch) is the standard optimization
that halves the number of interaction trace columns.

## Component Structure

**File**: `crates/constraint-framework/src/component.rs`

A component implements:
1. `FrameworkEval::evaluate(&self, eval: &mut impl EvalAtRow)` — constraint logic
2. Column layout (number and sizes of trace columns)
3. LogUp interaction declarations

The `FrameworkComponent` wrapper connects a `FrameworkEval` to the STWO
proving system.

## Constraint Correctness Criteria

### Completeness
Every valid trace must satisfy all constraints. If a correct witness
produces a constraint violation, the proof system rejects valid proofs.

### Soundness (No Under-Constraint)
Every trace satisfying all constraints must represent a valid computation.
Missing constraints allow the prover to create proofs for invalid witnesses.

**This is the more dangerous failure mode.** A missing constraint is invisible
in happy-path testing. It only manifests when an adversary exploits it.

### Degree Bounds
Constraints must not exceed the declared degree. If a constraint has actual
degree > declared degree, the quotient polynomial will not be low-degree,
and the proof will fail for valid witnesses.

## Debugging Constraint Failures

### Workflow

1. Use `AssertEvaluator` (`assert_constraints_on_trace`) to check constraints
   row-by-row on a concrete trace.
2. Use `RelationTracker` (`prover/relation_tracker.rs`) to track logup
   relation sums and identify imbalanced lookups.
3. Check constraint degree with `InfoEvaluator` — it reports max degree.
4. If FRI fails but constraints pass: the quotient polynomial degree bound
   may be wrong (check COMPOSITION_LOG_SPLIT).

### Common Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Proof fails, constraints pass on trace | Degree bound mismatch | Check declared vs actual constraint degree |
| Logup sum != 0 | Unbalanced lookups | Use RelationTracker to find imbalanced relation |
| Verifier rejects valid proof | OODS eval mismatch | Check composition polynomial extraction |
| Constraint evaluates to non-zero | Wrong constraint logic | Debug with AssertEvaluator on single rows |

## Security Invariants

INVARIANT-AIR-1: The constraint system must be complete — all valid
witnesses satisfy all constraints.

INVARIANT-AIR-2: The constraint system must be sound — only valid
witnesses satisfy all constraints. This requires careful analysis of
what the constraints actually enforce.

INVARIANT-AIR-3: LogUp interaction sums must balance — total sum of
all logup fractions must equal zero.

INVARIANT-AIR-4: Constraint degree must not exceed the declared bound.
The composition polynomial quotient assumes a specific degree.

## Review Checklist

Before approving constraint changes:
- [ ] All new constraints are tested with AssertEvaluator on valid traces
- [ ] Constraint degree is verified (use InfoEvaluator)
- [ ] LogUp interactions balance (use RelationTracker)
- [ ] No constraints were removed without documented justification
- [ ] New columns are properly registered in the trace layout
- [ ] Selector polynomials correctly activate/deactivate constraints

## Forbidden Actions

In this domain, agents must NEVER:
- Remove a constraint without proving it is redundant (formally)
- Add a constraint that exceeds the degree bound without updating the bound
- Modify logup interaction definitions without verifying balance
- Change COMPOSITION_LOG_SPLIT without understanding its impact on all users
- Leave a "TODO: validate logup" in production constraint code
