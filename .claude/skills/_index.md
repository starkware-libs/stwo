# STWO Skill Registry

## Tier 1 — Critical (Mathematical Foundations + Review)

| Skill | File | Load When |
|-------|------|-----------|
| ZK-STARK Foundations | `zk-stark-foundations.md` | Working on any proof system code |
| Circle STARK Mathematics | `circle-stark-mathematics.md` | Modifying circle points, cosets, domains, FFT, polynomials |
| AIR Constraint Engineering | `air-constraint-engineering.md` | Defining/reviewing constraints, logup, EvalAtRow |
| Finite Field Arithmetic | `finite-field-arithmetic.md` | Modifying M31/CM31/QM31 ops, SIMD field code |
| Soundness Review Checklist | `soundness-review-checklist.md` | Reviewing ANY soundness-critical change |
| Security Review Checklist | `security-review-checklist.md` | Reviewing security-critical changes |

## Tier 2 — Protocol Specifics

| Skill | File | Load When |
|-------|------|-----------|
| FRI Protocol | `fri-protocol.md` | Modifying FRI prover/verifier, parameters, folding |
| Performance Optimization | `performance-optimization.md` | Benchmarking, SIMD, memory, profiling |
| Testing Strategy | `testing-strategy.md` | Adding tests, reviewing coverage, debugging failures |

## Tier 3 — Operations

| Skill | File | Load When |
|-------|------|-----------|
| Rust Codebase Conventions | `rust-codebase-conventions.md` | Contributing code, understanding patterns |
| Debugging ZKP | `debugging-zkp.md` | Proof failures, constraint debugging |

## Living Documents

| Document | File | Purpose |
|----------|------|---------|
| Divergence Log | `paper-implementation-divergence-log.md` | Paper vs code divergences (READ BEFORE MODIFYING THEORY CODE) |

## Distilled Theory References

| Document | File | Purpose |
|----------|------|---------|
| Distillation Index | `.agents/papers/llm/INDEX.llm.md` | Entry point and notation map for theory references |
| Circle STARK Distillation | `.agents/papers/llm/Circle_STARKs.llm.md` | Canonical Circle STARK definitions, algorithms, and invariants |
| STWO Distillation | `.agents/papers/llm/Stwo_Whitepaper.llm.md` | Canonical STWO protocol model, soundness assumptions, and parameters |

## Loading Protocol

1. Load `.agents/papers/llm/INDEX.llm.md` to map concepts and anchors.
2. Load the relevant distilled paper file(s) from `.agents/papers/llm/`.
3. Always load `paper-implementation-divergence-log.md` before modifying any
   theoretically-grounded component.
4. Load the most specific relevant Tier 1 skill for the domain you're working in.
5. For reviews, load the appropriate checklist skill.
6. Tier 2 and 3 skills are loaded as needed for context.
