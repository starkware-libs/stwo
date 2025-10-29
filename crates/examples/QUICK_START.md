# Quick Start: Learning stwo STARKs

## 🚀 Start Here (5 minutes)

### Step 1: Read the Learning Guide
```bash
cd /home/uacias/dev/stwo/stwo
cat crates/examples/LEARNING_GUIDE.md | less
```

### Step 2: Run the Simplest Example
```bash
cargo test --package stwo-examples test_docs_component_proper_prove_and_verify -- --nocapture
```

You'll see:
- **Inputs**: All `10`
- **Outputs**: All `100001` (10^5 + 1)
- **LogUp verification**: Sums cancel to `0` ✓
- **Proof verified** ✓

### Step 3: Read the Heavily Commented Code
```bash
cat crates/examples/src/docs_component_proper/mod.rs | less
```

Look for comment blocks starting with:
```rust
// ============================================================================
// CONCEPT: [Topic Name]
// ============================================================================
```

---

## 📚 The 3 Examples (In Order)

### 1. docs_component_proper (START HERE)
**What**: Proves `10^5 + 1 = 100001`
**Why**: Simple math, all concepts explained
**Time**: 1-2 hours to fully understand

### 2. scheduler_poseidon (NEXT)
**What**: Proves Poseidon2 hash computations
**Why**: Real crypto, same pattern as #1, adds batching
**Time**: 2-3 hours

### 3. new_example (ADVANCED)
**What**: Dynamic number of Poseidon calls
**Why**: Conditional execution, production flexibility
**Time**: 1-2 hours

---

## 🎯 Key Concepts (5 minute summary)

### 1. Component Composition
```
Provider Component (e.g., Computing/Poseidon)
  ├─ Generates data (trace generation, off-chain)
  ├─ Has CONSTRAINTS (verify correctness)
  └─ Provides verified data via LogUp

Consumer Component (e.g., Scheduling/Scheduler)
  ├─ COPIES data from Provider
  ├─ NO constraints
  └─ Requests data via LogUp

LogUp ensures they use IDENTICAL data (sums cancel to 0)
```

### 2. Trace Generation vs Constraints
| Aspect | Trace Generation | Constraints |
|--------|------------------|-------------|
| Where | Off-chain (prover) | On-chain (verified) |
| Language | Normal Rust | Algebraic expressions |
| Purpose | Compute results | Verify correctness |

### 3. LogUp (Lookup Arguments)
```rust
// Provider (has constraints):
eval.add_to_relation(-1, input);   // -1/hash(input)
eval.add_to_relation(+1, output);  // +1/hash(output)

// Consumer (no constraints):
eval.add_to_relation(+1, input);   // +1/hash(input)
eval.add_to_relation(-1, output);  // -1/hash(output)

// Verification:
// provider_sum + consumer_sum = 0 ✓
// This proves they use the SAME (input, output) pairs!
```

### 4. Batching Optimization
Instead of 1 operation per row, pack 8:
```
Row structure in scheduler_poseidon:
[input₀(16), output₀(16), input₁(16), output₁(16), ..., input₇(16), output₇(16)]
= 256 columns per row

Benefit: ~8× smaller proof!
```

---

## 🔍 Quick Reference

### File Structure
```
crates/examples/src/
├── docs_component_proper/mod.rs   ← Start here!
├── scheduler_poseidon/mod.rs      ← Real crypto
├── new_example/mod.rs             ← Advanced
└── poseidon/mod.rs                ← Poseidon2 implementation
```

### Key Functions to Read

**In `docs_component_proper/mod.rs`**:
1. Lines 1-32: Module doc (overview)
2. Lines 217-269: `SchedulingEval::evaluate()` (LogUp requesting)
3. Lines 299-355: `ComputingEval::evaluate()` (Constraints + LogUp providing)
4. Lines 409-452: `gen_computing_trace()` (Off-chain computation)
5. Lines 469-480: `gen_scheduling_trace()` (Copying pattern)

**In `scheduler_poseidon/mod.rs`**:
1. Lines 1-45: Module doc (batching explanation)
2. Lines 267-323: `SchedulerEval::evaluate()` (Batched LogUp)
3. Lines 366-406: `gen_scheduler_trace()` (Copying 8 instances)

---

## ✅ Learning Checklist

### Week 1: Fundamentals
- [ ] Read `LEARNING_GUIDE.md`
- [ ] Run `docs_component_proper` test
- [ ] Read `docs_component_proper/mod.rs` with all comments
- [ ] Can explain: What is LogUp?
- [ ] Can explain: Provider vs Consumer pattern
- [ ] Can explain: Trace generation vs Constraints

### Week 2: Production Patterns
- [ ] Run `scheduler_poseidon` test
- [ ] Read `scheduler_poseidon/mod.rs` with comments
- [ ] Can explain: Why batching?
- [ ] Can explain: Multi-element state LogUp
- [ ] Compare patterns between example 1 and 2

### Week 3: Advanced
- [ ] Run `new_example` test
- [ ] Understand `is_active` flag
- [ ] Understand conditional LogUp
- [ ] Try modifying examples

---

## 💡 Common Questions

**Q: Why does LogUp sum need to equal zero?**
A: Provider uses opposite signs from Consumer. If they use the SAME data, the fractions cancel out perfectly. If data differs, they DON'T cancel → proof fails!

**Q: What's the difference between docs_component and docs_component_proper?**
A:
- `docs_component` (original): Scheduling computes, Computing verifies (WRONG pattern)
- `docs_component_proper` (this): Computing computes, Scheduling copies (RIGHT pattern)

**Q: Can I skip to scheduler_poseidon?**
A: Not recommended! Start with `docs_component_proper` to understand fundamentals first. The patterns are identical, just simpler math.

**Q: How long to learn this?**
A:
- Basic understanding: 1 day
- Comfortable writing components: 1 week
- Production-ready: 2-3 weeks

---

## 🆘 Getting Help

1. **Read the comments** - Both examples are heavily documented
2. **Read `LEARNING_GUIDE.md`** - Comprehensive explanations
3. **Experiment** - Modify examples, break things, see what happens
4. **GitHub Issues** - https://github.com/starkware-libs/stwo

---

## 🎓 Next Steps After Completing Examples

1. **Build your own simple component** (e.g., prove `x^2 + 3x + 2`)
2. **Study Poseidon implementation** (`crates/examples/src/poseidon/mod.rs`)
3. **Read constraint framework** (`crates/constraint-framework/src/lib.rs`)
4. **Explore other examples** (`blake`, `wide_fibonacci`, `state_machine`)
5. **Read stwo internals** (Circle STARKs, FRI)

---

**Good luck! Start with `docs_component_proper` and work your way up!** 🚀
