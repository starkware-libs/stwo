# Learning Guide: STARK Proofs with stwo

## 🎯 Goal
This guide teaches you how to build STARK proofs using the `stwo` library through three progressive examples.

---

## 📚 Examples Overview (Start → Advanced)

### 1. **docs_component_proper** ⭐ START HERE!
**Path**: `crates/examples/src/docs_component_proper/mod.rs`

**What it proves**: `output = input^5 + 1`
- Input: `10`
- Output: `100001` (10^5 + 1 = 100,001)

**Why start here**:
- ✅ Simple math you can verify by hand
- ✅ Shows core concepts clearly
- ✅ **Heavily commented** - Every major concept explained in detail
- ✅ Only 2-3 columns per component
- ✅ Best for learning the fundamentals

**Run it**:
```bash
cargo test --package stwo-examples test_docs_component_proper_prove_and_verify -- --nocapture
```

**Key concepts**:
1. **Component composition** - Computing + Scheduling cooperation
2. **Constraints** - Algebraic rules that verify computation
3. **LogUp** - How components share verified data
4. **Trace generation** - Off-chain computation
5. **Proof/Verification** - The complete flow

---

### 2. **scheduler_poseidon** 🔐 REALISTIC EXAMPLE
**Path**: `crates/examples/src/scheduler_poseidon/mod.rs`

**What it proves**: Poseidon2 hash computation
- Input: 16-element state `[0,1,2,...,15]`
- Output: 16-element hashed state `[462565134, 1755139106, ...]`

**Why learn this**:
- ✅ Real cryptographic primitive
- ✅ Shows batching optimization (8 hashes per row)
- ✅ **Heavily commented** - Explains batching and multi-element states
- ✅ Production-ready pattern
- ✅ Reusable component architecture

**Run it**:
```bash
cargo test --package stwo-examples test_scheduler_poseidon_prove_and_verify -- --nocapture
```

**New concepts**:
1. **Batching** - Multiple instances per row (proof size optimization)
2. **Complex state** - 16 elements instead of 1
3. **Reusable components** - Poseidon can be used by many schedulers

---

### 3. **new_example** 🚀 ADVANCED
**Path**: `crates/examples/src/new_example/mod.rs`

**What it proves**: Dynamic number of Poseidon calls
- Variable from 1 to 1024 calls
- Uses `is_active` flag for conditional execution

**Why learn this**:
- ✅ Real-world flexibility (variable workloads)
- ✅ Conditional LogUp (only active rows participate)
- ✅ Padding strategies

**Run it**:
```bash
cargo test --package stwo-examples test_new_example_prove_and_verify -- --nocapture
```

**New concepts**:
1. **Conditional LogUp** - Multiply coefficient by `is_active`
2. **Dynamic sizing** - Handle variable number of operations
3. **Padding** - Fill unused rows with zeros

---

## 🧠 Core Concepts Explained

### 1. Component Composition Pattern

```
┌─────────────────────┐         ┌──────────────────────┐
│ Computing Component │         │ Scheduling Component │
│  (The "Worker")     │         │  (The "Requester")   │
├─────────────────────┤         ├──────────────────────┤
│ • Generates input   │         │ • COPIES input       │
│ • Computes output   │────────▶│ • COPIES output      │
│ • Has CONSTRAINTS   │  LogUp  │ • NO constraints     │
│ • Verifies math     │         │ • Just requests data │
└─────────────────────┘         └──────────────────────┘
```

**Why this pattern?**
- Computing has complex logic → keep it in one reusable place
- Scheduling just needs results → copies via LogUp
- LogUp ensures they use IDENTICAL data

---

### 2. Constraints vs Trace Generation

| Aspect | Trace Generation | Constraints |
|--------|------------------|-------------|
| **Where** | Off-chain (prover only) | On-chain (verified) |
| **Language** | Normal Rust code | Algebraic expressions |
| **Purpose** | Compute results | Verify correctness |
| **Example** | `v.pow(5) + 1` | `output - input^5 - 1 = 0` |

**Flow**:
1. Prover generates trace (off-chain): "10^5 + 1 = 100001"
2. Prover creates proof using constraints
3. Verifier checks constraints hold (doesn't recompute!)

---

### 3. LogUp (Logarithmic Derivative Lookups)

**Purpose**: Let components share verified data without duplicating constraints.

**How it works**:
```rust
// Computing (Provider):
eval.add_to_relation(-1, input);   // -1/hash(input)
eval.add_to_relation(+1, output);  // +1/hash(output)
// Creates: (hash(input) - hash(output)) / (...)

// Scheduling (Consumer):
eval.add_to_relation(+1, input);   // +1/hash(input)
eval.add_to_relation(-1, output);  // -1/hash(output)
// Creates: (hash(output) - hash(input)) / (...)

// Verification:
// Provider's sum + Consumer's sum = 0 ✓
// This proves they use the SAME (input, output) pairs!
```

**Key insight**: Opposite signs cancel out ONLY if data matches exactly.

---

### 4. The Complete Proof Flow

```
PROVER SIDE:
1. Generate trace columns (off-chain computation)
   └─▶ Computing: [input, intermediate, output]
   └─▶ Scheduling: [input, output]

2. Commit trace to Merkle tree
   └─▶ Sends commitment to verifier

3. Draw lookup elements (Fiat-Shamir)
   └─▶ Random challenge from transcript

4. Generate LogUp interaction columns
   └─▶ Computing: claimed_sum_1
   └─▶ Scheduling: claimed_sum_2
   └─▶ Verify: claimed_sum_1 + claimed_sum_2 = 0 ✓

5. Commit LogUp columns

6. Generate STARK proof
   └─▶ FRI polynomial commitment
   └─▶ Constraint evaluation queries

VERIFIER SIDE:
1. Receive commitments and claimed sums

2. Draw same lookup elements (Fiat-Shamir)

3. Verify claimed_sum_1 + claimed_sum_2 = 0

4. Verify STARK proof
   └─▶ Check FRI
   └─▶ Check constraint equations
   └─▶ Check Merkle proofs

Result: Proof verified ✓
```

---

## 🎓 Recommended Learning Path

### Week 1: Understand `docs_component_proper`
1. **Read the file top-to-bottom** - Start with the module doc comment (lines 1-32)
2. **Follow the comment blocks** - Each major concept has a "CONCEPT:" block
3. **Run the test** and observe output showing inputs/outputs
4. **Experiment**: Try modifying:
   - Change computation to `x^3 + 7` instead of `x^5 + 1`
   - Change input from `10` to `5`, observe changes
   - Try breaking LogUp (use different data in Scheduling)

**Key questions to answer**:
- Where does trace generation happen? (Hint: `gen_computing_trace()`)
- Where do constraints get evaluated? (Hint: `ComputingEval::evaluate()`)
- What would happen if Scheduling used different data than Computing? (Sum wouldn't be 0!)
- How does LogUp ensure data matches? (Opposite signs cancel out)

### Week 2: Study `scheduler_poseidon`
1. **Read module doc comment** (lines 1-45) - Understand batching concept
2. **Compare to `docs_component_proper`** - Find identical patterns:
   - Provider/Consumer pattern (same!)
   - LogUp with opposite signs (same!)
   - Trace copying (same!)
   - Only difference: 16-element states + batching
3. **Read comment blocks** - Focus on "CONCEPT: Batched LogUp" (line 268)
4. **Observe output** - See the 16-element input/output states
5. **Calculate proof size** - 256 hashes: 32 rows vs 256 rows without batching

**Key questions to answer**:
- Why batch 8 instances instead of 1? (8× smaller proof!)
- Where does Poseidon generate inputs? (`poseidon/mod.rs`, `gen_trace()`)
- Where does Scheduler copy data? (`gen_scheduler_trace()`, lines 387-399)
- Could you use 16 per row? (Yes! Just change `N_INSTANCES_PER_ROW`)

### Week 3: Explore `new_example`
1. Understand `is_active` flag purpose
2. See conditional LogUp multiplication
3. Try changing `num_calls` to different values
4. Understand padding strategy

**Key questions to answer**:
- When would you use `new_example` vs `scheduler_poseidon`?
- What happens if `is_active = 0`?
- How does proof size change with different `num_calls`?

---

## 💡 Common Patterns

### Pattern 1: Provider/Consumer (Most Common)
```rust
// Provider (has constraints):
eval.add_constraint(output - compute(input));  // Verify
eval.add_to_relation(-1, input);               // Provide
eval.add_to_relation(+1, output);

// Consumer (no constraints):
eval.add_to_relation(+1, input);               // Request
eval.add_to_relation(-1, output);
```

**Use case**: Reusable crypto primitives (Poseidon, ECDSA, etc.)

---

### Pattern 2: Batching (Performance)
```rust
// Instead of 1 operation per row:
for _ in 0..BATCH_SIZE {  // e.g., 8
    let input = eval.next_trace_mask();
    let output = eval.next_trace_mask();
    // ... LogUp ...
}
```

**Benefit**: Reduces proof size by ~BATCH_SIZE factor

---

### Pattern 3: Conditional Execution (Flexibility)
```rust
let is_active = eval.next_trace_mask();
let input = eval.next_trace_mask();
let output = eval.next_trace_mask();

// Only participate in LogUp if active:
eval.add_to_relation(is_active.clone(), input);
eval.add_to_relation(-is_active, output);
```

**Benefit**: Handle variable workloads efficiently

---

## 🔧 Hands-On Exercises

### Exercise 1: Modify `docs_component_proper`
Change the computation from `x^5 + 1` to `x^3 + 7`:

```rust
// In gen_computing_trace():
let output_col = BaseColumn::from_iter(
    input_col.as_slice().iter().map(|&v| v.pow(3) + M31::from(7))
);

// In ComputingEval::evaluate():
eval.add_constraint(
    output_col.clone() - input_col.clone() * input_col.clone() * input_col.clone() - E::F::from(7)
);
```

**Expected**: Proof still verifies! Output changes to `10^3 + 7 = 1007`.

---

### Exercise 2: Change Batching in `scheduler_poseidon`
Try changing `N_INSTANCES_PER_ROW` from 8 to 4:

```rust
const N_INSTANCES_PER_ROW: usize = 4;  // Was 8
```

**Expected**: More rows, but still works! Proof will be larger.

---

### Exercise 3: Add Debug Prints
Add prints to see constraint evaluation:

```rust
// In ComputingEval::evaluate():
println!("Evaluating constraint: intermediate = input^3");
eval.add_constraint(...);
```

---

## 📖 Additional Resources

### Key Files to Read:
1. `crates/constraint-framework/src/lib.rs` - Core framework traits
2. `crates/stwo/src/core/air/mod.rs` - AIR (Algebraic Intermediate Representation)
3. `crates/examples/src/poseidon/mod.rs` - Poseidon2 implementation

### Concepts to Research:
- **Circle STARKs** - stwo uses circle curves instead of traditional FFT
- **FRI (Fast Reed-Solomon IOP)** - The polynomial commitment scheme
- **Fiat-Shamir** - Making interactive proofs non-interactive
- **LogUp** - Logarithmic derivative lookups (lookup arguments)

---

## ❓ FAQ

**Q: Why do we need both trace generation AND constraints?**
A: Trace generation does the actual computation (off-chain, fast). Constraints verify it's correct (on-chain, succinct). This separation makes proving efficient!

**Q: Can Scheduling compute instead of copy?**
A: Technically yes, but it defeats the purpose! The pattern is: one component does work + has constraints, others just copy via LogUp. This keeps constraints modular and reusable.

**Q: What's the difference between `add_constraint` and `add_to_relation`?**
A:
- `add_constraint()` - Algebraic rule that must equal zero (e.g., `output - input^5 - 1`)
- `add_to_relation()` - LogUp entry for component communication

**Q: Why does LogUp sum need to equal zero?**
A: Provider adds `+1/input - 1/output`, Consumer adds `-1/input + 1/output`. If they use the same data, these cancel: `(+1-1) + (-1+1) = 0`. If data differs, they DON'T cancel → proof fails!

**Q: When should I use batching?**
A: When you have many identical operations (e.g., 256 Poseidon hashes). Batching reduces proof size significantly (8× smaller with BATCH_SIZE=8).

**Q: What's the performance cost of STARK proofs?**
A:
- Proving time: ~seconds to minutes (depends on computation size)
- Proof size: ~100KB typical (very small!)
- Verification time: ~milliseconds (very fast!)

---

## 🚀 Next Steps

After mastering these examples:

1. **Build your own component** - Start with simple arithmetic
2. **Compose multiple components** - Try 3+ components cooperating
3. **Optimize for production** - Experiment with batching strategies
4. **Study advanced examples** - Look at `blake`, `wide_fibonacci`, `state_machine`
5. **Read stwo internals** - Understand Circle STARKs and FRI

---

## 📬 Getting Help

- **GitHub Issues**: https://github.com/starkware-libs/stwo
- **Documentation**: Read comments in the code (heavily documented!)
- **Community**: Ask questions in PRs or discussions

---

**Good luck! Start with `docs_component_proper` and work your way up!** 🎓
