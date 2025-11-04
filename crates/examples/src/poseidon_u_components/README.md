# Poseidon Component-Based Implementation

This is a component-based implementation of Poseidon hash following the patterns from `stark_appv2_safe`.

## Architecture

The implementation consists of two components:

1. **Computing Component** (`computing.rs`) - Computes the actual Poseidon permutation
2. **Scheduler Component** (`scheduler.rs`) - Manages which rows are active

## Key Features

### Optimization: Selective Computation
- **Only computes Poseidon for active rows** (rows with actual messages)
- **Padding rows are left as zeros** and disabled via `is_active` selector
- Example: Computing 2 messages out of 128 rows → **64x speedup**

### LogUp Protocol
- Uses LogUp (Logarithmic Derivative Lookup) for state transitions
- Computing component provides: `+is_active/initial_state` and `-is_active/final_state`
- The sum of all claimed_sums must equal zero (LogUp property)

## File Structure

```
poseidon_u_components/
├── mod.rs              # Main module, prove function, helper functions
├── computing.rs        # Computing component constraints
├── scheduler.rs        # Scheduler component constraints
├── trace_gen.rs        # Trace generation for both components
└── README.md           # This file
```

## Following stark_appv2_safe Pattern

This implementation closely follows `stark_appv2_safe/circuit/src/multi_fib/`:

### Similarities:
1. **Preprocessed columns**: `is_first` and `is_active` (like `is_target` in fib)
2. **Reading from trace**: Interaction trace reads DIRECTLY from trace columns (not from separate lookup_data)
3. **Selector masking**: Uses preprocessed column for multiplicity in LogUp
4. **Combined LogUp formula**: Combines multiple fractions in trace generation (like scheduler in fib)

### Key Implementation Details:

#### Trace Generation (`trace_gen.rs`)
```rust
// Read state values from trace columns (already bit-reversed!)
let initial_state_packed: [_; N_STATE] =
    std::array::from_fn(|i| trace[initial_state_start + i].values.data[vec_row]);
let final_state_packed: [_; N_STATE] =
    std::array::from_fn(|i| trace[final_state_start + i].values.data[vec_row]);

// Use preprocessed is_active for masking
let is_active_packed: PackedSecureField = is_active_data.data[vec_row].into();

// Combined formula (for finalize_logup_in_pairs):
// +is_active/initial + (-is_active)/final = is_active*(final-initial)/(initial*final)
let numerator = is_active_packed * (denom1 - denom0);
let denominator = denom0 * denom1;
```

#### Constraints (`computing.rs`)
```rust
// TWO add_to_relation calls (paired by finalize_logup_in_pairs):
eval.add_to_relation(RelationEntry::new(
    &self.lookup_elements,
    is_active.clone().into(),    // +is_active
    &initial_state_curr,
));
eval.add_to_relation(RelationEntry::new(
    &self.lookup_elements,
    (-is_active.clone()).into(), // -is_active
    &final_state_curr,
));

eval.finalize_logup_in_pairs(); // Expects 2 fractions
```

## Known Issue: ConstraintsNotSatisfied

**STATUS**: The proof generation currently fails with `ConstraintsNotSatisfied`.

**Symptom**: The claimed_sum from computing component is non-zero, when it should be ~0.

**What We've Tried**:
1. ✅ Reading from trace columns (like stark_appv2_safe) - DONE
2. ✅ Using preprocessed is_active column - DONE
3. ✅ Correct LogUp formula matching constraints - DONE
4. ✅ Proper bit-reversing - DONE

**Potential Issues to Investigate**:
1. **Column indexing**: Are we reading the correct columns for initial_state/final_state?
   - initial_state should be at columns `RATE..(RATE+N_STATE)` = 8-23
   - final_state should be at last N_STATE columns = 166-181

2. **LogUp formula**: Is our combined formula mathematically correct?
   - Formula: `is_active * (denom1 - denom0) / (denom0 * denom1)`
   - Should equal: `+is_active/denom0 + (-is_active)/denom1`

3. **finalize_logup_in_pairs()**: Does it expect exactly 2 fractions?
   - We have 2 add_to_relation calls in constraints
   - We generate 1 combined column in trace_gen
   - Is this the correct pairing?

4. **Trace column order**: Are all N_COLUMNS=182 columns in the expected order?

## Testing

Run tests:
```bash
# Simple trace generation test
cargo test --package stwo-examples --lib poseidon_u_components::tests::test_trace_generation -- --nocapture

# Full prove and verify test (currently fails)
cargo test --package stwo-examples --lib poseidon_u_components::tests::test_prove_and_verify_with_optimization -- --nocapture
```

## For Team Review

**What to check**:
1. Is the LogUp formula in `trace_gen.rs` correct for combining two fractions?
2. Are we reading the right columns from trace (initial_state at 8-23, final_state at 166-181)?
3. Does `finalize_logup_in_pairs()` work correctly with our setup (2 constraints, 1 combined trace column)?
4. Is there something different about Poseidon (16-element state) vs Fibonacci (1-element state) that breaks the pattern?

**Debugging suggestions**:
- Add `RUST_LOG=info` to see LogUp claimed_sum values
- Compare with working stark_appv2_safe to see structural differences
- Check if the issue is in constraints vs trace generation

## References

- Similar implementation: `stark_appv2_safe/circuit/src/multi_fib/`
- LogUp in stwo: `crates/constraint-framework/src/prover/logup.rs`
- Poseidon spec: https://eprint.iacr.org/2019/458.pdf
