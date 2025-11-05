# Poseidon Optimized - Simplified Implementation with is_active

## Overview

This is a simplified monolithic implementation of Poseidon hash with `is_active` selector optimization, derived from `poseidon_uacias`. The goal was to enable padding rows (inactive rows) to reduce computation when we have fewer messages than table rows.

**Example**: Hash 2 messages in a 128-row table → compute only 2 Poseidon permutations instead of 128 (64x speedup).

## Architecture

- **Single monolithic component** (no Computing/Scheduler split like poseidon_u_components)
- **Preprocessed columns**: is_first, is_active
- **Optimization**: Only computes Poseidon for active rows (n_messages), padding rows remain zeros
- **Constraints**: All multiplied by `is_active` selector to disable them for padding rows

## Implementation Details

### Changes from poseidon_uacias:

1. **Added is_active preprocessed column**:
   - `gen_is_active_column(log_size, n_messages)`: 1 for first n_messages rows, 0 for padding
   - `is_active_column_id(log_size, n_messages)`: PreProcessedColumnId

2. **Modified PoseidonEval struct**:
   ```rust
   pub struct PoseidonEval {
       pub is_active_id: PreProcessedColumnId,  // NEW
       pub n_messages: usize,                   // NEW
       // ... existing fields
   }
   ```

3. **All constraints masked by is_active**:
   - Constraint 1 (first row capacity=0): `is_active * is_first * initial_state[8..16]`
   - Constraint 2 (transitions): `is_active * not_first * (...)`
   - Constraint 3 (Poseidon permutation): `is_active * (...)`

4. **LogUp uses is_active multiplicity**:
   ```rust
   eval.add_to_relation(RelationEntry::new(
       lookup_elements,
       is_active.clone().into(),      // +is_active (not +1)
       &initial_state_curr,
   ));
   eval.add_to_relation(RelationEntry::new(
       lookup_elements,
       (-is_active.clone()).into(),   // -is_active
       &final_state_curr,
   ));
   ```

5. **gen_trace optimized**:
   - Only computes Poseidon for `0..n_messages` rows
   - Padding rows (n_messages..n_rows) remain zeros

6. **gen_interaction_trace fixed**:
   - Uses is_active as numerator multiplier:
     ```rust
     let numerator = is_active_packed * (denom1 - denom0);
     col_gen.write_frac(vec_row, numerator, denom0 * denom1);
     ```

## Test Results

### ✅ Passing Tests (100% active rows):

| Test | Rows | Messages | Active % | Result |
|------|------|----------|----------|--------|
| `test_poseidon_constraints` | 256 | 256 | 100% | ✅ PASS |
| `test_simd_poseidon_prove` | 1024 | 1024 | 100% | ✅ PASS |
| `test_poseidon_128_of_128` | 128 | 128 | 100% | ✅ PASS |

### ❌ Failing Tests (with padding rows):

| Test | Rows | Messages | Active % | Result |
|------|------|----------|----------|--------|
| `test_poseidon_127_of_128` | 128 | 127 | 99.2% | ❌ FAIL (ConstraintsNotSatisfied) |
| `test_poseidon_optimization_2_of_128` | 128 | 2 | 1.56% | ❌ FAIL (ConstraintsNotSatisfied) |

## Known Issue: ConstraintsNotSatisfied with Padding Rows

**STATUS**: ❌ **Any padding rows cause proof generation to fail**

### Symptoms:
- `prove()` returns `Err(ConstraintsNotSatisfied)`
- Occurs even with just 1 padding row (127/128 test)
- claimed_sum from gen_interaction_trace is non-zero (but this is expected, it should be balanced by constraints)

### What We've Tried:

1. ✅ **Fixed gen_interaction_trace** to use is_active selector as numerator
   - Before: `col_gen.write_frac(vec_row, denom1 - denom0, denom0 * denom1)`
   - After: `col_gen.write_frac(vec_row, is_active * (denom1 - denom0), denom0 * denom1)`
   - Result: Still fails

2. ✅ **Verified all constraints are masked by is_active**
   - Constraint 1: ✅ Masked
   - Constraint 2: ✅ Masked
   - Constraint 3: ✅ Masked
   - LogUp: ✅ Uses is_active multiplicity

3. ✅ **Analyzed circular boundary conditions**:
   - Row 127 (padding) → Row 0 (active, is_first=1): ✅ Transition disabled by `not_first=0`
   - Padding rows have all constraints = 0 due to is_active=0: ✅ Should be satisfied

4. ✅ **Checked trace generation**:
   - Padding rows correctly left as zeros
   - Active rows correctly computed with Poseidon permutation
   - Bit-reversal applied correctly

### Possible Root Causes (Unresolved):

1. **Circular trace evaluation issue**?
   - STARKs use circular traces where last row wraps to first
   - Padding rows at the end might interact incorrectly with first active row
   - But our analysis shows constraints should be satisfied...

2. **LogUp claimed_sum mismatch**?
   - claimed_sum from gen_interaction_trace is non-zero for both working and failing cases
   - Framework should balance this via constraints
   - But something in the prove() step fails to verify constraints

3. **PreprocessedColumnId with runtime parameters**?
   - `is_active_column_id(log_size, n_messages)` includes n_messages in ID
   - This is runtime-dependent, unlike `is_first_column_id(log_size)`
   - Might cause issues with framework's column tracking?

4. **Padding rows with all-zero state in LogUp**?
   - lookup_elements.combine([0,0,0,...]) might cause numerical issues
   - But is_active=0 should make numerator=0, so fraction should be 0/denom = 0
   - Shouldn't matter what denom is...

5. **Something fundamental about stwo framework**?
   - Maybe padding rows must have valid intermediate states, not zeros?
   - Maybe preprocessed columns can't be message-count dependent?
   - Maybe circular traces require special handling for inactive rows?

## Comparison with poseidon_u_components

Both implementations have the SAME issue with padding rows:

| Feature | poseidon_optimized | poseidon_u_components |
|---------|-------------------|----------------------|
| Architecture | Monolithic | Computing + Scheduler components |
| Reads from | lookup_data | trace columns (stark_appv2_safe pattern) |
| is_active handling | Preprocessed column | Preprocessed column |
| LogUp formula | Single combined | Two separate components |
| **100% utilization** | ✅ Works | ✅ Works |
| **With padding rows** | ❌ Fails | ❌ Fails |

This suggests the issue is **fundamental to how we're using is_active with padding rows**, not specific to the architecture.

## Next Steps for Team

1. **Investigate stwo framework internals**:
   - How does `prove()` validate constraints?
   - What exactly causes `ConstraintsNotSatisfied` error?
   - Are there examples of working padding/selector patterns in stwo?

2. **Check if PreProcessedColumnId can depend on runtime params**:
   - Is `is_active_column_id(log_size, n_messages)` valid?
   - Or must preprocessed columns be fully determined by log_size alone?

3. **Try alternative approaches**:
   - Option A: Always fill full table (no optimization), use is_active only for LogUp multiplicity
   - Option B: Don't use preprocessed column for is_active, compute it in constraints from row index
   - Option C: Use fixed-size blocks (e.g., always multiples of 16 rows) to avoid partial padding

4. **Compare with other STARKs implementations**:
   - How does Polygon Miden handle variable-length inputs?
   - How does Winterfell handle padding rows with selectors?

## Running Tests

```bash
# Passing tests (100% utilization)
cargo test --package stwo-examples poseidon_optimized::tests::test_poseidon_128_of_128 -- --nocapture

# Failing tests (with padding)
cargo test --package stwo-examples poseidon_optimized::tests::test_poseidon_127_of_128 -- --nocapture
cargo test --package stwo-examples poseidon_optimized::tests::test_poseidon_optimization_2_of_128 -- --nocapture
```

## Summary

**Simplified implementation achieved**:
- ✅ Cleaner code than poseidon_u_components
- ✅ All constraints properly masked by is_active
- ✅ gen_interaction_trace uses is_active selector
- ✅ Works perfectly with 100% utilization

**Optimization goal NOT achieved**:
- ❌ Padding rows cause ConstraintsNotSatisfied
- ❌ Cannot prove with fewer messages than table rows
- ❌ Root cause still unknown after extensive debugging

The implementation is ready for team review to identify the missing piece.
