# STARK App Wide - Horizontal Fibonacci

Wide Fibonacci implementation using **HORIZONTAL structure** instead of vertical.

## Key Difference from stark_appv3

| Feature | stark_appv3 (Vertical) | stark_app_wide (Horizontal) |
|---------|------------------------|------------------------------|
| **Structure** | 3 columns, many rows | 50 columns, few rows |
| **Each row** | One step: [f(n-2), f(n-1), f(n)] | Complete sequence: [f(0), f(1), ..., f(49)] |
| **Constraint direction** | Within row (intra-row) | Across columns (chained) |
| **Continuity** | No enforcement between rows | Enforced across columns |
| **Use case** | Proving specific f(N) | Proving entire sequence |

## Structure

```
stark_appv3 (VERTICAL):
       Col A    Col B    Col C
Row 0: f(0)     f(1)     f(2)    ← One step
Row 1: f(1)     f(2)     f(3)    ← Next step
Row 2: f(2)     f(3)     f(4)    ← Next step
...

stark_app_wide (HORIZONTAL):
       Col 0   Col 1   Col 2   ...  Col 49
Row 0: f(0)    f(1)    f(2)    ...  f(49)   ← Complete sequence!
Row 1: f(0)    f(1)    f(2)    ...  f(49)   ← Another instance
Row 2: f(0)    f(1)    f(2)    ...  f(49)   ← Another instance
```

## How Constraints Work

### Chained Constraints (Wide Fibonacci)
```rust
fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
    let mut a = eval.next_trace_mask(); // Col 0
    let mut b = eval.next_trace_mask(); // Col 1
    for _ in 2..N {
        let c = eval.next_trace_mask(); // Col 2, 3, 4...
        eval.add_constraint(c.clone() - (a + b));
        a = b;  // ← Shift across columns!
        b = c;
    }
    eval
}
```

This creates a **chain of constraints across columns**:
- Col2 = Col0 + Col1
- Col3 = Col1 + Col2
- Col4 = Col2 + Col3
- ...

**Enforces continuity ACROSS THE ROW!** ✓

## Usage

### Build
```bash
cargo build
```

### Prover

```bash
# Default: 8 rows, f(0)=0, f(1)=1
cargo run --bin prover

# Custom: 4 rows
cargo run --bin prover -- 2

# Custom initial values
cargo run --bin prover -- 3 1 1

# With trace dump
cargo run --bin prover -- 3 --dump-trace
```

**Output:**
```
=== STARK Prover - Wide Fibonacci ===
Horizontal structure: 50 columns per row

Configuration:
  Rows: 8 (2^3)
  Columns: 50 (each row = complete Fibonacci sequence)
  Initial values: f(0)=0, f(1)=1

Generating wide Fibonacci trace...
✓ Trace generated
  8 rows × 50 columns = 400 total values
  Last value: f(49) = 1848850790 (mod 2^31-1)
```

### Verifier

```bash
cargo run --bin verifier
```

### Testing

```bash
cargo test --package circuit
```

## Trace Dump Example

```
=== Wide Fibonacci Trace Dump ===

Structure: 50 columns (HORIZONTAL)
Each row contains: f(0) to f(49)
Total rows: 8

Row      f(0)        f(1)        f(2)        f(3)        ...
--------------------------------------------------------------------------------
0        0           1           1           2           ...  1848850790
1        0           1           1           2           ...  1848850790
2        0           1           1           2           ...  1848850790
...
```

**Notice:** All rows are identical (same Fibonacci sequence)!

## Why Multiple Rows?

Each row is an independent **instance** of the Fibonacci sequence.

In this simple example, all rows are identical. But in more complex scenarios (like stwo's wide_fibonacci), you could have:
- Row 0: Fibonacci starting from (0, 1)
- Row 1: Fibonacci starting from (1, 2)
- Row 2: Fibonacci starting from (2, 3)

This allows **batch proving** - proving multiple sequences in parallel!

## Comparison: When to Use Each?

### Use VERTICAL (stark_appv3) when:
✓ You want to prove a specific f(N)
✓ N is variable (user specifies)
✓ Zero-padding optimization matters
✓ Simple, intuitive structure

### Use HORIZONTAL (stark_app_wide) when:
✓ You want to prove entire sequence
✓ Batch proving multiple instances
✓ Fixed sequence length
✓ Continuity enforcement across data

## Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_wide_fibonacci_trace
```

Tests verify:
- Correct Fibonacci values in each row
- Constraint satisfaction
- All rows contain same sequence
- Different initial values work

## Limitations

- **No Boundary Constraints**: Doesn't enforce f(0)=0, f(1)=1 (could start with any values)
- **Fixed Length**: Sequence length (50) is hardcoded at compile time
- **Proof Serialization**: Not yet implemented

## Future Improvements

- [ ] Add boundary constraints
- [ ] Variable sequence length
- [ ] Batch proving with different initial values
- [ ] Proof serialization
