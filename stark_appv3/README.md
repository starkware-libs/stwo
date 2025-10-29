# STARK App v3

Intuitive STARK proof generation for specific Fibonacci numbers with zero padding.

## Key Features

- **Intuitive Interface**: Just specify which Fibonacci number you want to prove (e.g., `f(50)`)
- **Automatic Optimization**: Automatically selects minimum trace size (next power of 2)
- **Zero Padding**: Efficiently pads unused rows with zeros (constraint: `0 = 0 + 0`)
- **Clear Output**: Shows exactly what was computed and proved

## Architecture

- **circuit** (library) - Fibonacci circuit with zero-padding support
- **prover** (binary) - Generates STARK proofs for specific Fibonacci numbers
- **verifier** (binary) - Verifies STARK proofs

## Usage

### Prover

Compute and prove a specific Fibonacci number:

```bash
# Prove f(50) with default values (f(0)=0, f(1)=1)
cargo run --bin prover -- 50

# Prove f(100) with custom initial values
cargo run --bin prover -- 100 1 1

# Prove f(10)
cargo run --bin prover -- 10

# Dump trace to file for inspection
cargo run --bin prover -- 10 --dump-trace
```

**Output:**
```
=== STARK Prover v3 ===
Compute and prove specific Fibonacci number

Target: Compute f(50)
Initial values: f(0)=0, f(1)=1

Generating Fibonacci trace...
✓ Trace generated
  Total rows: 64 (2^6)
  Computed rows: 49 (f(0) to f(50))
  Padding rows: 15 (filled with zeros)

✓ Target value: f(50) = 2044720959 (mod 2^31-1)

Generating STARK proof...
✓ Proof generated!

Summary:
  Proved: f(50) = 2044720959
  Trace size: 64 rows (with 15 padding)
```

### Verifier

```bash
cargo run --bin verifier
```

**Output:**
```
=== STARK Verifier v3 ===

Reading proof metadata...
✓ Metadata loaded

Proof claims:
  f(50) = 2044720959 (mod 2^31-1)
  Initial values: f(0)=0, f(1)=1
  Trace size: 64 rows (49 computed, 15 padding)

Verifying proof...
✓ Proof verified successfully!

Verification result:
  ✓ f(50) = 2044720959 is CORRECT
```

### Trace Inspection

Dump the complete trace to a file for inspection:

```bash
cargo run --bin prover -- 10 --dump-trace
```

This creates `trace_dump.txt` showing all rows and columns:

```
=== Fibonacci Trace Dump ===

Target: f(10)
Total rows: 16
Computed rows: 9 (f(0) to f(10))
Padding rows: 7

Structure:
  Column A: f(n-2)
  Column B: f(n-1)
  Column C: f(n)

Row      Col A (f(n-2))  Col B (f(n-1))  Col C (f(n))
------------------------------------------------------------
0        0               1               1
1        1               1               2
2        1               2               3
3        2               3               5
4        3               5               8
5        5               8               13
6        8               13              21
7        13              21              34
8        21              34              55
9        0               0               0               (padding)
10       0               0               0               (padding)
...
```

**Key observations:**
- Row 8 contains f(10) = 55 in Column C
- Rows 9-15 are zero-padded
- Each row shows the transition: `f(n) = f(n-1) + f(n-2)`

## How It Works

### Automatic Trace Sizing

The circuit automatically calculates the minimum trace size needed:

| Target | Min Rows Needed | Actual Rows (2^n) | Padding |
|--------|----------------|-------------------|---------|
| f(10)  | 9              | 16 (2^4)          | 7       |
| f(50)  | 49             | 64 (2^6)          | 15      |
| f(100) | 99             | 128 (2^7)         | 29      |
| f(200) | 199            | 256 (2^8)         | 57      |

### Zero Padding

Since our constraint is **intra-row** (checks values within the same row):
```
f(n) = f(n-1) + f(n-2)
```

Padding with zeros satisfies the constraint:
```
Row 48: [f(48), f(49), f(50)]  ✓ f(50) = f(48) + f(49)
Row 49: [0,     0,     0]      ✓ 0 = 0 + 0  (padding)
Row 50: [0,     0,     0]      ✓ 0 = 0 + 0  (padding)
...
```

This is more efficient than computing unnecessary Fibonacci values!

## Testing

Run circuit tests:
```bash
cargo test --package circuit
```

Tests include:
- Automatic log_size calculation
- Fibonacci values for specific indices
- Constraint validation with padding
- Zero-padding correctness

## Comparison with v2

| Feature | v2 | v3 |
|---------|----|----|
| Interface | `cargo run --bin prover -- 8 1 1` (log_size) | `cargo run --bin prover -- 50` (target index) |
| User specifies | Power of 2 (technical) | Specific Fibonacci number (intuitive) |
| Padding | Computes all rows | Zero-pads unused rows |
| Output | Last computed value | Specific target value |
| Efficiency | May compute extra values | Only computes what's needed |

## Limitations

- **Proof Serialization**: Full STARK proof serialization not yet implemented
- **M31 Field**: All computations are modulo 2^31-1 (may wrap around for large indices)

## Future Improvements

- [ ] Full proof serialization/deserialization
- [ ] Batch proving for multiple Fibonacci numbers
- [ ] Support for other sequences beyond Fibonacci
- [ ] Proof size benchmarking
