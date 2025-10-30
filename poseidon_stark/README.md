# Poseidon2 STARK Proof System

This workspace demonstrates a complete STARK proof system for the **Poseidon2 hash function**, a cryptographic hash used in Starknet and other zero-knowledge systems.

## Overview

Poseidon2 is a cryptographic hash function designed for efficient zero-knowledge proof systems. This implementation:
- Uses a 16-element state
- Processes 8 hash instances per row (`N_INSTANCES_PER_ROW = 2^3`)
- Implements full and partial rounds with MDS matrices
- Uses LogUp for lookup arguments between initial and final states

**Reference**: [Poseidon2 Paper (eprint.iacr.org/2023/323)](https://eprint.iacr.org/2023/323.pdf)

## Workspace Structure

```
poseidon_stark/
├── circuit/          # Core Poseidon2 AIR implementation (library)
│   └── src/lib.rs   # Constraints, trace generation, LogUp
├── prover/          # Proof generation (binary)
│   └── src/main.rs  # CLI prover with JSON output
└── verifier/        # Proof verification (binary)
    └── src/main.rs  # CLI verifier reading JSON proof
```

## How It Works

### Hash Function Structure

Poseidon2 operates on a 16-element state through:

1. **Initial State**: 16 field elements as input
2. **First Full Rounds** (4 rounds):
   - Add round constants
   - Apply external MDS matrix
   - Apply S-box (x^5) to all elements
3. **Partial Rounds** (14 rounds):
   - Add round constant
   - Apply internal MDS matrix
   - Apply S-box (x^5) to first element only
4. **Second Full Rounds** (4 rounds):
   - Same as first full rounds
5. **Final State**: 16 field elements as output

### STARK Constraints

The AIR enforces:
- **Intra-row constraints**: Each round's computation is correct (S-box, MDS matrix)
- **LogUp lookups**: Initial state and final state are linked via lookup arguments
- **Multiple instances**: 8 hash instances computed per row for efficiency

### LogUp Verification

The prover provides two lookup entries per instance:
- Positive entry for initial state: `+1 / (α - initial_state)`
- Negative entry for final state: `-1 / (α - final_state)`

These ensure the hash computation is correct and the final state matches the initial state's hash.

## Running the Prover

Generate a STARK proof for Poseidon2 hash computations:

```bash
cd poseidon_stark/prover
cargo run --release -- <log_n_instances>
```

**Arguments**:
- `log_n_instances`: Log₂ of number of hash instances (default: 10 for 1024 instances)
  - Must be ≥ 7 (minimum 128 instances due to SIMD lanes)
  - Example: 10 → 1024 instances = 128 rows
- `--dump-trace`: Optional flag to dump trace to `trace_dump.txt`

**Outputs**:
- `proof.json` - Complete STARK proof (serialized)
- `proof_metadata.json` - Proof parameters and claimed sums
- `trace_dump.txt` - Trace dump (only if `--dump-trace` specified)

**Examples**:
```bash
# Generate proof for 4096 hash instances
cargo run --release -- 12

# Generate proof with trace dump for debugging
cargo run --release -- 7 --dump-trace

# Works in any order
cargo run --release -- --dump-trace 10
```

### Prover Output

```
=== STARK Prover - Poseidon2 Hash Function ===

Configuration:
  Log instances: 10
  Total instances: 1024
  Instances per row: 8
  Log rows: 7
  Rows: 128

Step 1: Committing preprocessed (empty)...
Step 2: Generating Poseidon trace...
  ✓ Trace generated: 128 rows
Step 3: Drawing lookup elements...
Step 4: Generating LogUp interaction trace...
  Claimed sum: QM31(...)
Step 5: Creating Poseidon component...
Step 6: Generating STARK proof...
✓ Proof generated!
  Commitments: 3
  Proof size: 45678 bytes

✓ Proof saved to proof.json
✓ Proof metadata saved to proof_metadata.json
✓ Prover completed successfully!
```

### Trace Dump Format

When using `--dump-trace`, the trace dump shows the complete computation flow for the first hash instance:

```
=== INSTANCE 0 (First Hash) ===

Initial State (cols 0-15):        # Input: 16 field elements
After Full Round 1 (cols 16-31):  # After mixing + S-box
After Full Round 2 (cols 32-47)
After Full Round 3 (cols 48-63)
After Full Round 4 (cols 64-79)
Partial Rounds 1-14 (cols 80-93): # Only first element (efficiency!)
After Full Round 5 (cols 94-109)
After Full Round 6 (cols 110-125)
After Full Round 7 (cols 126-141)
After Full Round 8 (cols 142-157): # Final output: 16 field elements

=== Summary for all 8 instances ===
Instance   Initial[0]   Final[0]
0          0            462565134
1          1            2107930580
...
```

Each instance has **158 columns** (16×9 full round states + 14 partial round states).

**Use cases**:
- Debug hash computation
- Verify round-by-round transformations
- Understand MDS matrix effects
- Compare with reference implementation

## Running the Verifier

Verify a STARK proof:

```bash
cd poseidon_stark/verifier
cargo run --release
```

The verifier:
1. Loads `proof.json` and `proof_metadata.json`
2. Reconstructs the Poseidon component with claimed sums
3. Verifies the proof cryptographically
4. Reports success or failure

**Example Output**:
```
=== STARK Verifier - Poseidon2 Hash Function ===

Loading proof from proof.json...
✓ Proof loaded

Loading metadata from proof_metadata.json...
✓ Metadata loaded
  Hash function: Poseidon2
  Instances: 1024
  Log rows: 7
  Claimed sum: QM31(...)

Creating Poseidon component for verification...
✓ Component created

Verifying STARK proof...
✓✓✓ PROOF VERIFICATION SUCCESSFUL! ✓✓✓

The proof is cryptographically valid.
Poseidon2 hash computations verified: 1024 instances
```

## Technical Details

### Constants

- `N_STATE = 16` - Hash state size
- `N_INSTANCES_PER_ROW = 8` - Hash instances per row
- `N_LOG_INSTANCES_PER_ROW = 3` - Log₂(instances per row)
- `N_PARTIAL_ROUNDS = 14` - Number of partial rounds
- `N_HALF_FULL_ROUNDS = 4` - Full rounds at start/end (8 total)

### Trace Columns

Each hash instance uses:
- 16 columns for initial state
- 64 columns for full round states (4 rounds × 16 elements)
- 14 columns for partial round first elements
- Total: **94 columns per instance**
- **752 columns total** (8 instances × 94 columns)

### Security Note

**WARNING**: This implementation uses placeholder round constants:
```rust
const EXTERNAL_ROUND_CONSTS: [[BaseField; N_STATE]; 8] =
    [[BaseField::from_u32_unchecked(1234); N_STATE]; 8];
const INTERNAL_ROUND_CONSTS: [BaseField; N_PARTIAL_ROUNDS] =
    [BaseField::from_u32_unchecked(1234); N_PARTIAL_ROUNDS];
```

**For production use**, replace with actual Poseidon2 round constants from the specification.

## Performance Scaling

| log_n_instances | Instances | Rows | Trace Columns | Proof Size (est.) |
|-----------------|-----------|------|---------------|-------------------|
| 5               | 32        | 4    | 752           | ~20 KB            |
| 8               | 256       | 32   | 752           | ~30 KB            |
| 10              | 1,024     | 128  | 752           | ~45 KB            |
| 12              | 4,096     | 512  | 752           | ~70 KB            |
| 15              | 32,768    | 4,096| 752           | ~120 KB           |

*Note: Proof size grows logarithmically with instance count due to FRI.*

## Comparison with Other Examples

### vs. docs_component_proper
- **docs_component_proper**: Simple x^5+1 computation with dual components (Computing + Scheduling)
- **poseidon_stark**: Full cryptographic hash with 16-element state and complex round structure

### vs. stark_app_wide
- **stark_app_wide**: Fibonacci with N configurable columns (horizontal layout)
- **poseidon_stark**: Fixed 16-state hash, 8 instances per row for efficiency

## Files Generated

After running the prover:

**proof.json**:
```json
{
  "commitments": [...],
  "lookup_values": [...],
  "fri_proof": {...}
}
```

**proof_metadata.json**:
```json
{
  "log_n_instances": 10,
  "n_instances": 1024,
  "log_n_rows": 7,
  "n_rows": 128,
  "instances_per_row": 8,
  "claimed_sum": {
    "a": [a0, a1],
    "b": [b0, b1]
  },
  "commitments_count": 3,
  "proof_size_bytes": 45678,
  "hash_function": "Poseidon2"
}
```

## References

- [Poseidon2 Paper](https://eprint.iacr.org/2023/323.pdf) - Hash function specification
- [stwo Documentation](https://github.com/starkware-libs/stwo) - STARK prover library
- [Starknet](https://www.starknet.io/) - Production usage of Poseidon
