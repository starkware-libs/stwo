# Dual Component STARK - Computing & Scheduling

Demonstrates **proper component composition pattern** with LogUp.

## Components

### Computing Component (The "Worker")
- **Generates** inputs: `[10, 10, 10, ...]`
- **Computes** outputs: `[10^5+1, 10^5+1, ...] = [100001, 100001, ...]`
- **Has constraints** to verify: `output = input^5 + 1`
- **Provides** verified (input, output) pairs via LogUp

### Scheduling Component (The "Requester")
- **Does NOT compute** anything
- **Copies** input/output from Computing
- **No constraints** (only LogUp)
- **Requests** (input, output) pairs via LogUp

### LogUp Verification
- Ensures both components use **identical data**
- Claimed sums **must cancel**: `scheduling_sum + computing_sum = 0`
- This proves Scheduling's data matches Computing's verified data

## Why This Pattern?

✓ **Computing** has complex logic → keep it in one place
✓ **Scheduling** just needs results → copy via LogUp
✓ Same pattern as `scheduler_poseidon` (production code)

## Usage

### Build

```bash
cargo build
```

### Prover

```bash
# Default: log_size = 5 (32 rows)
cargo run --bin prover

# Custom log_size
cargo run --bin prover -- 6  # 64 rows
cargo run --bin prover -- 7  # 128 rows
```

**Output:**
```
=== STARK Prover - Dual Component (Computing + Scheduling) ===

Configuration:
  Log size: 5
  Rows: 32

Step 1: Committing preprocessed (empty)...
Step 2: Computing generates inputs and computes x^5+1...
  ✓ Computing generated 32 rows
Step 3: Scheduling copies data from Computing...
  ✓ Scheduling copied data

Step 4: Committing trace columns...
Step 5: Drawing lookup elements...
Step 6: Generating LogUp interaction columns...
  Scheduling claimed sum: ...
  Computing claimed sum: ...
  Sum (should be zero): QM31(0, 0, 0, 0)

✓ Proof generated!
  Proof size: 2840 bytes

✓ Proof saved to proof.json
✓ Proof metadata saved to proof_metadata.json
```

**Generated files:**
- `proof.json` - Full STARK proof (serialized)
- `proof_metadata.json` - Parameters and claimed sums

### Verifier

```bash
cargo run --bin verifier
```

**Output:**
```
=== STARK Verifier - Dual Component ===

Reading proof metadata...
✓ Metadata loaded

Proof claims:
  Rows: 32
  Scheduling claimed sum: ...
  Computing claimed sum: ...
  Sum check: QM31(0, 0, 0, 0)

✓ LogUp sums cancel (valid)

Loading proof from file...
✓ Proof loaded from proof.json
  Proof size: 2840 bytes

Verifying proof...
✓ Proof verified successfully!

Verification result:
  ✓ Computing component verified (x^5+1 constraints)
  ✓ Scheduling component verified (LogUp copy)
  ✓ LogUp sums cancel → data matches
  ✓ Proof loaded from file and verified
```

## How It Works

### 1. Computing Component

**Constraints:**
```rust
// Constraint 1: intermediate = input^3
eval.add_constraint(intermediate - input * input * input);

// Constraint 2: output = input^5 + 1 = intermediate * input^2 + 1
eval.add_constraint(output - intermediate * input * input - 1);
```

**LogUp (Provider):**
```rust
eval.add_to_relation(RelationEntry::new(&lookup_elements, -1, &[input]));
eval.add_to_relation(RelationEntry::new(&lookup_elements, +1, &[output]));
```

### 2. Scheduling Component

**No Constraints!** Only LogUp:

**LogUp (Consumer):**
```rust
eval.add_to_relation(RelationEntry::new(&lookup_elements, +1, &[input]));
eval.add_to_relation(RelationEntry::new(&lookup_elements, -1, &[output]));
```

### 3. LogUp Magic

**Computing provides:** `-1/hash(input) + 1/hash(output)`
**Scheduling requests:** `+1/hash(input) - 1/hash(output)`

**When summed:**
`(-1 + 1)/hash(input) + (1 - 1)/hash(output) = 0` ✓

This **proves** both components use the **same (input, output) pairs**!

## Comparison to Fibonacci

| Feature | Fibonacci | Dual Component |
|---------|-----------|----------------|
| Components | 1 | 2 (Computing + Scheduling) |
| Constraints | Intra-row (c = a+b) | Computing: x^5+1 verification |
| LogUp | ❌ | ✅ Data sharing |
| Complexity | Simple | Production pattern |
| Use case | Learning | Real applications |

## Key Concepts

### Component Cooperation
- **Computing** does heavy lifting (math verification)
- **Scheduling** just copies results (no math)
- **LogUp** ensures they match (cryptographic guarantee)

### Why Split Components?
In real applications (like Starknet):
- VM component needs Poseidon hashes
- Poseidon component computes + verifies hashes
- VM just copies results via LogUp
- **No need to duplicate Poseidon constraints in VM!**

## Testing

```bash
cargo test --package dual_component_circuit
```

## Limitations

- **No Proof Streaming**: Entire proof in memory
- **Fixed Input**: Always uses `input = 10` (educational)
- **No Batch Processing**: Single instance per row

## Future Improvements

- [ ] Add configurable input values
- [ ] Batch multiple computations per row
- [ ] Add more complex constraints
- [ ] Demonstrate with real-world use case (Poseidon)
