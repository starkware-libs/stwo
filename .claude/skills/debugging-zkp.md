---
name: debugging-zkp
description: >
  Debugging failing proofs and constraint violations in STWO. Covers failure
  mode diagnosis by symptom, FRI error triage, channel desync detection, and
  distilled theory cross-references. Use when prove() fails, verify() rejects a valid
  proof, constraints don't hold on a trace, or logup sums don't balance.
---

# Debugging ZK Proofs in STWO

## Canonical Theory Sources

- `.agents/papers/llm/INDEX.llm.md` — notation/source map
- `.agents/papers/llm/Circle_STARKs.llm.md` — FFT/FRI/AIR math anchors
- `.agents/papers/llm/Stwo_Whitepaper.llm.md` — STWO protocol and soundness anchors

## Common Failure Modes

### 1. Constraint Evaluation Mismatch

**Symptom**: `VerificationError::OodsNotMatching` or proof generation fails.

**Diagnosis**:
```rust
// Use AssertEvaluator to check constraints row-by-row
use stwo_constraint_framework::assert_constraints_on_trace;
assert_constraints_on_trace(&component, &trace);
```

If this passes but proof fails:
- **Degree mismatch**: The actual constraint degree exceeds the declared degree
- **Use InfoEvaluator** to check: it reports the maximum constraint degree

### 2. LogUp Sum Imbalance

**Symptom**: Proof fails with interaction trace issues.

**Diagnosis**:
```rust
// Use RelationTracker to find which relation is imbalanced
use stwo_constraint_framework::relation_tracker;
```

Common causes:
- Table entries don't match access entries
- Multiplicity mismatch
- Missing logup entry in one direction

### 3. FRI Verification Failure

**Symptom**: `FriVerificationError` variants.

**Diagnosis by error**:

| Error | Cause | Check |
|-------|-------|-------|
| `InvalidNumFriLayers` | Wrong number of folding layers | Check log_degree_bound vs config |
| `FirstLayerEvaluationsInvalid` | Circle fold mismatch | Check twiddle factors |
| `FirstLayerCommitmentInvalid` | Merkle proof bad | Check commitment domain size |
| `InnerLayerEvaluationsInvalid` | Line fold mismatch | Check folding alpha |
| `InnerLayerCommitmentInvalid` | Merkle proof bad | Check Merkle tree construction |
| `LastLayerDegreeInvalid` | Final poly too large | Check degree bound config |
| `LastLayerEvaluationsInvalid` | Final poly wrong | Check polynomial evaluation |

### 4. OODS Evaluation Mismatch

**Symptom**: Composition polynomial OODS eval doesn't match.

**Diagnosis**:
- Check `extract_composition_oods_eval` in `core/proof.rs`
- Verify the composition polynomial split (COMPOSITION_LOG_SPLIT)
- Verify OODS point sampling uses correct channel state

### 5. Merkle Decommitment Failure

**Symptom**: `MerkleVerificationError`

**Diagnosis**:
- Check that prover and verifier use the same hash function
- Check that column ordering in the tree matches
- Check that the lifting domain size is consistent

## Debugging Workflow

### Step 1: Isolate the Layer

```
Trace generation → Commitment → Constraint eval → Composition → FRI → PoW → Queries
```

Determine which step fails by progressively checking:
1. Do constraints hold on the raw trace? (AssertEvaluator)
2. Does the composition polynomial have the expected degree?
3. Does FRI commit succeed?
4. Does FRI decommit succeed?

### Step 2: Check Channel Consistency

The most subtle bugs come from prover-verifier channel desync:
- Prover mixes something the verifier doesn't (or vice versa)
- Different ordering of mix operations
- Different data format in mix

**Use logging channel**: `prover/channel/logging_channel.rs` wraps a channel
with tracing of all mix/draw operations.

### Step 3: Binary Search

For constraint failures, binary search the trace rows:
```rust
for row in 0..trace_len {
    if !constraint_holds_at_row(row) {
        // Found the failing row
    }
}
```

### Step 4: Check Paper Reference

Many subtle bugs come from incorrect mathematical operations:
- Wrong twiddle factor sign
- Off-by-one in domain size
- Incorrect coset offset
- Missing conjugate in circle fold

Load the relevant math skill and cross-reference with the distilled files.

## Useful Code Locations

| Need | File | Function |
|------|------|----------|
| Assert constraints | `constraint-framework/src/prover/assert.rs` | `AssertEvaluator` |
| Track logup | `constraint-framework/src/prover/relation_tracker.rs` | `RelationTracker` |
| Log channel ops | `prover/channel/logging_channel.rs` | `LoggingChannel` |
| Check constraint degree | `constraint-framework/src/info.rs` | `InfoEvaluator` |
| Evaluate at point | `constraint-framework/src/point.rs` | `PointEvaluator` |

## Distilled References for Debugging

| Component | Distilled File | Anchor |
|-----------|----------------|--------|
| Vanishing polynomial derivative | `Circle_STARKs.llm.md` | Source Anchor Map -> vanishing/quotients |
| FRI fold formula | `Circle_STARKs.llm.md` | `prot:IOP:proximity`, `e:FRI:g:even`, `e:FRI:g:odd` |
| Composition decomposition | `Circle_STARKs.llm.md` | `lem:quotient:decomposition`, `e:quotient:decomposition` |
| DEEP quotient | `Circle_STARKs.llm.md` | `prop:deep:quotients` |
| Constraint polynomial model | `Circle_STARKs.llm.md` | `e:overall:identity` |
| Circle FFT identities | `Circle_STARKs.llm.md` | `def:FFT:basis`, `thm:FFT` |
