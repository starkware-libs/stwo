---
name: finite-field-arithmetic
description: >
  M31/CM31/QM31 field arithmetic for STWO. Covers reduction logic, SIMD field
  implementations, extension field embedding, and batch inverse. Use when
  modifying code in core/fields/, prover/backend/simd/m31.rs, cm31.rs, qm31.rs,
  or reviewing any field operation changes.
---

# Finite Field Arithmetic

## Canonical Theory Sources

- `.agents/papers/llm/INDEX.llm.md` — notation normalization (`p` vs `q`, field tower naming)
- `.agents/papers/llm/Stwo_Whitepaper.llm.md` — base/extension field definitions
- `.agents/papers/llm/Circle_STARKs.llm.md` — circle-field assumptions used by FFT/FRI

## Field Tower

```
M31 (base field)
 |
 +-- CM31 = M31[i] / (i^2 + 1)        degree-2 extension
      |
      +-- QM31 = CM31[u] / (u^2 - 2 - i)   degree-2 extension of CM31
                                              degree-4 extension of M31
```

| Field | Elements | Modulus | Size |
|-------|----------|---------|------|
| M31 | u32 in [0, P) | P = 2^31 - 1 = 2147483647 | ~2^31 |
| CM31 | (M31, M31) = a + bi | P^2 | ~2^62 |
| QM31 | (CM31, CM31) = a + bu | P^4 | ~2^124 |

**Source**: `.agents/papers/llm/Stwo_Whitepaper.llm.md` — field preliminaries
(`s:fields`, `e:complex:extension`, `e:quartic:extension`)

## M31 Implementation

**File**: `crates/stwo/src/core/fields/m31.rs`

### Representation
```rust
#[repr(transparent)]
pub struct M31(pub u32);  // Invariant: 0 <= self.0 < P
```

### Key Constants
```rust
pub const P: u32 = 2147483647;        // 2^31 - 1
pub const MODULUS_BITS: u32 = 31;
```

### Reduction

**partial_reduce(val: u32)** — for val in [0, 2P):
```rust
Self(val.checked_sub(P).unwrap_or(val))
```

**reduce(val: u64)** — for val in [0, P^2):
```rust
Self((((((val >> 31) + val + 1) >> 31) + val) & (P as u64)) as u32)
```

This exploits the Mersenne prime structure: for p = 2^31 - 1,
reduction is just: val mod p = (val >> 31) + (val & p), with a
conditional subtraction.

### Arithmetic
- **Add**: `partial_reduce(self.0 + rhs.0)` — works because self.0 + rhs.0 < 2P
- **Sub**: `partial_reduce(self.0 + P - rhs.0)` — avoids underflow
- **Neg**: `partial_reduce(P - self.0)`
- **Mul**: `reduce((self.0 as u64) * (rhs.0 as u64))` — product fits in u64
- **Inverse**: Fermat's little theorem: a^{-1} = a^{P-2} = a^{2147483645}

### Common Bugs to Watch For
1. **Overflow in add chain**: M31 values are < P < 2^31, so adding two is < 2^32.
   Safe in u32. But chaining 3+ additions without reduction overflows.
2. **Reduction after multiply**: The product of two M31 values fits in u62.
   Must use `reduce()` (u64 reduction), not `partial_reduce()` (u32).
3. **Zero representation**: Only 0 is valid, not P. The `partial_reduce` ensures this.

## CM31 Implementation

**File**: `crates/stwo/src/core/fields/cm31.rs`

### Representation
```rust
pub struct CM31(pub M31, pub M31);  // a + bi
```

### Arithmetic
- **Mul**: (a+bi)(c+di) = (ac-bd) + (ad+bc)i — standard complex multiplication
- **Inverse**: (a+bi)^{-1} = (a-bi)/(a^2+b^2) — conjugate divided by norm

### Irreducible Polynomial
x^2 + 1 is irreducible over M31 because P = 3 mod 4, so -1 is not a
quadratic residue mod P.

## QM31 Implementation

**File**: `crates/stwo/src/core/fields/qm31.rs`

### Representation
```rust
pub struct QM31(pub CM31, pub CM31);  // a + bu, where u^2 = 2+i
pub type SecureField = QM31;
```

### Key Constant
```rust
pub const R: CM31 = CM31::from_u32_unchecked(2, 1);  // u^2 = 2+i
```

### Arithmetic
- **Mul**: (a+bu)(c+du) = (ac + R*bd) + (ad+bc)u
  Where R = 2+i is the "non-residue" defining the extension.

### Irreducible Polynomial
x^2 - (2+i) is irreducible over CM31. The choice R = 2+i ensures this.

### Coordinate Decomposition
```rust
pub fn from_partial_evals(evals: [Self; 4]) -> Self
```
Reconstructs QM31 from 4 base-field evaluations: a + b*i + c*u + d*iu.

## SIMD Implementations

**Files**:
- `crates/stwo/src/prover/backend/simd/m31.rs` — `PackedM31` (16 lanes)
- `crates/stwo/src/prover/backend/simd/cm31.rs` — `PackedCM31`
- `crates/stwo/src/prover/backend/simd/qm31.rs` — `PackedQM31`

### PackedM31
Processes 16 M31 elements in parallel using platform SIMD:
- x86_64: AVX512 (u32x16 native) or AVX2 (2x u32x8)
- AArch64: NEON (4x u32x4)
- WASM: SIMD128 (4x u32x4)

### Critical Safety: `from_simd_unchecked`
```rust
pub unsafe fn from_simd_unchecked(v: u32x16) -> Self
```
Creates a PackedM31 WITHOUT checking that values are in [0, P).
Used only when the caller can guarantee validity (e.g., after reduction,
or when storing values known to be < P by construction).

**Every call site must be audited for the guarantee.**

## Batch Inverse

**File**: `crates/stwo/src/core/fields/mod.rs` — `batch_inverse_classic()`, `batch_inverse_in_place()`

Montgomery's trick for batch inversion:
1. Compute cumulative products
2. Invert the final product (single inversion)
3. Unwind to get individual inverses

The implementation uses WIDTH=4 interleaved cumulative products for
better instruction pipelining.

## Security Invariants

INVARIANT-FIELD-1: All M31 values must be in [0, P). The `partial_reduce`
and `reduce` functions ensure this. Bypass via `from_u32_unchecked` must
only use values known to be < P.

INVARIANT-FIELD-2: The extension field irreducible polynomials must actually
be irreducible. x^2+1 is irreducible over M31 because P = 3 mod 4.
x^2-(2+i) is irreducible over CM31.

INVARIANT-FIELD-3: Field inverse must satisfy: a * a.inverse() = 1 for all
nonzero a. The Fermat's little theorem approach guarantees this.

INVARIANT-FIELD-4: SIMD field operations must produce identical results to
scalar operations for all inputs.

## Review Checklist

Before approving changes to field arithmetic:
- [ ] Reduction is correct for all input ranges
- [ ] No integer overflow in intermediate computations
- [ ] SIMD implementation matches scalar semantics exactly
- [ ] `from_simd_unchecked` call sites have documented value-range proofs
- [ ] Extension field multiplication uses the correct irreducible polynomial
- [ ] Inverse is tested: a * a.inverse() == 1

## Forbidden Actions

In this domain, agents must NEVER:
- Change the prime P or its representation
- Modify the reduction algorithm without proving correctness for all inputs
- Add new `from_u32_unchecked` calls without proving the value is < P
- Change the extension field irreducible polynomials (x^2+1, x^2-2-i)
- Modify SIMD field ops without verifying against scalar reference
