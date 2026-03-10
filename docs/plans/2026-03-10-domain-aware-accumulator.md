# Domain-Aware DomainEvaluationAccumulator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `DomainEvaluationAccumulator` track the `CircleDomain` of each sub-accumulation so that constraint quotients evaluated on non-canonical domains (e.g. D_sub) can be stored directly, with the canonical-domain conversion deferred to `finalize()`.

**Architecture:** Change `sub_accumulations` from `Vec<Option<SecureColumnByCoords<B>>>` (one slot per log_size, implicitly canonical) to `Vec<Vec<(CircleDomain, SecureColumnByCoords<B>)>>` (multiple domain-tagged entries per log_size). In `finalize()`, convert any non-canonical entries to canonical via IFFT+FFT before passing to `lift_and_accumulate`. The `columns()` method now takes a `CircleDomain` instead of just `log_size`. Callers pass the domain they're evaluating on.

**Tech Stack:** Rust, stwo prover internals

---

### Task 1: Change `sub_accumulations` storage to include domain

**Files:**
- Modify: `crates/stwo/src/prover/air/accumulation.rs`

**Step 1: Update the struct**

Change `DomainEvaluationAccumulator`:

```rust
// Before:
sub_accumulations: Vec<Option<SecureColumnByCoords<B>>>,

// After:
sub_accumulations: Vec<Vec<(CircleDomain, SecureColumnByCoords<B>)>>,
```

The outer Vec is indexed by log_size (0..=max_log_size). Each inner Vec holds (domain, column) pairs — typically 0 or 1 entries, but can have multiple if different components use different domains for the same log_size.

**Step 2: Update `new()`**

```rust
sub_accumulations: (0..(max_log_size + 1)).map(|_| Vec::new()).collect(),
```

**Step 3: Update `log_size()`**

No change needed — it's based on `sub_accumulations.len()`.

**Step 4: Build and fix any compilation errors in this file**

Run: `cargo build --features prover -p stwo`
Expected: Errors in `columns()` and `finalize()` — fix in next tasks.

**Step 5: Commit**

```
feat: change sub_accumulations storage to Vec<Vec<(CircleDomain, SecureColumnByCoords)>>
```

---

### Task 2: Update `columns()` to accept `CircleDomain`

**Files:**
- Modify: `crates/stwo/src/prover/air/accumulation.rs`

**Step 1: Change `columns()` signature and body**

```rust
// Before:
pub fn columns<const N: usize>(
    &mut self,
    n_cols_per_size: [(u32, usize); N],
) -> [ColumnAccumulator<'_, B>; N] {

// After:
pub fn columns<const N: usize>(
    &mut self,
    n_cols_per_domain: [(CircleDomain, usize); N],
) -> [ColumnAccumulator<'_, B>; N] {
```

The body needs to:
1. For each `(domain, n_cols)`, look up `self.sub_accumulations[domain.log_size()]`.
2. Find or create a `(domain, SecureColumnByCoords)` entry in that Vec.
3. Return a `ColumnAccumulator` referencing the column.

Note: Since we can no longer use `get_disjoint_mut` (entries are inside inner Vecs), we need a different approach. The simplest: iterate `n_cols_per_domain`, for each one push a new entry into the appropriate inner Vec, and collect mutable references. However, two entries in `n_cols_per_domain` must not share the same log_size (same constraint as before — panic otherwise).

```rust
pub fn columns<const N: usize>(
    &mut self,
    n_cols_per_domain: [(CircleDomain, usize); N],
) -> [ColumnAccumulator<'_, B>; N] {
    // Validate no duplicate log_sizes (same constraint as before).
    let log_sizes = n_cols_per_domain.map(|(domain, _)| domain.log_size() as usize);
    let slots = self
        .sub_accumulations
        .get_disjoint_mut(log_sizes)
        .unwrap_or_else(|e| panic!("invalid log_sizes: {e}"));

    slots
        .into_iter()
        .zip(n_cols_per_domain)
        .map(|(entries, (domain, n_cols))| {
            let random_coeffs = self
                .random_coeff_powers
                .split_off(self.random_coeff_powers.len() - n_cols);
            // Push a new entry for this domain.
            entries.push((domain, SecureColumnByCoords::zeros(1 << domain.log_size())));
            let col = &mut entries.last_mut().unwrap().1;
            ColumnAccumulator {
                random_coeff_powers: random_coeffs,
                col,
            }
        })
        .collect_vec()
        .try_into()
        .unwrap_or_else(|_| unreachable!())
}
```

**Step 2: Add `use crate::core::poly::circle::CircleDomain;` import**

**Step 3: Build**

Run: `cargo build --features prover -p stwo`
Expected: Errors in callers of `columns()` — fix in Task 3.

**Step 4: Commit**

```
feat: columns() now takes CircleDomain instead of log_size
```

---

### Task 3: Update all callers of `columns()`

**Files:**
- Modify: `crates/constraint-framework/src/prover/component_prover.rs`
- Modify: `crates/examples/src/xor/gkr_lookups/mle_eval.rs`

All callers currently pass `(log_size, n_constraints)`. They need to pass `(domain, n_constraints)` instead.

**Step 1: Update `component_prover.rs` (SimdBackend)**

In the SimdBackend impl, the accumulator call is:
```rust
// Before:
let [mut accum] = evaluation_accumulator.columns([(log_eval, self.n_constraints())]);

// After:
let [mut accum] = evaluation_accumulator.columns([(constraint_domain, self.n_constraints())]);
```

Where `constraint_domain` is `subdomain` (D_sub path) or `eval_domain` (normal path). This is already computed in the current code as `constraint_domain`.

**Step 2: Update `component_prover.rs` (CpuBackend)**

Same pattern — pass the actual domain. In the D_sub branch pass `subdomain`, in the normal branch pass `eval_domain`.

**Step 3: Update `mle_eval.rs`**

```rust
// Before:
let [mut acc] = accumulator.columns([(eval_domain.log_size(), self.n_constraints())]);

// After:
let [mut acc] = accumulator.columns([(eval_domain, self.n_constraints())]);
```

**Step 4: Search for any other callers**

Run: `grep -rn "\.columns(\[(" crates/ --include="*.rs"`

Update any remaining callers.

**Step 5: Build all**

Run: `cargo build --features prover`
Expected: Compiles successfully.

**Step 6: Commit**

```
feat: update all columns() callers to pass CircleDomain
```

---

### Task 4: Update `finalize()` to convert non-canonical domains

**Files:**
- Modify: `crates/stwo/src/prover/air/accumulation.rs`

**Step 1: Update `finalize()`**

Before passing to `lift_and_accumulate`, convert each non-canonical entry to canonical:

```rust
pub fn finalize(self) -> SecureCirclePoly<B> {
    assert_eq!(
        self.random_coeff_powers.len(),
        0,
        "not all random coefficients were used"
    );
    let log_size = self.log_size();
    let _span = span!(
        Level::INFO,
        "Constraints interpolation",
        class = "ConstraintInterpolation"
    )
    .entered();

    // Flatten all (domain, column) pairs, converting non-canonical domains to canonical.
    let mut canonical_accumulations: Vec<SecureColumnByCoords<B>> = Vec::new();
    for entries in self.sub_accumulations {
        for (domain, col) in entries {
            let canonical = if domain.is_canonic() {
                col
            } else {
                let canonical_domain =
                    CanonicCoset::new(domain.log_size()).circle_domain();
                let sub_twiddles = B::precompute_twiddles(domain.half_coset);
                let canon_twiddles =
                    B::precompute_twiddles(canonical_domain.half_coset);
                SecureEvaluation::<B, BitReversedOrder>::new(domain, col)
                    .interpolate_with_twiddles(&sub_twiddles)
                    .evaluate_with_twiddles(canonical_domain, &canon_twiddles)
                    .values
            };
            canonical_accumulations.push(canonical);
        }
    }

    // Sort by size (required by lift_and_accumulate).
    canonical_accumulations.sort_by_key(|c| c.len());

    // Accumulate entries of the same size.
    let mut merged: Vec<SecureColumnByCoords<B>> = Vec::new();
    for col in canonical_accumulations {
        if let Some(last) = merged.last_mut() {
            if last.len() == col.len() {
                B::accumulate(last, &col);
                continue;
            }
        }
        merged.push(col);
    }

    let lifted_accumulation = B::lift_and_accumulate(merged);

    if let Some(eval) = lifted_accumulation {
        let twiddles =
            B::precompute_twiddles(CanonicCoset::new(log_size).circle_domain().half_coset);

        SecureCirclePoly(eval.columns.map(|c| {
            CircleEvaluation::<B, BaseField, BitReversedOrder>::new(
                CanonicCoset::new(log_size).circle_domain(),
                c,
            )
            .interpolate_with_twiddles(&twiddles)
        }))
    } else {
        SecureCirclePoly(std::array::from_fn(|_| {
            CircleCoefficients::new(Col::<B, BaseField>::zeros(1 << log_size))
        }))
    }
}
```

**Step 2: Add imports**

```rust
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::prover::poly::circle::SecureEvaluation;
use crate::prover::poly::BitReversedOrder;
```

(Some may already be imported.)

**Step 3: Build**

Run: `cargo build --features prover -p stwo`

**Step 4: Commit**

```
feat: finalize() converts non-canonical domains to canonical before lifting
```

---

### Task 5: Remove IFFT+FFT from component_prover's D_sub path

**Files:**
- Modify: `crates/constraint-framework/src/prover/component_prover.rs`

Now that the accumulator handles domain conversion in `finalize()`, the D_sub path in the SimdBackend impl can write quotients directly into the accumulator on the subdomain, without the IFFT+FFT and manual addition.

**Step 1: Simplify the SimdBackend D_sub path**

Since the accumulator now tracks the domain, the flow becomes:
1. Prepare trace on subdomain (slice columns)
2. Compute denom_inv for subdomain
3. Get accumulator column (passing subdomain as domain)
4. Run `accumulate_quotients_simd` directly into `accum.col`
5. No IFFT+FFT, no manual add — `finalize()` handles it

The code after `accumulate_quotients_simd` that does IFFT+FFT and adds to the accumulator should be removed.

**Step 2: Simplify the CpuBackend D_sub path similarly**

Same cleanup — write directly to accumulator, remove `dsub_to_canonical_cpu` call.

**Step 3: Remove now-unused helpers**

Delete `dsub_to_canonical_cpu` and `add_secure_columns_cpu` if no longer used.

**Step 4: Build and test**

Run: `cargo build --features prover`
Run: `cargo test --release -p stwo-examples`
Run: `cargo test --release --features prover -p stwo-constraint-framework`
Expected: All pass.

**Step 5: Commit**

```
feat: remove manual IFFT+FFT from D_sub path, defer to accumulator finalize
```

---

### Task 6: Run full test suite and verify

**Step 1: Run stwo tests**

Run: `cargo test --features prover -p stwo`

**Step 2: Run constraint-framework tests**

Run: `cargo test --release --features prover -p stwo-constraint-framework`

**Step 3: Run examples tests**

Run: `cargo test --release -p stwo-examples`

**Step 4: Run the existing accumulator test**

Run: `cargo test --features prover -p stwo test_domain_evaluation_accumulator_lifted`

This test uses canonical domains only, so it validates the no-regression path.

**Step 5: Run clippy**

Run: `scripts/clippy.sh`

**Step 6: Commit any fixes**

---

### Task 7: Add test for non-canonical domain accumulation

**Files:**
- Modify: `crates/stwo/src/prover/air/accumulation.rs` (test module)

**Step 1: Write a test that accumulates on a non-canonical domain**

Create a test that:
1. Creates an accumulator with max_log_size = 6.
2. Creates a polynomial, evaluates it on a canonical domain (log_size=5) and on a non-canonical subdomain (obtained via `split()`).
3. Accumulates the subdomain evaluation into the accumulator.
4. Calls `finalize()`.
5. Verifies the result matches what you'd get by evaluating on canonical and accumulating normally.

**Step 2: Run the test**

Run: `cargo test --features prover -p stwo test_non_canonical_domain_accumulation`

**Step 3: Commit**

```
test: add test for non-canonical domain accumulation in finalize
```
