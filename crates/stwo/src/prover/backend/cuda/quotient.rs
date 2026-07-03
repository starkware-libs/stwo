use crate::core::fields::m31::BaseField;
use crate::core::pcs::quotients::{quotient_constants, ColumnSampleBatch};
use crate::core::poly::circle::CanonicCoset;
use crate::prover::backend::cuda::secure_column::CudaSecureColumn;
use crate::prover::backend::cuda::{fused_commit, CudaBackend};
use crate::prover::backend::simd::column::BaseColumn as SimdBaseColumn;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::Column;
use crate::prover::pcs::quotient_ops::AccumulatedNumerators;
use crate::prover::poly::circle::{
    CircleCoefficients, CircleEvaluation, PolyOps, SecureEvaluation,
};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::QuotientOps;
use crate::stwo_cuda as interface;
use crate::stwo_cuda::base_field_vec::BaseFieldVec;
use crate::stwo_cuda::bindings::{CirclePointSecureField, CudaSecureField};

// NATIVE (device-resident) LIFTED QuotientOps for CudaBackend, matching the 74951f79 lifted
// quotient scheme byte-identically to SimdBackend
// (crates/stwo/src/prover/backend/simd/quotients.rs).
//
// The lifted scheme is expressed as Rust orchestration of EXISTING NitrooZK device kernels plus the
// device-resident NTT (CudaBackend::interpolate / CudaBackend::evaluate). No new CUDA was written.
//
//   accumulate_numerators: NitrooZK's `accumulate_numerators_batch` kernel run over the SUBDOMAIN
//   (first `size >> log_blowup_factor` rows, in bit-reversed order — a prefix of the committed
//   device column), mirroring simd `accumulate_numerators_on_subdomain` (quotients.rs L208-253).
//
//   compute_quotients_and_combine: NitrooZK's `compute_quotients_and_combine` kernel run over the
//   SUBDOMAIN (giving the quotient on the subdomain, mirroring simd quotients.rs L116-179), then
// the   on-device lift: for each of the 4 secure coords, interpolate the subdomain eval -> coeffs
// and   evaluate on the full domain (simd quotients.rs L188-197).
//
// SAFETY OF THE PORT:
// * The combine kernel's lifting index `(row >> (log_ratio+1) << 1) + (row & 1)` was verified to be
//   byte-identical to simd `to_lifted_simd`'s scalar source-index mapping for all log_ratios.
// * The kernel's `domain_at_index(half_coset.initial_index, half_coset.step_size,
//   bit_reverse(row))` convention matches simd's `CircleDomainBitRevIterator` (==
//   `domain.at(bit_reverse_index(row))`), the same convention validated byte-identical-to-CPU by
//   the barycentric path/tests.
// * The denominator-inverse and numerator math in both kernels matches simd line-for-line (see the
//   per-function comments below).
//
// Small subdomains (`subdomain.log_size() < LOG_N_LANES == 4`) are handled by delegating to
// SimdBackend, which itself falls back to CPU for that case — this reproduces simd's exact small
// path byte-for-byte instead of relying on CudaBackend NTT's own (separate) <=3 CPU fallback.
//
// FALLBACK: setting env `CUDA_QUOTIENT_CPU_FALLBACK=1` forces the original SIMD-delegated (host
// round-trip) path for the whole op, so the maintainer can A/B native-vs-delegate on the box.

const LOG_N_LANES: u32 = 4;

fn cpu_fallback_enabled() -> bool {
    std::env::var("CUDA_QUOTIENT_CPU_FALLBACK")
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn eval_cuda_to_simd(
    c: &CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>,
) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    CircleEvaluation::new(c.domain, c.values.to_cpu().into_iter().collect())
}

fn securecol_cuda_to_simd(
    c: &SecureColumnByCoords<CudaBackend>,
) -> SecureColumnByCoords<SimdBackend> {
    SecureColumnByCoords {
        columns: std::array::from_fn(|i| {
            c.columns[i]
                .to_cpu()
                .into_iter()
                .collect::<SimdBaseColumn>()
        }),
    }
}

fn securecol_simd_to_cuda(
    c: SecureColumnByCoords<SimdBackend>,
) -> SecureColumnByCoords<CudaBackend> {
    SecureColumnByCoords {
        columns: c.columns.map(|col| BaseFieldVec::from_vec(col.to_cpu())),
    }
}

fn twiddles_cuda_to_simd(t: &TwiddleTree<CudaBackend>) -> TwiddleTree<SimdBackend> {
    // RECOMPUTE the SIMD twiddles from the root coset rather than converting the CudaBackend
    // buffers: NitrooZK's CUDA twiddle layout is NOT guaranteed to match SimdBackend's internal
    // (bit-reversed, layered) Vec<u32> layout, so a raw value-copy would feed the SIMD quotient
    // code mis-laid-out twiddles. Twiddles are a deterministic function of the coset.
    SimdBackend::precompute_twiddles(t.root_coset)
}

// === SIMD-delegated fallback (the pre-native v1 implementation, kept for A/B). ===

fn accumulate_numerators_simd_delegated(
    columns: &[&CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>],
    sample_batches: &[ColumnSampleBatch],
    accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<CudaBackend>>,
    log_blowup_factor: u32,
) {
    let simd_cols: Vec<_> = columns.iter().map(|c| eval_cuda_to_simd(c)).collect();
    let simd_refs: Vec<_> = simd_cols.iter().collect();
    let mut simd_acc: Vec<AccumulatedNumerators<SimdBackend>> = vec![];
    SimdBackend::accumulate_numerators(
        &simd_refs,
        sample_batches,
        &mut simd_acc,
        log_blowup_factor,
    );
    for a in simd_acc {
        accumulated_numerators_vec.push(AccumulatedNumerators {
            sample_point: a.sample_point,
            partial_numerators_acc: securecol_simd_to_cuda(a.partial_numerators_acc),
            first_linear_term_acc: a.first_linear_term_acc,
        });
    }
}

fn compute_quotients_and_combine_simd_delegated(
    accs: Vec<AccumulatedNumerators<CudaBackend>>,
    lifting_log_size: u32,
    log_blowup_factor: u32,
    twiddles: &TwiddleTree<CudaBackend>,
) -> SecureEvaluation<CudaBackend, BitReversedOrder> {
    let simd_accs: Vec<AccumulatedNumerators<SimdBackend>> = accs
        .iter()
        .map(|a| AccumulatedNumerators {
            sample_point: a.sample_point,
            partial_numerators_acc: securecol_cuda_to_simd(&a.partial_numerators_acc),
            first_linear_term_acc: a.first_linear_term_acc,
        })
        .collect();
    let simd_tw = twiddles_cuda_to_simd(twiddles);
    let res = SimdBackend::compute_quotients_and_combine(
        simd_accs,
        lifting_log_size,
        log_blowup_factor,
        &simd_tw,
    );
    SecureEvaluation::new(res.domain, securecol_simd_to_cuda(res.values))
}

/// Per-batch flattened accumulate-numerators terms, built once and reused across every
/// subdomain row-block in the staged (row-tiled) path. Mirrors the arguments the
/// `accumulate_numerators_batch` kernel consumes (`acc += c_j * col[idx_j][row] - b_j`).
struct BatchTerms {
    line_coeffs_b: Vec<CudaSecureField>,
    line_coeffs_c: Vec<CudaSecureField>,
    column_indices: Vec<u32>,
    n_terms: usize,
}

impl QuotientOps for CudaBackend {
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
        log_blowup_factor: u32,
    ) {
        let domain = columns[0].domain;
        // simd quotients.rs L42: subdomain = first `size >> log_blowup_factor` rows.
        let (subdomain, _) = domain.split(log_blowup_factor);

        // Mirror simd's small-subdomain CPU path (simd quotients.rs L45-65) and the env fallback by
        // delegating to SimdBackend (which itself falls back to CPU for sub-LANE subdomains).
        if cpu_fallback_enabled() || subdomain.log_size() < LOG_N_LANES {
            return accumulate_numerators_simd_delegated(
                columns,
                sample_batches,
                accumulated_numerators_vec,
                log_blowup_factor,
            );
        }

        let subdomain_size = subdomain.size();
        let any_staged = columns.iter().any(|c| fused_commit::is_staged(&c.values));

        // COMPOSITION_TILING_SCOPE (route c) §1.5: the accumulate kernel reads
        // `columns[column_index][row]` at global `row`, offset 0, pure pointwise over the SUBDOMAIN
        // prefix (`size >> blowup`, ~half the eval size at blowup 1). When the streamed commit
        // (GATE_AIR_STREAM_COMMIT) staged the columns, rehydrating them ALL whole is ~½ the commit-
        // peak eval set at once — still busting 40 GB at 2^24. So under `any_staged` we ROW-TILE:
        // loop subdomain row-blocks, H2D only that block's slice of every referenced column into a
        // reused tile buffer (`rehydrate_block`), and run the SAME kernel over `[0, block)` with
        // the RESULT pointers biased by `+off` so local row `r` writes global subdomain
        // slot `off+r`. The kernel loops `row in [0, size)` with `size = block`, so tile
        // columns are indexed LOCAL (no column-pointer bias) and only the result base moves
        // — byte-identical to the whole- subdomain launch (pure pointwise,
        // order-independent, slice-exact). Non-staged path is the legacy whole-subdomain
        // launch, byte-for-byte unchanged.
        let quotient_constants = quotient_constants(sample_batches);

        if !any_staged {
            // ---- Legacy resident path (byte-for-byte unchanged). ----
            let host_col_ptrs: Vec<*const u32> =
                columns.iter().map(|c| c.values.device_ptr).collect();
            let device_col_ptrs = unsafe {
                interface::bindings::copy_device_pointer_vec_from_host_to_device(
                    host_col_ptrs.as_ptr(),
                    host_col_ptrs.len(),
                )
            };
            for (batch, coeffs) in sample_batches.iter().zip(quotient_constants.line_coeffs) {
                let line_coeffs_b: Vec<CudaSecureField> = coeffs
                    .iter()
                    .map(|(_, b, _)| CudaSecureField::from(*b))
                    .collect();
                let line_coeffs_c: Vec<CudaSecureField> = coeffs
                    .iter()
                    .map(|(_, _, c)| CudaSecureField::from(*c))
                    .collect();
                let column_indices: Vec<u32> = batch
                    .cols_vals_randpows
                    .iter()
                    .map(|n| n.column_index as u32)
                    .collect();
                let result = unsafe { CudaSecureColumn::new_with_size(subdomain_size) };
                unsafe {
                    interface::bindings::accumulate_numerators_batch(
                        subdomain_size as u32,
                        device_col_ptrs,
                        line_coeffs_b.as_ptr(),
                        line_coeffs_c.as_ptr(),
                        column_indices.as_ptr(),
                        coeffs.len() as u32,
                        result.columns[0].device_ptr,
                        result.columns[1].device_ptr,
                        result.columns[2].device_ptr,
                        result.columns[3].device_ptr,
                    );
                }
                let first_linear_term_acc = coeffs.iter().map(|(a, ..)| *a).sum();
                accumulated_numerators_vec.push(AccumulatedNumerators {
                    sample_point: batch.point,
                    partial_numerators_acc: result,
                    first_linear_term_acc,
                });
            }
            unsafe {
                interface::bindings::cuda_free_memory(device_col_ptrs as *const std::ffi::c_void);
            }
            return;
        }

        // ---- Staged path: per-subdomain-block H2D from the stash (residency O(block * n_cols)).
        // ---- Tile size (rows/block); same knob family as the composition kernel. Clamp to
        // subdomain.
        let mut block_rows: usize = std::env::var("GATE_AIR_TILE_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&b| b > 0)
            .unwrap_or(1usize << 20);
        if block_rows > subdomain_size {
            block_rows = subdomain_size;
        }

        // One persistent result secure-column per batch (full subdomain), written block-by-block.
        let results: Vec<SecureColumnByCoords<CudaBackend>> = sample_batches
            .iter()
            .map(|_| unsafe { CudaSecureColumn::new_with_size(subdomain_size) })
            .collect();
        // Per-batch coefficient/index tables (batch-invariant across blocks; build once).
        let batch_terms: Vec<BatchTerms> = sample_batches
            .iter()
            .zip(quotient_constants.line_coeffs.iter())
            .map(|(batch, coeffs)| BatchTerms {
                line_coeffs_b: coeffs
                    .iter()
                    .map(|(_, b, _)| CudaSecureField::from(*b))
                    .collect(),
                line_coeffs_c: coeffs
                    .iter()
                    .map(|(_, _, c)| CudaSecureField::from(*c))
                    .collect(),
                column_indices: batch
                    .cols_vals_randpows
                    .iter()
                    .map(|n| n.column_index as u32)
                    .collect(),
                n_terms: coeffs.len(),
            })
            .collect();

        let mut off = 0usize;
        while off < subdomain_size {
            let this_block = (subdomain_size - off).min(block_rows);

            // Build this block's per-column device pointer for EVERY referenced column.
            //   * STAGED column (tree1's large eval cols, dehydrated at streamed commit): H2D only
            //     this block's slice from the host stash into a reused tile buffer
            //     (`rehydrate_block`). The tile buffer is indexed LOCAL (`[0, block)`).
            //   * RESIDENT column (tree0 preprocessed / tree2 interaction — never streamed, live on
            //     device): its committed bytes are ALREADY on the device in the same eval-domain
            //     order the kernel indexes, so we slice its LIVE buffer directly — a BORROWED view
            //     biased by `+off` (`device_ptr.add(off)`), so local row `r` reads global slot
            //     `off+r`. Same biased-pointer trick already used for the result columns below;
            //     produces byte-identical values to a staged slice (same bytes, same `col[row]`),
            //     with no D2H/H2D round-trip. `from_borrowed_ptr` keeps `owns_memory = false` so
            //     dropping the tile view never frees the live column. Fail-loud if a resident block
            //     would run past the buffer (never read a bad address — the 2^24 illegal-address
            //     class).
            // T2: split the staged block rehydrate H2D from the accumulate kernel compute.
            let t2 = fused_commit::t1_timers_on();
            let h2d_start = t2.then(std::time::Instant::now);
            let tile_cols: Vec<BaseFieldVec> = columns
                .iter()
                .map(|c| {
                    if fused_commit::is_staged(&c.values) {
                        fused_commit::rehydrate_block(&c.values, off, this_block)
                    } else {
                        assert!(
                            off + this_block <= c.values.size,
                            "quotient row-tiling: resident column slice [{off}, {}) exceeds \
                             column length {}",
                            off + this_block,
                            c.values.size
                        );
                        BaseFieldVec::from_borrowed_ptr(
                            unsafe { c.values.device_ptr.add(off) },
                            this_block,
                        )
                    }
                })
                .collect();
            let host_col_ptrs: Vec<*const u32> = tile_cols.iter().map(|c| c.device_ptr).collect();
            let device_col_ptrs = unsafe {
                interface::bindings::copy_device_pointer_vec_from_host_to_device(
                    host_col_ptrs.as_ptr(),
                    host_col_ptrs.len(),
                )
            };
            if let Some(s) = h2d_start {
                crate::prover::prove_ex_sync();
                fused_commit::t2_add(2, s.elapsed().as_nanos());
            }
            let k_start = t2.then(std::time::Instant::now);

            for (bi, terms) in batch_terms.iter().enumerate() {
                // Bias each result coord by +off so local row r writes global subdomain slot off+r.
                let r0 = unsafe { results[bi].columns[0].device_ptr.add(off) };
                let r1 = unsafe { results[bi].columns[1].device_ptr.add(off) };
                let r2 = unsafe { results[bi].columns[2].device_ptr.add(off) };
                let r3 = unsafe { results[bi].columns[3].device_ptr.add(off) };
                unsafe {
                    interface::bindings::accumulate_numerators_batch(
                        this_block as u32,
                        device_col_ptrs,
                        terms.line_coeffs_b.as_ptr(),
                        terms.line_coeffs_c.as_ptr(),
                        terms.column_indices.as_ptr(),
                        terms.n_terms as u32,
                        r0,
                        r1,
                        r2,
                        r3,
                    );
                }
            }
            unsafe {
                interface::bindings::cuda_free_memory(device_col_ptrs as *const std::ffi::c_void);
            }
            if let Some(s) = k_start {
                crate::prover::prove_ex_sync();
                fused_commit::t2_add(3, s.elapsed().as_nanos());
            }
            // tile_cols drop here -> per-column tile device buffers freed before the next block.
            off += this_block;
        }

        // Move each persistent (full-subdomain) result out by value, in batch order (results,
        // sample_batches, and line_coeffs are all built in the same order).
        for ((partial, batch), coeffs) in results
            .into_iter()
            .zip(sample_batches.iter())
            .zip(quotient_constants.line_coeffs.iter())
        {
            let first_linear_term_acc = coeffs.iter().map(|(a, ..)| *a).sum();
            accumulated_numerators_vec.push(AccumulatedNumerators {
                sample_point: batch.point,
                partial_numerators_acc: partial,
                first_linear_term_acc,
            });
        }
    }

    fn compute_quotients_and_combine(
        accumulations: Vec<AccumulatedNumerators<Self>>,
        lifting_log_size: u32,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<Self>,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        let eval_domain = CanonicCoset::new(lifting_log_size).circle_domain();
        // simd quotients.rs L91-92.
        let (eval_subdomain, _) = eval_domain.split(log_blowup_factor);

        // Mirror simd's small-subdomain CPU path (simd quotients.rs L94-115) and the env fallback.
        if cpu_fallback_enabled() || eval_subdomain.log_size() < LOG_N_LANES {
            return compute_quotients_and_combine_simd_delegated(
                accumulations,
                lifting_log_size,
                log_blowup_factor,
                twiddles,
            );
        }

        let subdomain_size = eval_subdomain.size();
        let subdomain_log_size = eval_subdomain.log_size();
        let num_acc = accumulations.len();

        // === Step 1: compute the quotient on the SUBDOMAIN (simd quotients.rs L116-179). ===
        // The kernel computes, per subdomain row, for each accumulation:
        //   den_inv  = inv((prx - p.x)*piy - (pry - p.y)*pix)        (simd L255-283 / cu L259-267)
        //   lifted   = partial_acc[lift(row, log_ratio)]             (simd L157-166 / cu L270-278)
        //   full_num = lifted - first_linear_term_acc * p.y          (simd L168-169 / cu L280-281)
        //   quotient += full_num * den_inv                           (simd L170     / cu L283-284)
        // `acc_log_sizes[a]` is the log size of accumulation a's partial numerators; the kernel
        // derives `log_ratio = subdomain_log_size - acc_log_sizes[a]` (== simd L126-129).
        let acc_partial_ptrs: Vec<*const u32> = accumulations
            .iter()
            .flat_map(|acc| {
                // Order a.a, a.b, b.a, b.b — matches accumulate kernel result_0..3 and simd coords.
                acc.partial_numerators_acc
                    .columns
                    .iter()
                    .map(|col| col.device_ptr)
                    .collect::<Vec<_>>()
            })
            .collect();
        let acc_log_sizes: Vec<i32> = accumulations
            .iter()
            .map(|acc| acc.partial_numerators_acc.columns[0].len().ilog2() as i32)
            .collect();
        let first_linear_term_accs: Vec<CudaSecureField> = accumulations
            .iter()
            .map(|acc| CudaSecureField::from(acc.first_linear_term_acc))
            .collect();
        let sample_points: Vec<CirclePointSecureField> = accumulations
            .iter()
            .map(|acc| CirclePointSecureField::from(acc.sample_point))
            .collect();

        let subdomain_quotient = unsafe { CudaSecureColumn::new_with_size(subdomain_size) };
        unsafe {
            interface::bindings::compute_quotients_and_combine(
                subdomain_size as u32,
                subdomain_log_size,
                eval_subdomain.half_coset.initial_index.0 as u32,
                eval_subdomain.half_coset.step_size.0 as u32,
                num_acc as u32,
                acc_partial_ptrs.as_ptr(),
                acc_log_sizes.as_ptr(),
                first_linear_term_accs.as_ptr(),
                sample_points.as_ptr(),
                subdomain_quotient.columns[0].device_ptr,
                subdomain_quotient.columns[1].device_ptr,
                subdomain_quotient.columns[2].device_ptr,
                subdomain_quotient.columns[3].device_ptr,
            );
        }

        // === Step 2: lift subdomain quotient to the full domain (simd quotients.rs L180-197). ===
        // For each of the 4 secure coords: interpolate the subdomain eval -> coeffs, then evaluate
        // on the full eval_domain. CudaBackend::interpolate / ::evaluate are the device NTTs. The
        // interpolation uses the SUBDOMAIN twiddles (deterministic from eval_subdomain.half_coset,
        // identical values to simd's `extract_subdomain_twiddles`); evaluation uses the full
        // `twiddles` (root_coset == eval_domain.half_coset), exactly as simd does.
        let subdomain_twiddles = CudaBackend::precompute_twiddles(eval_subdomain.half_coset);
        let lifted_columns: [BaseFieldVec; 4] = subdomain_quotient.columns.map(|coord| {
            let coeffs: CircleCoefficients<CudaBackend> = CudaBackend::interpolate(
                CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(
                    eval_subdomain,
                    coord,
                ),
                &subdomain_twiddles,
            );
            CudaBackend::evaluate(&coeffs, eval_domain, twiddles).values
        });

        SecureEvaluation::new(
            eval_domain,
            SecureColumnByCoords {
                columns: lifted_columns,
            },
        )
    }
}
