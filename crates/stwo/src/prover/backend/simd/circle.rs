use std::iter::zip;
use std::mem::transmute;
use std::simd::Simd;

use bytemuck::Zeroable;
#[cfg(not(feature = "parallel"))]
use itertools::Itertools;
use num_traits::{One, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use tracing::{span, Level};

use super::fft::{ifft, rfft, CACHED_FFT_LOG_SIZE, MIN_FFT_LOG_SIZE};
use super::m31::{PackedBaseField, LOG_N_LANES, N_LANES};
use super::qm31::PackedSecureField;
use super::SimdBackend;
use crate::core::circle::{CirclePoint, CirclePointIndex, Coset, M31_CIRCLE_LOG_ORDER};
use crate::core::constraints::{coset_vanishing, coset_vanishing_derivative, point_vanishing};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{batch_inverse, Field, FieldExpOps};
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::core::poly::utils::{domain_line_twiddles_from_tree, fold, get_folding_alphas};
use crate::core::utils::bit_reverse_index;
use crate::prover::backend::cpu::circle::slow_precompute_twiddles;
use crate::prover::backend::simd::column::BaseColumn;
use crate::prover::backend::simd::fft::transpose_vecs;
use crate::prover::backend::simd::fri::fold_circle_evaluation_into_line;
use crate::prover::backend::simd::m31::PackedM31;
use crate::prover::backend::{Col, Column, CpuBackend};
use crate::prover::fri::FriOps;
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;

impl SimdBackend {
    // TODO(Ohad): optimize.
    fn twiddle_at<F: Field>(mappings: &[F], mut index: usize) -> F {
        debug_assert!(
            (1 << mappings.len()) as usize >= index,
            "Index out of bounds. mappings log len = {}, index = {index}",
            mappings.len().ilog2()
        );

        let mut product = F::one();
        for num in mappings.iter() {
            if index & 1 == 1 {
                product *= *num;
            }
            index >>= 1;
            if index == 0 {
                break;
            }
        }

        product
    }

    // TODO(Ohad): consider moving this to to a more general place.
    // Note: CACHED_FFT_LOG_SIZE is specific to the backend.
    fn generate_evaluation_mappings<F: Field>(point: CirclePoint<F>, log_size: u32) -> Vec<F> {
        // Mappings are the factors used to compute the evaluation twiddle.
        // Every twiddle (i) is of the form (m[0])^b_0 * (m[1])^b_1 * ... * (m[log_size -
        // 1])^b_log_size.
        // Where (m)_j are the mappings, and b_i is the j'th bit of i.
        let mut mappings = vec![point.y, point.x];
        let mut x = point.x;
        for _ in 2..log_size {
            x = CirclePoint::double_x(x);
            mappings.push(x);
        }

        // The caller function expects the mapping in natural order. i.e. (y,x,h(x),h(h(x)),...).
        // If the polynomial is large, the fft does a transpose in the middle in a granularity of 16
        // (avx512). The coefficients would then be in transposed order of 16-sized chunks.
        // i.e. (a_(n-15), a_(n-14), ..., a_(n-1), a_(n-31), ..., a_(n-16), a_(n-32), ...).
        // To compute the twiddles in the correct order, we need to transpose the coprresponding
        // 'transposed bits' in the mappings. The result order of the mappings would then be
        // (y, x, h(x), h^2(x), h^(log_n-1)(x), h^(log_n-2)(x) ...). To avoid code
        // complexity for now, we just reverse the mappings, transpose, then reverse back.
        // TODO(Ohad): optimize. consider changing the caller to expect the mappings in
        // reversed-transposed order.
        if log_size > CACHED_FFT_LOG_SIZE {
            mappings.reverse();
            let n = mappings.len();
            let n0 = (n - LOG_N_LANES as usize) / 2;
            let n1 = (n - LOG_N_LANES as usize).div_ceil(2);
            let (ab, c) = mappings.split_at_mut(n1);
            let (a, _b) = ab.split_at_mut(n0);
            // Swap content of a,c.
            a.swap_with_slice(&mut c[0..n0]);
            mappings.reverse();
        }

        mappings
    }

    // Generates twiddle steps for efficiently computing the twiddles.
    // steps[i] = t_i/(t_0*t_1*...*t_i-1).
    fn twiddle_steps<F: Field + FieldExpOps>(mappings: &[F]) -> Vec<F> {
        let mut denominators: Vec<F> = vec![mappings[0]];

        for i in 1..mappings.len() {
            denominators.push(denominators[i - 1] * mappings[i]);
        }

        let denom_inverses = F::batch_inverse(&denominators);

        let mut steps = vec![mappings[0]];

        mappings
            .iter()
            .skip(1)
            .zip(denom_inverses.iter())
            .for_each(|(m, d)| {
                steps.push(*m * *d);
            });
        steps.push(F::one());
        steps
    }

    // Advances the twiddle by multiplying it by the next step. e.g:
    //      If idx(t) = 0b100..1010 , then f(t) = t * step[0]
    //      If idx(t) = 0b100..0111 , then f(t) = t * step[3]
    fn advance_twiddle<F: Field>(twiddle: F, steps: &[F], curr_idx: usize) -> F {
        twiddle * steps[curr_idx.trailing_ones() as usize]
    }
}

// TODO(shahars): Everything is returned in redundant representation, where values can also be P.
// Decide if and when it's ok and what to do if it's not.
impl PolyOps for SimdBackend {
    // The twiddles type is i32, and not BaseField. This is because the fast AVX mul implementation
    //  requires one of the numbers to be shifted left by 1 bit. This is not a reduced
    //  representation of the field.
    type Twiddles = Vec<u32>;

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleCoefficients<Self> {
        let _span = span!(Level::TRACE, "", class = "iFFT").entered();
        let log_size = eval.values.length.ilog2();
        if log_size < MIN_FFT_LOG_SIZE {
            let cpu_poly = eval.to_cpu().interpolate();
            return CircleCoefficients::new(cpu_poly.coeffs.into_iter().collect());
        }

        let mut values = eval.values;
        let twiddles = domain_line_twiddles_from_tree(eval.domain, &twiddles.itwiddles);

        // Safe because [PackedBaseField] is aligned on 64 bytes.
        unsafe {
            ifft::ifft(
                transmute::<*mut PackedBaseField, *mut u32>(values.data.as_mut_ptr()),
                &twiddles,
                log_size as usize,
            );
        }

        // TODO(alont): Cache this inversion.
        let inv = PackedBaseField::broadcast(BaseField::from(eval.domain.size()).inverse());
        values.data.iter_mut().for_each(|x| *x *= inv);

        CircleCoefficients::new(values)
    }

    fn eval_at_point(
        poly: &CircleCoefficients<Self>,
        point: CirclePoint<SecureField>,
    ) -> SecureField {
        // If the polynomial is small, fallback to evaluate directly.
        // TODO(Ohad): it's possible to avoid falling back. Consider fixing.
        if poly.log_size() <= 8 {
            return slow_eval_at_point(poly, point);
        }

        let mappings = Self::generate_evaluation_mappings(point, poly.log_size());

        // 8 lowest mappings produce the first 2^8 twiddles. Separate to optimize each calculation.
        let (map_low, map_high) = mappings.split_at(4);
        let twiddle_lows =
            PackedSecureField::from_array(std::array::from_fn(|i| Self::twiddle_at(map_low, i)));
        let (map_mid, map_high) = map_high.split_at(4);
        let twiddle_mids =
            PackedSecureField::from_array(std::array::from_fn(|i| Self::twiddle_at(map_mid, i)));

        // Compute the high twiddle steps.
        let twiddle_steps = Self::twiddle_steps(map_high);

        // Every twiddle is a product of mappings that correspond to '1's in the bit representation
        // of the current index. For every 2^n aligned chunk of 2^n elements, the twiddle
        // array is the same, denoted twiddle_low. Use this to compute sums of (coeff *
        // twiddle_high) mod 2^n, then multiply by twiddle_low, and sum to get the final result.
        let compute_chunk_sum = |coeff_chunk: &[PackedBaseField],
                                 twiddle_mids: PackedSecureField,
                                 offset: usize| {
            let mut sum = PackedSecureField::zeroed();
            let mut twiddle_high = Self::twiddle_at(&mappings, offset * N_LANES);
            for (i, coeff_chunk) in coeff_chunk.array_chunks::<N_LANES>().enumerate() {
                // For every chunk of 2 ^ 4 * 2 ^ 4 = 2 ^ 8 elements, the twiddle high is the same.
                // Multiply it by every mid twiddle factor to get the factors for the current chunk.
                let high_twiddle_factors =
                    (PackedSecureField::broadcast(twiddle_high) * twiddle_mids).to_array();

                // Sum the coefficients multiplied by each corrseponsing twiddle. Result is
                // effectively an array[16] where the value at index 'i' is the sum
                // of all coefficients at indices that are i mod 16.
                for (&packed_coeffs, mid_twiddle) in zip(coeff_chunk, high_twiddle_factors) {
                    sum += PackedSecureField::broadcast(mid_twiddle) * packed_coeffs;
                }

                // Advance twiddle high.
                twiddle_high = Self::advance_twiddle(twiddle_high, &twiddle_steps, offset + i);
            }
            sum
        };

        #[cfg(not(feature = "parallel"))]
        let sum = compute_chunk_sum(&poly.coeffs.data, twiddle_mids, 0);

        #[cfg(feature = "parallel")]
        let sum: PackedSecureField = {
            const CHUNK_SIZE: usize = 1 << 10;
            let chunks = poly.coeffs.data.par_chunks(CHUNK_SIZE).enumerate();
            chunks
                .into_par_iter()
                .map(|(i, chunk)| compute_chunk_sum(chunk, twiddle_mids, i * CHUNK_SIZE))
                .sum()
        };

        (sum * twiddle_lows).pointwise_sum()
    }

    fn barycentric_weights(
        coset: CanonicCoset,
        p: CirclePoint<SecureField>,
    ) -> Col<SimdBackend, SecureField> {
        let domain = coset.circle_domain();
        let log_size = domain.log_size();
        let weights_vec_len = domain.size().div_ceil(N_LANES);
        if weights_vec_len == 1 {
            return Col::<SimdBackend, SecureField>::from_iter(CircleEvaluation::<
                CpuBackend,
                BaseField,
                BitReversedOrder,
            >::barycentric_weights(
                coset, p
            ));
        }

        let p = p.into_ef::<SecureField>();
        let p_0 = domain.at(0).into_ef::<SecureField>();
        let si_0 = SecureField::one()
            / ((p_0.y * SecureField::from(-2))
                * coset_vanishing_derivative(
                    Coset::new(CirclePointIndex::generator(), log_size),
                    p_0,
                ));

        #[cfg(not(feature = "parallel"))]
        let vi_p = (0..weights_vec_len)
            .map(|i| {
                PackedSecureField::from_array(std::array::from_fn(|j| {
                    point_vanishing(
                        domain
                            .at(bit_reverse_index(i * N_LANES + j, log_size))
                            .into_ef::<SecureField>(),
                        p,
                    )
                }))
            })
            .collect_vec();

        #[cfg(feature = "parallel")]
        let vi_p: Vec<PackedSecureField> = (0..weights_vec_len)
            .into_par_iter()
            .map(|i| {
                PackedSecureField::from_array(std::array::from_fn(|j| {
                    point_vanishing(
                        domain
                            .at(bit_reverse_index(i * N_LANES + j, log_size))
                            .into_ef::<SecureField>(),
                        p,
                    )
                }))
            })
            .collect();

        let vi_p_inverse = batch_inverse(&vi_p);

        let vn_p: SecureField = coset_vanishing(CanonicCoset::new(log_size).coset, p);

        // S_i(i) is invariant under G_(n−1) and alternate under J, meaning the S_i(i) values are
        // the same for each half coset, and the second half coset values are the conjugate
        // of the first half coset values.
        // weights_vec_len is even because domain.size() is a power of 2 (we already dealt with the
        // case where domain.size() < N_LANES).
        let si_i_vn_p = PackedSecureField::from_array(std::array::from_fn(|i| {
            if i.is_multiple_of(2) {
                si_0 * vn_p
            } else {
                -si_0 * vn_p
            }
        }));

        #[cfg(not(feature = "parallel"))]
        let weights = (0..weights_vec_len)
            .map(|i| vi_p_inverse[i] * si_i_vn_p)
            .collect_vec();

        #[cfg(feature = "parallel")]
        let weights: Vec<PackedSecureField> = (0..weights_vec_len)
            .into_par_iter()
            .map(|i| vi_p_inverse[i] * si_i_vn_p)
            .collect();

        Col::<Self, SecureField> {
            data: weights,
            length: domain.size(),
        }
    }

    fn barycentric_eval_at_point(
        evals: &CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>,
        weights: &Col<SimdBackend, SecureField>,
    ) -> SecureField {
        #[cfg(not(feature = "parallel"))]
        return (0..evals.domain.size().div_ceil(N_LANES))
            .fold(PackedSecureField::zero(), |acc, i| {
                acc + (weights.data[i] * evals.values.data[i])
            })
            .pointwise_sum();

        #[cfg(feature = "parallel")]
        return (0..evals.domain.size().div_ceil(N_LANES))
            .into_par_iter()
            .fold(
                PackedSecureField::zero,
                |acc: PackedSecureField, i: usize| acc + (weights.data[i] * evals.values.data[i]),
            )
            .sum::<PackedSecureField>()
            .to_array()
            .into_par_iter()
            .sum::<SecureField>();
    }

    fn eval_at_point_by_folding(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        point: CirclePoint<SecureField>,
        twiddles: &TwiddleTree<Self>,
    ) -> SecureField {
        let log_size = evals.domain.log_size();
        let mut folding_alphas = get_folding_alphas(point, log_size as usize);

        let mut layer_evaluation =
            fold_circle_evaluation_into_line(evals, folding_alphas.pop().unwrap(), twiddles);

        while layer_evaluation.len() > 1 {
            layer_evaluation =
                SimdBackend::fold_line(&layer_evaluation, folding_alphas.pop().unwrap(), twiddles);
        }

        layer_evaluation.values.at(0) / SecureField::from(2_u32.pow(log_size))
    }

    fn extend(poly: &CircleCoefficients<Self>, log_size: u32) -> CircleCoefficients<Self> {
        // TODO(shahars): Get rid of extends.
        poly.evaluate(CanonicCoset::new(log_size).circle_domain())
            .interpolate()
    }

    fn evaluate(
        poly: &CircleCoefficients<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let _span = span!(Level::TRACE, "", class = "rFFT").entered();
        let log_size = domain.log_size();
        let fft_log_size = poly.log_size();
        assert!(
            log_size >= fft_log_size,
            "Can only evaluate on larger domains"
        );

        if fft_log_size < MIN_FFT_LOG_SIZE {
            let cpu_poly: CircleCoefficients<CpuBackend> =
                CircleCoefficients::new(poly.coeffs.to_cpu());
            let cpu_eval = cpu_poly.evaluate(domain);
            return CircleEvaluation::new(
                cpu_eval.domain,
                Col::<SimdBackend, BaseField>::from_iter(cpu_eval.values),
            );
        }

        let twiddles = domain_line_twiddles_from_tree(domain, &twiddles.twiddles);

        // Evaluate on a big domains by evaluating on several subdomains.
        let log_subdomains = log_size - fft_log_size;

        // Allocate the destination buffer without initializing.
        let mut values = Vec::with_capacity(domain.size() >> LOG_N_LANES);
        #[allow(clippy::uninit_vec)]
        unsafe {
            values.set_len(domain.size() >> LOG_N_LANES)
        };

        for i in 0..(1 << log_subdomains) {
            // The subdomain twiddles are a slice of the large domain twiddles.
            let subdomain_twiddles = (0..(fft_log_size - 1))
                .map(|layer_i| {
                    &twiddles[layer_i as usize]
                        [i << (fft_log_size - 2 - layer_i)..(i + 1) << (fft_log_size - 2 - layer_i)]
                })
                .collect::<Vec<_>>();

            // FFT from the coefficients buffer to the values chunk.
            unsafe {
                rfft::fft(
                    transmute::<*const PackedBaseField, *const u32>(poly.coeffs.data.as_ptr()),
                    transmute::<*mut PackedBaseField, *mut u32>(
                        values[i << (fft_log_size - LOG_N_LANES)
                            ..(i + 1) << (fft_log_size - LOG_N_LANES)]
                            .as_mut_ptr(),
                    ),
                    &subdomain_twiddles,
                    fft_log_size as usize,
                );
            }
        }

        CircleEvaluation::new(
            domain,
            BaseColumn {
                data: values,
                length: domain.size(),
            },
        )
    }

    /// Precomputes the (doubled) twiddles for a given coset tower.
    /// The twiddles are the x values of each coset in bit-reversed order.
    /// Note: the coset point are symmetrical over the x-axis so only the first half of the coset is
    /// needed.
    fn precompute_twiddles(mut coset: Coset) -> TwiddleTree<Self> {
        let _span = span!(Level::TRACE, "", class = "PrecomputeTwiddles").entered();
        let root_coset = coset;

        if root_coset.size() < N_LANES {
            return compute_small_coset_twiddles(root_coset);
        }

        let mut twiddles = Vec::with_capacity(coset.size() / N_LANES);
        while coset.log_size() > LOG_N_LANES {
            compute_coset_twiddles(coset, &mut twiddles);
            coset = coset.double();
        }

        // Handle cosets smaller than `N_LANES`.
        let remaining_twiddles = slow_precompute_twiddles(coset);

        twiddles.push(PackedM31::from_array(
            remaining_twiddles.try_into().unwrap(),
        ));

        let itwiddles = PackedBaseField::batch_inverse(&twiddles);

        let dbl_twiddles = twiddles
            .into_iter()
            .flat_map(|x| (x.into_simd() * Simd::splat(2)).to_array())
            .collect();
        let dbl_itwiddles = itwiddles
            .into_iter()
            .flat_map(|x| (x.into_simd() * Simd::splat(2)).to_array())
            .collect();

        TwiddleTree {
            root_coset,
            twiddles: dbl_twiddles,
            itwiddles: dbl_itwiddles,
        }
    }

    fn split_at_mid(
        mut poly: CircleCoefficients<Self>,
    ) -> (CircleCoefficients<Self>, CircleCoefficients<Self>) {
        let length = poly.coeffs.length;

        // If the length fits only in one SIMD vector, need to split from the cpu vector.
        if length <= 1 << LOG_N_LANES {
            let mut cpu_vec = poly.coeffs.to_cpu();
            let right = cpu_vec.split_off(cpu_vec.len() / 2);
            return (
                CircleCoefficients::new(cpu_vec.into_iter().collect()),
                CircleCoefficients::new(right.into_iter().collect()),
            );
        }

        let log_length = length.ilog2();
        let log_n_vecs = log_length - LOG_N_LANES;

        // When the poly is large, IFFT doesn't end with a transpose, so we need to transpose the
        // coefficients before splitting.
        if log_length > CACHED_FFT_LOG_SIZE {
            unsafe {
                transpose_vecs(
                    transmute::<*mut PackedBaseField, *mut u32>(poly.coeffs.data.as_mut_ptr()),
                    log_n_vecs as usize,
                );
            }
        }

        let mut second = poly.coeffs.data.split_off(poly.coeffs.data.len() / 2);

        // If the new polynomials are large, we need to transpose the coefficients back before
        // returning because the FFT algorithm assumes the coefficients are transposed.
        if log_length - 1 > CACHED_FFT_LOG_SIZE {
            // transpose first and second
            unsafe {
                transpose_vecs(
                    transmute::<*mut PackedBaseField, *mut u32>(poly.coeffs.data.as_mut_ptr()),
                    (log_n_vecs - 1) as usize,
                );
                transpose_vecs(
                    transmute::<*mut PackedBaseField, *mut u32>(second.as_mut_ptr()),
                    (log_n_vecs - 1) as usize,
                );
            }
        }

        let left_length = length / 2;
        let right_length = length - left_length;

        (
            CircleCoefficients::new(BaseColumn {
                data: poly.coeffs.data,
                length: left_length,
            }),
            CircleCoefficients::new(BaseColumn {
                data: second,
                length: right_length,
            }),
        )
    }
}

fn compute_small_coset_twiddles(coset: Coset) -> TwiddleTree<SimdBackend> {
    let twiddles = slow_precompute_twiddles(coset);

    let dbl_twiddles = twiddles.iter().map(|x| x.0 * 2).collect();
    let dbl_itwiddles = twiddles.iter().map(|x| x.inverse().0 * 2).collect();
    TwiddleTree {
        root_coset: coset,
        twiddles: dbl_twiddles,
        itwiddles: dbl_itwiddles,
    }
}

/// Computes the twiddles of the coset in bit-reversed order. Optimized for SIMD.
fn compute_coset_twiddles(coset: Coset, twiddles: &mut Vec<PackedM31>) {
    let log_size = coset.log_size() - 1;
    assert!(log_size >= LOG_N_LANES);

    // Compute the first `N_LANES` circle points.
    let initial_points = std::array::from_fn(|i| coset.at(bit_reverse_index(i, log_size)));
    let mut current = CirclePoint {
        x: PackedM31::from_array(initial_points.each_ref().map(|p| p.x)),
        y: PackedM31::from_array(initial_points.each_ref().map(|p| p.y)),
    };

    // Precompute the steps needed to compute the next circle points in bit reversed order.
    let mut steps = [CirclePoint::zero(); (M31_CIRCLE_LOG_ORDER - LOG_N_LANES) as usize];
    for i in 0..(log_size - LOG_N_LANES) {
        let prev_mul = bit_reverse_index((1 << i) - 1, log_size - LOG_N_LANES);
        let new_mul = bit_reverse_index(1 << i, log_size - LOG_N_LANES);
        let step = coset.step.mul(new_mul as u128) - coset.step.mul(prev_mul as u128);
        steps[i as usize] = step;
    }

    for i in 0u32..1 << (log_size - LOG_N_LANES) {
        // Extract twiddle and compute the next `N_LANES` circle points.
        let x = current.x;
        let step_index = i.trailing_ones() as usize;
        let step = CirclePoint {
            x: PackedM31::broadcast(steps[step_index].x),
            y: PackedM31::broadcast(steps[step_index].y),
        };
        current = current + step;
        twiddles.push(x);
    }
}

fn slow_eval_at_point(
    poly: &CircleCoefficients<SimdBackend>,
    point: CirclePoint<SecureField>,
) -> SecureField {
    let mut mappings = vec![point.y];
    if poly.log_size() > 1 {
        mappings.push(point.x);
        let mut x = point.x;
        for _ in 2..poly.log_size() {
            x = CirclePoint::double_x(x);
            mappings.push(x);
        }
        mappings.reverse();
    }

    // If the polynomial is large, the fft does a transpose in the middle.
    if poly.log_size() > CACHED_FFT_LOG_SIZE {
        let n = mappings.len();
        let n0 = (n - LOG_N_LANES as usize) / 2;
        let n1 = (n - LOG_N_LANES as usize).div_ceil(2);
        let (ab, c) = mappings.split_at_mut(n1);
        let (a, _b) = ab.split_at_mut(n0);
        // Swap content of a,c.
        a.swap_with_slice(&mut c[0..n0]);
    }
    fold(poly.coeffs.as_slice(), &mappings)
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::circle::CirclePoint;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::CanonicCoset;
    use crate::prover::backend::simd::circle::slow_eval_at_point;
    use crate::prover::backend::simd::column::BaseColumn;
    use crate::prover::backend::simd::fft::{CACHED_FFT_LOG_SIZE, MIN_FFT_LOG_SIZE};
    use crate::prover::backend::simd::m31::LOG_N_LANES;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::backend::{Column, CpuBackend};
    use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
    use crate::prover::poly::{BitReversedOrder, NaturalOrder};

    #[test]
    fn test_interpolate_and_eval() {
        for log_size in MIN_FFT_LOG_SIZE..CACHED_FFT_LOG_SIZE + 4 {
            let domain = CanonicCoset::new(log_size).circle_domain();
            let evaluation = CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                domain,
                (0..1 << log_size).map(BaseField::from).collect(),
            );

            let poly = evaluation.clone().interpolate();
            let evaluation2 = poly.evaluate(domain);

            assert_eq!(evaluation.values.to_cpu(), evaluation2.values.to_cpu());
        }
    }

    #[test]
    fn test_eval_extension() {
        for log_size in MIN_FFT_LOG_SIZE..CACHED_FFT_LOG_SIZE + 2 {
            let domain = CanonicCoset::new(log_size).circle_domain();
            let domain_ext = CanonicCoset::new(log_size + 2).circle_domain();
            let evaluation = CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                domain,
                (0..1 << log_size).map(BaseField::from).collect(),
            );
            let poly = evaluation.clone().interpolate();

            let evaluation2 = poly.evaluate(domain_ext);

            assert_eq!(
                poly.extend(log_size + 2).coeffs.to_cpu(),
                evaluation2.interpolate().coeffs.to_cpu()
            );
        }
    }

    #[test]
    fn test_eval_at_point() {
        for log_size in MIN_FFT_LOG_SIZE + 1..CACHED_FFT_LOG_SIZE + 4 {
            let domain = CanonicCoset::new(log_size).circle_domain();
            let evaluation = CircleEvaluation::<SimdBackend, BaseField, NaturalOrder>::new(
                domain,
                (0..1 << log_size).map(BaseField::from).collect(),
            );
            let poly = evaluation.bit_reverse().interpolate();
            for i in [0, 1, 3, 1 << (log_size - 1), 1 << (log_size - 2)] {
                let p = domain.at(i);

                let eval = poly.eval_at_point(p.into_ef());

                assert_eq!(
                    eval,
                    BaseField::from(i).into(),
                    "log_size={log_size}, i={i}"
                );
            }
        }
    }

    #[test]
    fn test_simd_eval_at_point_by_folding() {
        let poly = CircleCoefficients::<SimdBackend>::new(BaseColumn::from_cpu(
            &[691, 805673, 5, 435684, 4832, 23876431, 197, 897346068].map(BaseField::from),
        ));
        let s = CanonicCoset::new(10);
        let domain = s.circle_domain();
        let eval = poly.evaluate(domain);
        let twiddles =
            SimdBackend::precompute_twiddles(CanonicCoset::new(11).circle_domain().half_coset);
        let sampled_points = [
            CirclePoint::get_point(348),
            CirclePoint::get_point(9736524),
            CirclePoint::get_point(13),
            CirclePoint::get_point(346752),
        ];
        let sampled_values = sampled_points
            .iter()
            .map(|point| poly.eval_at_point(*point))
            .collect_vec();

        let sampled_folding_values = sampled_points
            .iter()
            .map(|point| eval.eval_at_point_by_folding(*point, &twiddles))
            .collect_vec();

        assert_eq!(
            sampled_folding_values, sampled_values,
            "Evaluation by folding should be equal to the polynomial evaluation"
        );
    }

    #[test]
    fn test_circle_poly_extend() {
        for log_size in MIN_FFT_LOG_SIZE..CACHED_FFT_LOG_SIZE + 2 {
            let poly = CircleCoefficients::<SimdBackend>::new(
                (0..1 << log_size).map(BaseField::from).collect(),
            );
            let eval0 = poly.evaluate(CanonicCoset::new(log_size + 2).circle_domain());

            let eval1 = poly
                .extend(log_size + 2)
                .evaluate(CanonicCoset::new(log_size + 2).circle_domain());

            assert_eq!(eval0.values.to_cpu(), eval1.values.to_cpu());
        }
    }

    #[test]
    fn test_eval_securefield() {
        let mut rng = SmallRng::seed_from_u64(0);
        for log_size in MIN_FFT_LOG_SIZE..CACHED_FFT_LOG_SIZE + 2 {
            let domain = CanonicCoset::new(log_size).circle_domain();
            let evaluation = CircleEvaluation::<SimdBackend, BaseField, NaturalOrder>::new(
                domain,
                (0..1 << log_size).map(BaseField::from).collect(),
            );
            let poly = evaluation.bit_reverse().interpolate();
            let x = rng.gen();
            let y = rng.gen();
            let p = CirclePoint { x, y };

            let eval = PolyOps::eval_at_point(&poly, p);

            assert_eq!(eval, slow_eval_at_point(&poly, p), "log_size = {log_size}");
        }
    }

    #[test]
    fn test_optimized_precompute_twiddles() {
        let coset = CanonicCoset::new(10).half_coset();
        let twiddles = SimdBackend::precompute_twiddles(coset);
        let expected_twiddles = CpuBackend::precompute_twiddles(coset);

        assert_eq!(
            twiddles.twiddles,
            expected_twiddles
                .twiddles
                .iter()
                .map(|x| x.0 * 2)
                .collect_vec()
        );
    }
    #[test]
    fn test_circle_poly_split_at_mid_small() {
        let log_size = LOG_N_LANES;
        let poly = CircleCoefficients::<SimdBackend>::new(
            (0..1 << log_size).map(BaseField::from).collect(),
        );
        let (left, right) = poly.clone().split_at_mid();
        let random_point = CirclePoint::get_point(21903);

        assert_eq!(
            left.eval_at_point(random_point)
                + random_point.repeated_double(log_size - 2).x * right.eval_at_point(random_point),
            poly.eval_at_point(random_point)
        );
    }

    #[test]
    fn test_circle_poly_split_at_mid_medium() {
        let log_size = (CACHED_FFT_LOG_SIZE - LOG_N_LANES) / 2;
        let poly = CircleCoefficients::<SimdBackend>::new(
            (0..1 << log_size).map(BaseField::from).collect(),
        );
        let (left, right) = poly.clone().split_at_mid();
        let random_point = CirclePoint::get_point(21903);

        assert_eq!(
            left.eval_at_point(random_point)
                + random_point.repeated_double(log_size - 2).x * right.eval_at_point(random_point),
            poly.eval_at_point(random_point)
        );
    }

    #[test]
    fn test_circle_poly_split_at_mid_large() {
        let log_size = CACHED_FFT_LOG_SIZE + 1;
        let poly = CircleCoefficients::<SimdBackend>::new(
            (0..1 << log_size).map(BaseField::from).collect(),
        );
        let (left, right) = poly.clone().split_at_mid();
        let random_point = CirclePoint::get_point(21903);

        assert_eq!(
            left.eval_at_point(random_point)
                + random_point.repeated_double(log_size - 2).x * right.eval_at_point(random_point),
            poly.eval_at_point(random_point)
        );
    }

    #[test]
    fn test_simd_barycentric_evaluation() {
        let poly = CircleCoefficients::<SimdBackend>::new(BaseColumn::from_cpu(
            &[691, 805673, 5, 435684, 4832, 23876431, 197, 897346068].map(BaseField::from),
        ));
        let s = CanonicCoset::new(10);
        let domain = s.circle_domain();
        let eval = poly.evaluate(domain);
        let sampled_points = [
            CirclePoint::get_point(348),
            CirclePoint::get_point(9736524),
            CirclePoint::get_point(13),
            CirclePoint::get_point(346752),
        ];
        let sampled_values = sampled_points
            .iter()
            .map(|point| poly.eval_at_point(*point))
            .collect_vec();

        let sampled_barycentric_values = sampled_points
            .iter()
            .map(|point| {
                eval.barycentric_eval_at_point(&CircleEvaluation::<
                    SimdBackend,
                    BaseField,
                    BitReversedOrder,
                >::barycentric_weights(s, *point))
            })
            .collect_vec();

        assert_eq!(
            sampled_barycentric_values, sampled_values,
            "Barycentric evaluation should be equal to the polynomial evaluation"
        );
    }

    #[test]
    fn test_simd_barycentric_weights() {
        let s = CanonicCoset::new(10);
        let sampled_points = [
            CirclePoint::get_point(348),
            CirclePoint::get_point(9736524),
            CirclePoint::get_point(13),
            CirclePoint::get_point(346752),
        ];

        let cpu_weights = sampled_points
            .iter()
            .map(|point| {
                CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::barycentric_weights(
                    s, *point,
                )
            })
            .collect_vec();
        let simd_weights = sampled_points
            .iter()
            .map(|point| {
                CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::barycentric_weights(
                    s, *point,
                )
            })
            .collect_vec();

        cpu_weights
            .iter()
            .zip(simd_weights.iter())
            .for_each(|(cpu_weights, simd_weights)| {
                assert_eq!(*cpu_weights, simd_weights.to_cpu());
            });
    }
}
