use std::iter::zip;
use std::mem::transmute;
use std::simd::Simd;

use bytemuck::Zeroable;
use num_traits::One;

use super::fft::{ifft, rfft, CACHED_FFT_LOG_SIZE, MIN_FFT_LOG_SIZE};
use super::m31::{PackedBaseField, LOG_N_LANES, N_LANES};
use super::qm31::PackedSecureField;
use super::SimdBackend;
use crate::core::backend::cpu::circle::slow_precompute_twiddles;
use crate::core::backend::simd::column::BaseColumn;
use crate::core::backend::simd::m31::PackedM31;
use crate::core::backend::{Col, Column, CpuBackend};
use crate::core::circle::{CirclePoint, Coset, M31_CIRCLE_LOG_ORDER};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{Field, FieldExpOps};
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::utils::{domain_line_twiddles_from_tree, fold};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::bit_reverse_index;

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
        // (avx512). The coefficients would then be in tranposed order of 16-sized chunks.
        // i.e. (a_(n-15), a_(n-14), ..., a_(n-1), a_(n-31), ..., a_(n-16), a_(n-32), ...).
        // To compute the twiddles in the correct order, we need to transpose the coprresponding
        // 'transposed bits' in the mappings. The result order of the mappings would then be
        // (y, x, h(x), h^2(x), h^(log_n-1)(x), h^(log_n-2)(x) ...). To avoid code
        // complexity for now, we just reverse the mappings, transpose, then reverse back.
        // TODO(Ohad): optimize. consider changing the caller to expect the mappings in
        // reversed-tranposed order.
        if log_size > CACHED_FFT_LOG_SIZE {
            mappings.reverse();
            let n = mappings.len();
            let n0 = (n - LOG_N_LANES as usize) / 2;
            let n1 = (n - LOG_N_LANES as usize + 1) / 2;
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

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // TODO(Ohad): Optimize.
        let eval = CpuBackend::new_canonical_ordered(coset, values.into_cpu_vec());
        CircleEvaluation::new(
            eval.domain,
            Col::<SimdBackend, BaseField>::from_iter(eval.values),
        )
    }

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        let log_size = eval.values.length.ilog2();
        if log_size < MIN_FFT_LOG_SIZE {
            let cpu_poly = eval.to_cpu().interpolate();
            return CirclePoly::new(cpu_poly.coeffs.into_iter().collect());
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

        CirclePoly::new(values)
    }

    fn eval_at_point(poly: &CirclePoly<Self>, point: CirclePoint<SecureField>) -> SecureField {
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
        // of the current index. For every 2^n alligned chunk of 2^n elements, the twiddle
        // array is the same, denoted twiddle_low. Use this to compute sums of (coeff *
        // twiddle_high) mod 2^n, then multiply by twiddle_low, and sum to get the final result.
        let mut sum = PackedSecureField::zeroed();
        let mut twiddle_high = SecureField::one();
        for (i, coeff_chunk) in poly.coeffs.data.array_chunks::<N_LANES>().enumerate() {
            // For every chunk of 2 ^ 4 * 2 ^ 4 = 2 ^ 8 elements, the twiddle high is the same.
            // Multiply it by every mid twiddle factor to get the factors for the current chunk.
            let high_twiddle_factors =
                (PackedSecureField::broadcast(twiddle_high) * twiddle_mids).to_array();

            // Sum the coefficients multiplied by each corrseponsing twiddle. Result is effectivley
            // an array[16] where the value at index 'i' is the sum of all coefficients at indices
            // that are i mod 16.
            for (&packed_coeffs, mid_twiddle) in zip(coeff_chunk, high_twiddle_factors) {
                sum += PackedSecureField::broadcast(mid_twiddle) * packed_coeffs;
            }

            // Advance twiddle high.
            twiddle_high = Self::advance_twiddle(twiddle_high, &twiddle_steps, i);
        }

        (sum * twiddle_lows).pointwise_sum()
    }

    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self> {
        // TODO(shahars): Get rid of extends.
        poly.evaluate(CanonicCoset::new(log_size).circle_domain())
            .interpolate()
    }

    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let log_size = domain.log_size();
        let fft_log_size = poly.log_size();
        assert!(
            log_size >= fft_log_size,
            "Can only evaluate on larger domains"
        );

        if fft_log_size < MIN_FFT_LOG_SIZE {
            let cpu_poly: CirclePoly<CpuBackend> = CirclePoly::new(poly.coeffs.to_cpu());
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
    poly: &CirclePoly<SimdBackend>,
    point: CirclePoint<SecureField>,
) -> SecureField {
    let mut mappings = vec![point.y, point.x];
    let mut x = point.x;
    for _ in 2..poly.log_size() {
        x = CirclePoint::double_x(x);
        mappings.push(x);
    }
    mappings.reverse();

    // If the polynomial is large, the fft does a transpose in the middle.
    if poly.log_size() > CACHED_FFT_LOG_SIZE {
        let n = mappings.len();
        let n0 = (n - LOG_N_LANES as usize) / 2;
        let n1 = (n - LOG_N_LANES as usize + 1) / 2;
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

    use crate::core::backend::simd::circle::slow_eval_at_point;
    use crate::core::backend::simd::fft::{CACHED_FFT_LOG_SIZE, MIN_FFT_LOG_SIZE};
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::circle::CirclePoint;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly, PolyOps};
    use crate::core::poly::{BitReversedOrder, NaturalOrder};

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
    fn test_circle_poly_extend() {
        for log_size in MIN_FFT_LOG_SIZE..CACHED_FFT_LOG_SIZE + 2 {
            let poly =
                CirclePoly::<SimdBackend>::new((0..1 << log_size).map(BaseField::from).collect());
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
}
