use std::iter::zip;
use std::mem::transmute;

use bytemuck::{cast_slice, Zeroable};
use num_traits::One;

use super::fft::{ifft, rfft, CACHED_FFT_LOG_SIZE};
use super::m31::{PackedBaseField, PackedM31, LOG_N_LANES, N_LANES};
use super::qm31::PackedSecureField;
use super::SimdBackend;
use crate::core::backend::simd::column::BaseFieldVec;
use crate::core::backend::{Col, CpuBackend};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{Field, FieldExpOps};
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::utils::{domain_line_twiddles_from_tree, fold};
use crate::core::poly::BitReversedOrder;

impl SimdBackend {
    // TODO(Ohad): optimize.
    fn twiddle_at<F: Field>(mappings: &[F], mut index: usize) -> F {
        debug_assert!(
            (1 << mappings.len()) as usize >= index,
            "Index out of bounds. mappings log len = {}, index = {index}",
            mappings.len().ilog2()
        );

        let mut product = F::one();
        for &num in mappings.iter() {
            if index & 1 == 1 {
                product *= num;
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
    fn twiddle_steps<F: Field>(mappings: &[F]) -> Vec<F>
    where
        F: FieldExpOps,
    {
        let mut denominators: Vec<F> = vec![mappings[0]];

        for i in 1..mappings.len() {
            denominators.push(denominators[i - 1] * mappings[i]);
        }

        let mut denom_inverses = vec![F::zero(); denominators.len()];
        F::batch_inverse(&denominators, &mut denom_inverses);

        let mut steps = vec![mappings[0]];

        mappings
            .iter()
            .skip(1)
            .zip(denom_inverses.iter())
            .for_each(|(&m, &d)| {
                steps.push(m * d);
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

// TODO(spapini): Everything is returned in redundant representation, where values can also be P.
// Decide if and when it's ok and what to do if it's not.
impl PolyOps for SimdBackend {
    // The twiddles type is i32, and not BaseField. This is because the fast AVX mul implementation
    //  requries one of the numbers to be shifted left by 1 bit. This is not a reduced
    //  representation of the field.
    type Twiddles = Vec<u32>;

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // TODO(spapini): Optimize.
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
        let mut values = eval.values;
        let log_size = values.length.ilog2();

        let twiddles = domain_line_twiddles_from_tree(eval.domain, &twiddles.itwiddles);

        // Safe because [PackedBaseField] is aligned on 64 bytes.
        unsafe {
            ifft::ifft(
                transmute(values.data.as_mut_ptr()),
                &twiddles,
                log_size as usize,
            );
        }

        // TODO(spapini): Fuse this multiplication / rotation.
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
        // TODO(spapini): Optimize or get rid of extend.
        poly.evaluate(CanonicCoset::new(log_size).circle_domain())
            .interpolate()
    }

    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // TODO(spapini): Precompute twiddles.
        // TODO(spapini): Handle small cases.
        let log_size = domain.log_size();
        let fft_log_size = poly.log_size();
        assert!(
            log_size >= fft_log_size,
            "Can only evaluate on larger domains"
        );

        let twiddles = domain_line_twiddles_from_tree(domain, &twiddles.twiddles);

        // Evaluate on a big domains by evaluating on several subdomains.
        let log_subdomains = log_size - fft_log_size;

        // Alllocate the destination buffer without initializing.
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
                    transmute(poly.coeffs.data.as_ptr()),
                    transmute(
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
            BaseFieldVec {
                data: values,
                length: domain.size(),
            },
        )
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        let mut twiddles = vec![1; coset.size()];
        let mut itwiddles = vec![1; coset.size()];

        compute_twiddles_of_first_coset(&coset, &mut twiddles, &mut itwiddles);

        compute_twiddles_of_doubled_cosets(&coset, &mut twiddles, &mut itwiddles);

        // multiply everyone by 2.
        for i in 0..coset.size() {
            twiddles[i] = 2 * twiddles[i];
            itwiddles[i] = 2 * itwiddles[i];
        }
        assert_eq!(twiddles.len(), coset.size());
        assert_eq!(itwiddles.len(), coset.size());

        TwiddleTree {
            root_coset: coset,
            twiddles,
            itwiddles,
        }
    }
}

fn compute_twiddles_of_doubled_cosets(
    coset: &Coset,
    twiddles: &mut Vec<u32>,
    itwiddles: &mut Vec<u32>,
) {
    let mut offset = 1 << (coset.log_size() - 1); // Index where each chunk starts.

    for i in 1..coset.log_size() {
        let chunk_log_size: u32 = coset.log_size() - i - 1;
        let chunk_length = 1 << chunk_log_size;

        let previous_offset = offset - 2 * chunk_length; // index where the previous chunk started

        // use simd for large chunks
        if chunk_length >= N_LANES {
            let iter = 0..(1 << (chunk_log_size - LOG_N_LANES));
            iter.for_each(|k| {
                let next_twiddles: [M31; N_LANES] = (0..N_LANES)
                    .map(|j| {
                        M31::from_u32_unchecked(twiddles[previous_offset + 2 * N_LANES * k + 2 * j])
                    })
                    .collect::<Vec<M31>>()
                    .try_into()
                    .expect("Expected N_LANES elements");
                let mut packed_twiddles: PackedM31 = PackedM31::from_array(next_twiddles);

                packed_twiddles = (packed_twiddles * packed_twiddles).double() - PackedM31::one();

                let packed_itwiddles = packed_twiddles.inverse();

                let packed_twiddles_array = (packed_twiddles).to_array();
                let packed_itwiddles_array = (packed_itwiddles).to_array();
                for j in 0..N_LANES {
                    twiddles[offset + N_LANES * k + j] = packed_twiddles_array[j].0;
                    itwiddles[offset + N_LANES * k + j] = packed_itwiddles_array[j].0;
                }
            });
        } else {
            let iter = 0..(chunk_length);
            iter.for_each(|k| {
                let next_twiddle = CirclePoint::double_x(M31::from_u32_unchecked(
                    twiddles[previous_offset + 2 * k],
                ));

                twiddles[offset + k] = next_twiddle.0;
                itwiddles[offset + k] = next_twiddle.inverse().0;
            });
        }
        offset += 1 << (coset.log_size() - i - 1);
    }
}

// Computes the first chunk of twiddles in bit-reverse order.
fn compute_twiddles_of_first_coset(
    coset: &Coset,
    twiddles: &mut Vec<u32>,
    itwiddles: &mut Vec<u32>,
) {
    let steps = precompute_bit_reverse_steps(&coset);

    // Put first twiddle of first coset.
    let mut first_chunk = vec![coset.at(0)];
    twiddles[0] = coset.at(0).x.0;
    itwiddles[0] = coset.at(0).x.inverse().0;

    // We want the bit reversed first (coset_size / 2) twiddles,
    // so we go through every other element of the coset.
    let mut j = 1;
    for i in (0..((coset.size()) - 2)).step_by(2) {
        let last_twiddle = first_chunk.last().unwrap().clone();
        let step_1 = steps[i.trailing_ones() as usize];
        let step_2 = steps[(i + 1).trailing_ones() as usize];
        let next_twiddle = last_twiddle + step_1 + step_2;

        first_chunk.push(next_twiddle);

        twiddles[j] = next_twiddle.x.0;
        itwiddles[j] = next_twiddle.x.inverse().0;

        j = j + 1;
    }
}

// Computes which integer to add to get the next bit_reversed index.
// This value depends only on the number of trailing ones of the current index.
fn bit_reversed_step_index(trailing_ones: u32, coset_log_size: u32) -> u32 {
    let value_index = (3 << (coset_log_size - 1 - trailing_ones)) & ((1 << (coset_log_size)) - 1);
    value_index
}

// Compute the coset elements needed to get from one element to the next in bit_reverse order.
fn precompute_bit_reverse_steps(coset: &Coset) -> Vec<CirclePoint<M31>> {
    let mut steps = vec![];
    for i in 0..coset.log_size() as usize {
        steps.push(
            coset
                .step
                .mul(bit_reversed_step_index(i as u32, coset.log_size()) as u128),
        );
    }
    steps
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
    fold(cast_slice::<_, BaseField>(&poly.coeffs.data), &mappings)
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::bit_reversed_step_index;
    use crate::core::backend::simd::circle::slow_eval_at_point;
    use crate::core::backend::simd::fft::{CACHED_FFT_LOG_SIZE, MIN_FFT_LOG_SIZE};
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::circle::CirclePoint;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly, PolyOps};
    use crate::core::poly::line::LineDomain;
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
    fn test_precompute_twiddles() {
        let log_size = 6;
        let canonic_coset = CanonicCoset::new(log_size + 1).half_coset();
        let domain = LineDomain::new(canonic_coset);
        let coset = domain.coset();

        let cpu_twiddle_tree = CpuBackend::precompute_twiddles(coset);
        let cpu_twiddles: Vec<u32> = cpu_twiddle_tree
            .twiddles
            .iter()
            .map(|value| 2 * value.0)
            .collect();
        let cpu_itwiddles: Vec<u32> = cpu_twiddle_tree
            .itwiddles
            .iter()
            .map(|value| 2 * value.0)
            .collect();
        let simd_twiddle_tree = SimdBackend::precompute_twiddles(coset);
        let simd_twiddles: Vec<u32> = simd_twiddle_tree.twiddles;
        let simd_itwiddles: Vec<u32> = simd_twiddle_tree.itwiddles;

        // assert each chunk separately
        let mut offset = 0;
        let mut chunk_log_size = log_size;
        for _k in 0..(log_size) {
            chunk_log_size = chunk_log_size - 1;
            let chunk_size = 1 << chunk_log_size;
            assert_eq!(
                cpu_twiddles[offset..(offset + chunk_size)],
                simd_twiddles[offset..(offset + chunk_size)]
            );
            assert_eq!(
                cpu_itwiddles[offset..(offset + chunk_size)],
                simd_itwiddles[offset..(offset + chunk_size)]
            );

            offset = offset + chunk_size;
        }
    }

    #[test]
    fn test_bit_reverse_step_index() {
        assert_eq!(8, bit_reversed_step_index((0 as u32).trailing_ones(), 4));
        assert_eq!(12, bit_reversed_step_index((1 as u32).trailing_ones(), 4));
        assert_eq!(8, bit_reversed_step_index((2 as u32).trailing_ones(), 4));
        assert_eq!(6, bit_reversed_step_index((3 as u32).trailing_ones(), 4));
        assert_eq!(3, bit_reversed_step_index((7 as u32).trailing_ones(), 4));

        assert_eq!(4, bit_reversed_step_index((0 as u32).trailing_ones(), 3));
        assert_eq!(6, bit_reversed_step_index((1 as u32).trailing_ones(), 3));
        assert_eq!(4, bit_reversed_step_index((2 as u32).trailing_ones(), 3));
        assert_eq!(3, bit_reversed_step_index((3 as u32).trailing_ones(), 3));
    }
}
