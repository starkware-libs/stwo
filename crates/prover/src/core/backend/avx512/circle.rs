use bytemuck::{cast_slice, Zeroable};
use num_traits::One;

use super::fft::{ifft, CACHED_FFT_LOG_SIZE};
use super::m31::PackedBaseField;
use super::qm31::PackedSecureField;
use super::{as_cpu_vec, AVX512Backend, K_BLOCK_SIZE, VECS_LOG_SIZE};
use crate::core::backend::avx512::fft::rfft;
use crate::core::backend::avx512::BaseFieldVec;
use crate::core::backend::{CPUBackend, Col};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{Field, FieldExpOps};
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::utils::{domain_line_twiddles_from_tree, fold};
use crate::core::poly::BitReversedOrder;

impl AVX512Backend {
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
        if log_size as usize > CACHED_FFT_LOG_SIZE {
            mappings.reverse();
            let n = mappings.len();
            let n0 = (n - VECS_LOG_SIZE) / 2;
            let n1 = (n - VECS_LOG_SIZE + 1) / 2;
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
impl PolyOps for AVX512Backend {
    // The twiddles type is i32, and not BaseField. This is because the fast AVX mul implementation
    //  requries one of the numbers to be shifted left by 1 bit. This is not a reduced
    //  representation of the field.
    type Twiddles = Vec<i32>;

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // TODO(spapini): Optimize.
        let eval = CPUBackend::new_canonical_ordered(coset, as_cpu_vec(values));
        CircleEvaluation::new(
            eval.domain,
            Col::<AVX512Backend, BaseField>::from_iter(eval.values),
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
                std::mem::transmute(values.data.as_mut_ptr()),
                &twiddles,
                log_size as usize,
            );
        }

        // TODO(spapini): Fuse this multiplication / rotation.
        let inv = BaseField::from_u32_unchecked(eval.domain.size() as u32).inverse();
        let inv = PackedBaseField::from_array([inv; 16]);
        for x in values.data.iter_mut() {
            *x *= inv;
        }

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
        for (i, coeff_chunk) in poly.coeffs.data.array_chunks::<K_BLOCK_SIZE>().enumerate() {
            // For every chunk of 2 ^ 4 * 2 ^ 4 = 2 ^ 8 elements, the twiddle high is the same.
            // Multiply it by every mid twiddle factor to get the factors for the current chunk.
            let high_twiddle_factors =
                (PackedSecureField::broadcast(twiddle_high) * twiddle_mids).to_array();

            // Sum the coefficients multiplied by each corrseponsing twiddle. Result is effectivley
            // an array[16] where the value at index 'i' is the sum of all coefficients at indices
            // that are i mod 16.
            for (&packed_coeffs, &mid_twiddle) in
                coeff_chunk.iter().zip(high_twiddle_factors.iter())
            {
                sum += PackedSecureField::broadcast(mid_twiddle).mul_packed_m31(packed_coeffs);
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
        let log_size = domain.log_size() as usize;
        let fft_log_size = poly.log_size() as usize;
        assert!(
            log_size >= fft_log_size,
            "Can only evaluate on larger domains"
        );

        let twiddles = domain_line_twiddles_from_tree(domain, &twiddles.twiddles);

        // Evaluate on a big domains by evaluating on several subdomains.
        let log_subdomains = log_size - fft_log_size;

        // Alllocate the destination buffer without initializing.
        let mut values = Vec::with_capacity(domain.size() >> VECS_LOG_SIZE);
        #[allow(clippy::uninit_vec)]
        unsafe {
            values.set_len(domain.size() >> VECS_LOG_SIZE)
        };

        for i in 0..(1 << log_subdomains) {
            // The subdomain twiddles are a slice of the large domain twiddles.
            let subdomain_twiddles = (0..(fft_log_size - 1))
                .map(|layer_i| {
                    &twiddles[layer_i]
                        [i << (fft_log_size - 2 - layer_i)..(i + 1) << (fft_log_size - 2 - layer_i)]
                })
                .collect::<Vec<_>>();

            // FFT from the coefficients buffer to the values chunk.
            unsafe {
                rfft::fft(
                    std::mem::transmute(poly.coeffs.data.as_ptr()),
                    std::mem::transmute(
                        values[i << (fft_log_size - VECS_LOG_SIZE)
                            ..(i + 1) << (fft_log_size - VECS_LOG_SIZE)]
                            .as_mut_ptr(),
                    ),
                    &subdomain_twiddles,
                    fft_log_size,
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
        let mut twiddles = Vec::with_capacity(coset.size());
        let mut itwiddles = Vec::with_capacity(coset.size());

        // TODO(spapini): Optimize.
        for layer in &rfft::get_twiddle_dbls(coset) {
            twiddles.extend(layer);
        }
        // Pad by any value, to make the size a power of 2.
        twiddles.push(1);
        assert_eq!(twiddles.len(), coset.size());
        for layer in &ifft::get_itwiddle_dbls(coset) {
            itwiddles.extend(layer);
        }
        // Pad by any value, to make the size a power of 2.
        itwiddles.push(1);
        assert_eq!(itwiddles.len(), coset.size());

        TwiddleTree {
            root_coset: coset,
            twiddles,
            itwiddles,
        }
    }
}

fn slow_eval_at_point(
    poly: &CirclePoly<AVX512Backend>,
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
    if poly.log_size() as usize > CACHED_FFT_LOG_SIZE {
        let n = mappings.len();
        let n0 = (n - VECS_LOG_SIZE) / 2;
        let n1 = (n - VECS_LOG_SIZE + 1) / 2;
        let (ab, c) = mappings.split_at_mut(n1);
        let (a, _b) = ab.split_at_mut(n0);
        // Swap content of a,c.
        a.swap_with_slice(&mut c[0..n0]);
    }
    fold(cast_slice::<_, BaseField>(&poly.coeffs.data), &mappings)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::backend::avx512::circle::slow_eval_at_point;
    use crate::core::backend::avx512::fft::{CACHED_FFT_LOG_SIZE, MIN_FFT_LOG_SIZE};
    use crate::core::backend::avx512::AVX512Backend;
    use crate::core::backend::Column;
    use crate::core::circle::CirclePoint;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly, PolyOps};
    use crate::core::poly::{BitReversedOrder, NaturalOrder};
    use crate::qm31;

    #[test]
    fn test_interpolate_and_eval() {
        for log_size in MIN_FFT_LOG_SIZE..(CACHED_FFT_LOG_SIZE + 4) {
            let domain = CanonicCoset::new(log_size as u32).circle_domain();
            let evaluation = CircleEvaluation::<AVX512Backend, BaseField, BitReversedOrder>::new(
                domain,
                (0..(1 << log_size))
                    .map(BaseField::from_u32_unchecked)
                    .collect(),
            );
            let poly = evaluation.clone().interpolate();
            let evaluation2 = poly.evaluate(domain);
            assert_eq!(evaluation.values.to_cpu(), evaluation2.values.to_cpu());
        }
    }

    #[test]
    fn test_eval_extension() {
        for log_size in MIN_FFT_LOG_SIZE..(CACHED_FFT_LOG_SIZE + 4) {
            let log_size = log_size as u32;
            let domain = CanonicCoset::new(log_size).circle_domain();
            let domain_ext = CanonicCoset::new(log_size + 3).circle_domain();
            let evaluation = CircleEvaluation::<AVX512Backend, BaseField, BitReversedOrder>::new(
                domain,
                (0..(1 << log_size))
                    .map(BaseField::from_u32_unchecked)
                    .collect(),
            );
            let poly = evaluation.clone().interpolate();
            let evaluation2 = poly.evaluate(domain_ext);
            let poly2 = evaluation2.interpolate();
            assert_eq!(
                poly.extend(log_size + 3).coeffs.to_cpu(),
                poly2.coeffs.to_cpu()
            );
        }
    }

    #[test]
    fn test_eval_at_point() {
        for log_size in MIN_FFT_LOG_SIZE..(CACHED_FFT_LOG_SIZE + 4) {
            let domain = CanonicCoset::new(log_size as u32).circle_domain();
            let evaluation = CircleEvaluation::<AVX512Backend, BaseField, NaturalOrder>::new(
                domain,
                (0..(1 << log_size))
                    .map(BaseField::from_u32_unchecked)
                    .collect(),
            );
            let poly = evaluation.bit_reverse().interpolate();
            for i in [0, 1, 3, 1 << (log_size - 1), 1 << (log_size - 2)] {
                let p = domain.at(i);
                assert_eq!(
                    poly.eval_at_point(p.into_ef()),
                    BaseField::from_u32_unchecked(i as u32).into(),
                    "log_size = {log_size} i = {i}"
                );
            }
        }
    }

    #[test]
    fn test_circle_poly_extend() {
        for log_size in MIN_FFT_LOG_SIZE..(CACHED_FFT_LOG_SIZE + 2) {
            let log_size = log_size as u32;
            let poly = CirclePoly::<AVX512Backend>::new(
                (0..(1 << log_size))
                    .map(BaseField::from_u32_unchecked)
                    .collect(),
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
        use crate::core::backend::avx512::fft::MIN_FFT_LOG_SIZE;
        let mut rng = SmallRng::seed_from_u64(0);

        for log_size in MIN_FFT_LOG_SIZE..(CACHED_FFT_LOG_SIZE + 2) {
            let domain = CanonicCoset::new(log_size as u32).circle_domain();
            let evaluation = CircleEvaluation::<AVX512Backend, BaseField, NaturalOrder>::new(
                domain,
                (0..(1 << log_size))
                    .map(BaseField::from_u32_unchecked)
                    .collect(),
            );
            let poly = evaluation.bit_reverse().interpolate();

            let x: SecureField = rng.gen();
            let y: SecureField = rng.gen();

            let p = CirclePoint { x, y };

            assert_eq!(
                <AVX512Backend as PolyOps>::eval_at_point(&poly, p),
                slow_eval_at_point(&poly, p),
                "log_size = {log_size}"
            );

            println!(
                "log_size = {log_size} passed, eval{}",
                <AVX512Backend as PolyOps>::eval_at_point(&poly, p)
            );
        }
    }
}
