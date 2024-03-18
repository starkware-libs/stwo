use std::arch::x86_64::_mm512_set1_epi32;

use bytemuck::{cast_slice, Zeroable};
use num_traits::{One, Zero};

use super::fft::{ifft, CACHED_FFT_LOG_SIZE};
use super::m31::PackedBaseField;
use super::{as_cpu_vec, AVX512Backend, VECS_LOG_SIZE};
use crate::core::backend::avx512::fft::rfft;
use crate::core::backend::avx512::BaseFieldVec;
use crate::core::backend::{CPUBackend, Col};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::{ExtensionOf, Field, FieldExpOps};
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::utils::{domain_line_twiddles_from_tree, fold};
use crate::core::poly::BitReversedOrder;

impl AVX512Backend {
    fn twiddle_lows(low_mappings: &[BaseField; 4]) -> PackedBaseField {
        let mut res = [BaseField::zero(); 16];
        let t0 = low_mappings[0];
        let t1 = low_mappings[1];
        let t2 = low_mappings[2];
        let t3 = low_mappings[3];

        res[0] = BaseField::one();
        res[1] = t0;
        res[2] = t1;
        res[3] = t0 * t1;
        res[4] = t2;
        res[5] = t0 * t2;
        res[6] = t1 * t2;
        res[7] = t0 * t1 * t2;
        res[8] = t3;
        res[9] = t0 * t3;
        res[10] = t1 * t3;
        res[11] = t0 * t1 * t3;
        res[12] = t2 * t3;
        res[13] = t0 * t2 * t3;
        res[14] = t1 * t2 * t3;
        res[15] = t0 * t1 * t2 * t3;

        PackedBaseField::from_array(res)
    }

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
        CircleEvaluation::new(eval.domain, Col::<AVX512Backend, _>::from_iter(eval.values))
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

    fn eval_at_point<E: ExtensionOf<BaseField>>(
        poly: &CirclePoly<Self>,
        point: CirclePoint<E>,
    ) -> E {
        // TODO(spapini): Optimize.
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
        fold(cast_slice(&poly.coeffs.data), &mappings)
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
        let mut values = Vec::with_capacity(domain.size() >> VECS_LOG_SIZE);
        for i in 0..(1 << log_subdomains) {
            // The subdomain twiddles are a slice of the large domain twiddles.
            let subdomain_twiddles = (0..(fft_log_size - 1))
                .map(|layer_i| {
                    &twiddles[layer_i]
                        [i << (fft_log_size - 2 - layer_i)..(i + 1) << (fft_log_size - 2 - layer_i)]
                })
                .collect::<Vec<_>>();

            // Copy the coefficients of the polynomial to the values vector.
            values.extend_from_slice(&poly.coeffs.data);

            // FFT inplace on the values chunk.
            unsafe {
                rfft::fft(
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

    fn eval_at_basefield_point(
        poly: &CirclePoly<Self>,
        point: CirclePoint<BaseField>,
    ) -> BaseField {
        let mut mappings = vec![point.y, point.x];
        let mut x = point.x;
        for _ in 2..poly.log_size() {
            x = CirclePoint::double_x(x);
            mappings.push(x);
        }

        // If the polynomial is large, the fft does a transpose in the middle.
        // TODO(Ohad): to avoid complexity for now, we just reverse the mappings, transpose, then
        // reverse back so the original transpose works. Optimize.
        if poly.log_size() as usize > CACHED_FFT_LOG_SIZE {
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

        // 4 lowest mappings produce the first 2^4 twiddles.
        let (map_low, map_high) = mappings.split_at(4);
        let tl = AVX512Backend::twiddle_lows(map_low.try_into().unwrap());

        // Every twiddle is a product of mappings that correspond to '1's in the bit representation
        // of the current index. For every 2^n alligned chunk of 2^n elements, the twiddle
        // array is the same, denoted twiddle_low. Use this to compute sums of (coeff *
        // twiddle_high) mod 2^n, then multiply by twiddle_low, and sum to get the final result.
        let mut sum = PackedBaseField::zeroed();
        for (i, &coeff_chunk) in poly.coeffs.data.iter().enumerate() {
            let cur_twiddle_high = AVX512Backend::twiddle_at(map_high, i);
            let curr_twiddle_broadcast =
                PackedBaseField(unsafe { _mm512_set1_epi32(cur_twiddle_high.0 as i32) });
            sum += coeff_chunk * curr_twiddle_broadcast;
        }

        (sum * tl).pointwise_sum()
    }
}

// #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use crate::core::backend::avx512::fft::{CACHED_FFT_LOG_SIZE, MIN_FFT_LOG_SIZE};
    use crate::core::backend::avx512::AVX512Backend;
    use crate::core::backend::Column;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::{CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps};
    use crate::core::poly::{BitReversedOrder, NaturalOrder};

    #[test]
    fn test_interpolate_and_eval() {
        for log_size in MIN_FFT_LOG_SIZE..(CACHED_FFT_LOG_SIZE + 4) {
            let domain = CanonicCoset::new(log_size as u32).circle_domain();
            let evaluation = CircleEvaluation::<AVX512Backend, _, BitReversedOrder>::new(
                domain,
                (0..(1 << log_size))
                    .map(BaseField::from_u32_unchecked)
                    .collect(),
            );
            let poly = evaluation.clone().interpolate();
            let evaluation2 = poly.evaluate(domain);
            assert_eq!(evaluation.values.to_vec(), evaluation2.values.to_vec());
        }
    }

    #[test]
    fn test_eval_extension() {
        for log_size in MIN_FFT_LOG_SIZE..(CACHED_FFT_LOG_SIZE + 4) {
            let log_size = log_size as u32;
            let domain = CircleDomain::constraint_evaluation_domain(log_size);
            let domain_ext = CircleDomain::constraint_evaluation_domain(log_size + 3);
            let evaluation = CircleEvaluation::<AVX512Backend, _, BitReversedOrder>::new(
                domain,
                (0..(1 << log_size))
                    .map(BaseField::from_u32_unchecked)
                    .collect(),
            );
            let poly = evaluation.clone().interpolate();
            let evaluation2 = poly.evaluate(domain_ext);
            assert_eq!(
                evaluation2.values.to_vec()[..(1 << log_size)],
                evaluation.values.to_vec()
            );
        }
    }

    #[test]
    fn test_eval_at_point() {
        for log_size in MIN_FFT_LOG_SIZE..(CACHED_FFT_LOG_SIZE + 4) {
            let domain = CanonicCoset::new(log_size as u32).circle_domain();
            let evaluation = CircleEvaluation::<AVX512Backend, _, NaturalOrder>::new(
                domain,
                (0..(1 << log_size))
                    .map(BaseField::from_u32_unchecked)
                    .collect(),
            );
            let poly = evaluation.bit_reverse().interpolate();
            for i in [0, 1, 3, 1 << (log_size - 1), 1 << (log_size - 2)] {
                let p = domain.at(i);
                assert_eq!(
                    poly.eval_at_point(p),
                    BaseField::from_u32_unchecked(i as u32),
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

            assert_eq!(eval0.values.to_vec(), eval1.values.to_vec());
        }
    }

    #[test]
    fn test_eval_basefield() {
        use crate::core::backend::avx512::fft::MIN_FFT_LOG_SIZE;

        for log_size in MIN_FFT_LOG_SIZE..(CACHED_FFT_LOG_SIZE + 4) {
            let domain = CanonicCoset::new(log_size as u32).circle_domain();
            let evaluation = CircleEvaluation::<AVX512Backend, _, NaturalOrder>::new(
                domain,
                (0..(1 << log_size))
                    .map(BaseField::from_u32_unchecked)
                    .collect(),
            );
            let poly = evaluation.bit_reverse().interpolate();
            for i in [0, 1, 3, 1 << (log_size - 1), 1 << (log_size - 2)] {
                let p = domain.at(i);
                assert_eq!(
                    <AVX512Backend as PolyOps>::eval_at_basefield_point(&poly, p),
                    BaseField::from_u32_unchecked(i as u32),
                    "log_size = {log_size} i = {i}"
                );
            }
        }
    }
}
