use bytemuck::cast_slice;
use itertools::Itertools;

use super::fft::{ifft, CACHED_FFT_LOG_SIZE};
use super::m31::PackedBaseField;
use super::{as_cpu_vec, AVX512Backend, VECS_LOG_SIZE};
use crate::core::backend::avx512::fft::rfft;
use crate::core::backend::avx512::BaseFieldVec;
use crate::core::backend::{CPUBackend, Col};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::{ExtensionOf, FieldExpOps};
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::utils::fold;
use crate::core::poly::BitReversedOrder;

// TODO(spapini): Everything is returned in redundant representation, where values can also be P.
// Decide if and when it's ok and what to do if it's not.
impl PolyOps for AVX512Backend {
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
        itwiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        let mut values = eval.values;
        let log_size = values.length.ilog2();

        let twiddle_buffer = &itwiddles.itwiddles;
        let twiddles = (0..eval.domain.half_coset.log_size())
            .map(|i| {
                let len = 1 << i;
                &twiddle_buffer[twiddle_buffer.len() - len * 2..twiddle_buffer.len() - len]
            })
            .rev()
            .collect_vec();

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

        let twiddle_buffer = &twiddles.twiddles;
        let twiddles = (0..domain.half_coset.log_size())
            .map(|i| {
                let len = 1 << i;
                &twiddle_buffer[twiddle_buffer.len() - len * 2..twiddle_buffer.len() - len]
            })
            .rev()
            .collect_vec();

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

    type Twiddles = Vec<i32>;
    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        let mut twiddles = Vec::with_capacity(coset.size());
        let mut itwiddles = Vec::with_capacity(coset.size());

        // Optimize.
        for layer in &rfft::get_twiddle_dbls(coset)[1..] {
            twiddles.extend(layer);
        }
        twiddles.push(2);
        assert_eq!(twiddles.len(), coset.size());
        for layer in &ifft::get_itwiddle_dbls(coset)[1..] {
            itwiddles.extend(layer);
        }
        itwiddles.push(2);
        assert_eq!(itwiddles.len(), coset.size());

        TwiddleTree {
            root_coset: coset,
            twiddles,
            itwiddles,
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use crate::core::backend::avx512::fft::{CACHED_FFT_LOG_SIZE, MIN_FFT_LOG_SIZE};
    use crate::core::backend::avx512::AVX512Backend;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::Column;
    use crate::core::poly::circle::{CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly};
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
}
