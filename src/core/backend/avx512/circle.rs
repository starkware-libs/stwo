use super::fft::ifft;
use super::m31::PackedBaseField;
use super::{as_cpu_vec, AVX512Backend};
use crate::core::backend::avx512::fft::rfft;
use crate::core::backend::avx512::{BaseFieldVec, VECS_LOG_SIZE};
use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::{Col, FieldExpOps};
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};
use crate::core::poly::BitReversedOrder;

// TODO(spapini): Everything is returned in redundant representation, where values can also be P.
// Decide if and when it's ok and what to do if it's not.
impl PolyOps<BaseField> for AVX512Backend {
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
    ) -> CirclePoly<Self, BaseField> {
        let mut values = eval.values;
        let log_size = values.length.ilog2();

        // TODO(spapini): Precompute twiddles.
        let twiddle_dbls = ifft::get_itwiddle_dbls(eval.domain);
        // TODO(spapini): Handle small cases.

        // Safe because [PackedBaseField] is aligned on 64 bytes.
        unsafe {
            ifft::ifft(
                std::mem::transmute(values.data.as_mut_ptr()),
                &twiddle_dbls[1..]
                    .iter()
                    .map(|x| x.as_slice())
                    .collect::<Vec<_>>(),
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

    fn eval_at_point<E: crate::core::fields::ExtensionOf<BaseField>>(
        _poly: &CirclePoly<Self, BaseField>,
        _point: crate::core::circle::CirclePoint<E>,
    ) -> E {
        todo!()
    }

    fn evaluate(
        poly: &CirclePoly<Self, BaseField>,
        domain: CircleDomain,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // TODO(spapini): Precompute twiddles.
        // TODO(spapini): Handle small cases.
        let log_size = domain.log_size() as usize;
        let fft_log_size = poly.log_size() as usize;
        assert!(
            log_size >= fft_log_size,
            "Can only evaluate on larger domains"
        );

        let twiddles = rfft::get_twiddle_dbls(domain);

        // Evaluate on a big domains by evaluating on several subdomains.
        let log_subdomains = log_size - fft_log_size;
        let mut values = Vec::with_capacity(domain.size() >> VECS_LOG_SIZE);
        for i in 0..(1 << log_subdomains) {
            // The subdomain twiddles are a slice of the large domain twiddles.
            let subdomain_twiddles = (1..fft_log_size)
                .map(|layer_i| {
                    &twiddles[layer_i]
                        [i << (fft_log_size - 1 - layer_i)..(i + 1) << (fft_log_size - 1 - layer_i)]
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

    fn extend(poly: &CirclePoly<Self, BaseField>, log_size: u32) -> CirclePoly<Self, BaseField> {
        // TODO(spapini): Optimize or get rid of extend.
        poly.evaluate(CanonicCoset::new(log_size).circle_domain())
            .interpolate()
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use crate::core::backend::avx512::AVX512Backend;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::Column;
    use crate::core::poly::circle::{CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly};
    use crate::core::poly::BitReversedOrder;

    #[test]
    fn test_interpolate_and_eval() {
        const LOG_SIZE: u32 = 6;
        let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let evaluation = CircleEvaluation::<AVX512Backend, _, BitReversedOrder>::new(
            domain,
            (0..(1 << LOG_SIZE))
                .map(BaseField::from_u32_unchecked)
                .collect(),
        );
        let poly = evaluation.clone().interpolate();
        let evaluation2 = poly.evaluate(domain);
        assert_eq!(evaluation.values.to_vec(), evaluation2.values.to_vec());
    }

    #[test]
    fn test_eval_extension() {
        const LOG_SIZE: u32 = 6;
        let domain = CircleDomain::constraint_evaluation_domain(LOG_SIZE);
        let domain_ext = CircleDomain::constraint_evaluation_domain(LOG_SIZE + 3);
        let evaluation = CircleEvaluation::<AVX512Backend, _, BitReversedOrder>::new(
            domain,
            (0..(1 << LOG_SIZE))
                .map(BaseField::from_u32_unchecked)
                .collect(),
        );
        let poly = evaluation.clone().interpolate();
        let evaluation2 = poly.evaluate(domain_ext);
        for i in 0..(1 << LOG_SIZE) {
            assert_eq!(evaluation2.values.at(i), evaluation.values.at(i));
        }
    }

    #[test]
    fn test_circle_poly_extend() {
        let poly = CirclePoly::<AVX512Backend, _>::new(
            (0..(1 << 6)).map(BaseField::from_u32_unchecked).collect(),
        );
        let eval0 = poly.evaluate(CanonicCoset::new(8).circle_domain());
        let eval1 = poly
            .extend(8)
            .evaluate(CanonicCoset::new(8).circle_domain());

        // Compare.
        for i in 0..eval0.values.len() {
            assert_eq!(eval0.values.at(i), eval1.values.at(i));
        }
    }
}
