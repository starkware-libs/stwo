use super::fft::ifft;
use super::AVX512Backend;
use crate::core::backend::avx512::fft::rfft;
use crate::core::backend::{CPUBackend, Col, Column, FieldOps};
use crate::core::fields::m31::BaseField;
use crate::core::fields::Field;
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};

impl PolyOps<BaseField> for AVX512Backend {
    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField> {
        // TODO(spapini): Optimize.
        let eval = CPUBackend::new_canonical_ordered(coset, values.to_vec());
        CircleEvaluation::new(eval.domain, Col::<AVX512Backend, _>::from_iter(eval.values))
    }

    fn interpolate(eval: CircleEvaluation<Self, BaseField>) -> CirclePoly<Self, BaseField> {
        let mut values = eval.values;

        // TODO(spapini): Precompute twiddles.
        let twiddles = ifft::get_itwiddle_dbls(eval.domain);
        // TODO(spapini): Remove.
        AVX512Backend::bit_reverse_column(&mut values);
        // TODO(spapini): Handle small cases.
        let log_size = values.length.ilog2();

        unsafe {
            ifft::ifft(
                std::mem::transmute(values.data.as_mut_ptr()),
                &twiddles[1..],
                log_size as usize,
            );
        }

        // TODO(spapini): Fuse this multiplication / rotation.
        let inv = BaseField::from_u32_unchecked(eval.domain.size() as u32).inverse();
        for x in values.data.iter_mut() {
            for y in x.0.iter_mut() {
                *y = BaseField::from(y.0) * inv;
            }
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
    ) -> CircleEvaluation<Self, BaseField> {
        let mut values = poly.coeffs.clone();

        // TODO(spapini): Precompute twiddles.
        let twiddles = rfft::get_twiddle_dbls(domain);
        // TODO(spapini): Handle small cases.
        let log_size = values.length.ilog2();

        unsafe {
            rfft::fft(
                std::mem::transmute(values.data.as_mut_ptr()),
                &twiddles[1..],
                log_size as usize,
            );
        }

        // TODO(spapini): Remove.
        AVX512Backend::bit_reverse_column(&mut values);

        CircleEvaluation::new(domain, values)
    }

    fn extend(_poly: &CirclePoly<Self, BaseField>, _log_size: u32) -> CirclePoly<Self, BaseField> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::core::backend::avx512::AVX512Backend;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};

    #[test]
    fn test_interpolate_and_eval() {
        const LOG_SIZE: u32 = 6;
        let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let evaluation = CircleEvaluation::<AVX512Backend, _>::new(
            domain,
            (0..(1 << LOG_SIZE))
                .map(BaseField::from_u32_unchecked)
                .collect(),
        );
        let poly = evaluation.clone().interpolate();
        let evaluation2 = poly.evaluate(domain);
        assert_eq!(evaluation.values, evaluation2.values);
    }
}
