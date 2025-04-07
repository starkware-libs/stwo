use super::WebBackend;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Col;
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::BitReversedOrder;

// WARNING: This works because they are literally the same object layout.
//
// The only difference is the backend methods.
// When we implement all methods for WebGPU,
// we will no longer need this to convert back/forth.
pub fn convert_web_to_simd_column(col: Col<WebBackend, BaseField>) -> Col<SimdBackend, BaseField> {
    assert_eq!(std::mem::size_of::<WebBackend>(), 0);
    assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
    unsafe { std::mem::transmute(col) }
}

impl Into<CircleEvaluation<WebBackend, BaseField, BitReversedOrder>>
    for CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>
{
    fn into(self) -> CircleEvaluation<WebBackend, BaseField, BitReversedOrder> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl Into<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>
    for CircleEvaluation<WebBackend, BaseField, BitReversedOrder>
{
    fn into(self) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl AsRef<TwiddleTree<SimdBackend>> for TwiddleTree<WebBackend> {
    fn as_ref(&self) -> &TwiddleTree<SimdBackend> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl AsRef<CirclePoly<SimdBackend>> for CirclePoly<WebBackend> {
    fn as_ref(&self) -> &CirclePoly<SimdBackend> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl Into<CirclePoly<WebBackend>> for CirclePoly<SimdBackend> {
    fn into(self) -> CirclePoly<WebBackend> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl Into<TwiddleTree<WebBackend>> for TwiddleTree<SimdBackend> {
    fn into(self) -> TwiddleTree<WebBackend> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}
//

impl PolyOps for WebBackend {
    // The twiddles type is i32, and not BaseField. This is because the fast AVX mul implementation
    //  requires one of the numbers to be shifted left by 1 bit. This is not a reduced
    //  representation of the field.
    type Twiddles = Vec<u32>;

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        SimdBackend::new_canonical_ordered(coset, convert_web_to_simd_column(values)).into()
    }

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        SimdBackend::interpolate(eval.into(), twiddles.as_ref()).into()
    }

    fn eval_at_point(poly: &CirclePoly<Self>, point: CirclePoint<SecureField>) -> SecureField {
        SimdBackend::eval_at_point(poly.as_ref(), point)
    }

    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self> {
        SimdBackend::extend(poly.as_ref(), log_size).into()
    }

    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        SimdBackend::evaluate(poly.as_ref(), domain, twiddles.as_ref()).into()
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        SimdBackend::precompute_twiddles(coset).into()
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::WebBackend;
    use crate::core::backend::simd::circle::slow_eval_at_point;
    use crate::core::backend::simd::fft::{CACHED_FFT_LOG_SIZE, MIN_FFT_LOG_SIZE};
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::circle::CirclePoint;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly, PolyOps};
    use crate::core::poly::{BitReversedOrder, NaturalOrder};

    #[test]
    fn test_interpolate_and_eval() {
        for log_size in MIN_FFT_LOG_SIZE..CACHED_FFT_LOG_SIZE + 4 {
            let domain = CanonicCoset::new(log_size).circle_domain();
            let evaluation = CircleEvaluation::<WebBackend, BaseField, BitReversedOrder>::new(
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
            let evaluation = CircleEvaluation::<WebBackend, BaseField, BitReversedOrder>::new(
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
            let evaluation = CircleEvaluation::<WebBackend, BaseField, NaturalOrder>::new(
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
                CirclePoly::<WebBackend>::new((0..1 << log_size).map(BaseField::from).collect());
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
            let evaluation = CircleEvaluation::<WebBackend, BaseField, NaturalOrder>::new(
                domain,
                (0..1 << log_size).map(BaseField::from).collect(),
            );
            let poly = evaluation.bit_reverse().interpolate();
            let x = rng.gen();
            let y = rng.gen();
            let p = CirclePoint { x, y };

            let eval = PolyOps::eval_at_point(&poly, p);

            assert_eq!(
                eval,
                slow_eval_at_point(&poly.as_ref(), p),
                "log_size = {log_size}"
            );
        }
    }

    #[test]
    fn test_optimized_precompute_twiddles() {
        let coset = CanonicCoset::new(10).half_coset();
        let twiddles = WebBackend::precompute_twiddles(coset);
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
