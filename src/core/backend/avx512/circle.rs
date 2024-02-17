use super::{as_cpu_vec, AVX512Backend};
use crate::core::backend::CPUBackend;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::{Col, ExtensionOf};
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};

impl PolyOps<BaseField> for AVX512Backend {
    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField> {
        let eval = CPUBackend::new_canonical_ordered(coset, as_cpu_vec(values));
        CircleEvaluation::new(eval.domain, Col::<AVX512Backend, _>::from_iter(eval.values))
    }

    fn interpolate(eval: CircleEvaluation<Self, BaseField>) -> CirclePoly<Self, BaseField> {
        let cpu_eval = CircleEvaluation::<CPUBackend, _>::new(eval.domain, as_cpu_vec(eval.values));
        let cpu_poly = cpu_eval.interpolate();
        CirclePoly::new(Col::<AVX512Backend, _>::from_iter(cpu_poly.coeffs))
    }

    fn eval_at_point<E: ExtensionOf<BaseField>>(
        poly: &CirclePoly<Self, BaseField>,
        point: CirclePoint<E>,
    ) -> E {
        // TODO(spapini): Unnecessary allocation here.
        let cpu_poly = CirclePoly::<CPUBackend, _>::new(as_cpu_vec(poly.coeffs.clone()));
        cpu_poly.eval_at_point(point)
    }

    fn evaluate(
        poly: &CirclePoly<Self, BaseField>,
        domain: CircleDomain,
    ) -> CircleEvaluation<Self, BaseField> {
        let cpu_poly = CirclePoly::<CPUBackend, _>::new(as_cpu_vec(poly.coeffs.clone()));
        let cpu_eval = cpu_poly.evaluate(domain);
        CircleEvaluation::new(
            cpu_eval.domain,
            Col::<AVX512Backend, _>::from_iter(cpu_eval.values),
        )
    }

    fn extend(poly: &CirclePoly<Self, BaseField>, log_size: u32) -> CirclePoly<Self, BaseField> {
        let cpu_poly = CirclePoly::<CPUBackend, _>::new(as_cpu_vec(poly.coeffs.clone()));
        CirclePoly::new(Col::<AVX512Backend, _>::from_iter(
            cpu_poly.extend(log_size).coeffs,
        ))
    }
}
