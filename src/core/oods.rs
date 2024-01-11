use super::air::Mask;
use super::circle::{CirclePoint, CirclePointIndex};
use super::constraints::{point_vanishing, EvalByEvaluation, EvalByPoly, PolyOracle};
use super::fields::m31::BaseField;
use super::fields::qm31::QM31;
use super::fields::ExtensionOf;
use super::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly, PointMapping};

// TODO(AlonH): Consider duplicating function instead of using generics (check performance).
/// Evaluate the linear combination of the OODS quotients for the given values over the whole
/// domain. The evaluation is split into the evaluation over the half coset and the evaluation over
/// its conjugate.
pub fn get_oods_quotient_combination<
    F: ExtensionOf<BaseField>,
    EF: ExtensionOf<BaseField> + ExtensionOf<F>,
>(
    random_coeff: EF,
    oods_mapping: &PointMapping<EF>,
    eval: &CircleEvaluation<F>,
) -> CircleEvaluation<EF> {
    let mut values = Vec::with_capacity(eval.domain.size());
    for p_ind in eval.domain.iter_indices() {
        values.push(eval_combined_oods_quotient_point(
            random_coeff,
            oods_mapping,
            EvalByEvaluation::new(p_ind, eval),
        ));
    }
    CircleEvaluation::new(eval.domain, values)
}

/// Returns the value of the linear combination of quotients for the OODS points.
pub fn eval_combined_oods_quotient_point<
    F: ExtensionOf<BaseField>,
    EF: ExtensionOf<BaseField> + ExtensionOf<F>,
>(
    random_coeff: EF,
    oods_mapping: &PointMapping<EF>,
    eval: impl PolyOracle<F>,
) -> EF {
    let mut quotient_combination_value = EF::zero();
    let mut coeff = EF::one();
    for (oods_point, oods_value) in oods_mapping.iter() {
        quotient_combination_value +=
            coeff * eval_oods_quotient_point(*oods_point, *oods_value, eval);
        coeff *= random_coeff;
    }
    quotient_combination_value
}

// TODO(AlonH): Add a function that computes the OODS quotients for the whole domain (batch
// inversed).
/// Returns the quotient value for the OODS point.
pub fn eval_oods_quotient_point<
    F: ExtensionOf<BaseField>,
    EF: ExtensionOf<BaseField> + ExtensionOf<F>,
>(
    oods_point: CirclePoint<EF>,
    oods_value: EF,
    eval: impl PolyOracle<F>,
) -> EF {
    let num = -oods_value + eval.get_at(CirclePointIndex(0));
    let denom: EF = point_vanishing(oods_point, eval.point().into_ef());
    num / denom
}

/// Returns the mask values for the OODS point.
pub fn get_oods_values(
    mask: &Mask,
    oods_point: CirclePoint<QM31>,
    trace_domains: &[CanonicCoset],
    trace_polys: &[CirclePoly<BaseField>],
) -> PointMapping<QM31> {
    let mut oods_evals = Vec::with_capacity(trace_polys.len());
    for poly in trace_polys {
        oods_evals.push(EvalByPoly {
            point: oods_point,
            poly,
        });
    }
    mask.get_evaluation(trace_domains, &oods_evals[..])
}
