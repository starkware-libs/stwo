use super::air::Mask;
use super::circle::{CirclePoint, CirclePointIndex};
use super::constraints::{point_vanishing, EvalByEvaluation, EvalByPoly, PolyOracle};
use super::fields::m31::BaseField;
use super::fields::qm31::QM31;
use super::fields::ExtensionOf;
use super::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly, PointMapping};

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

// TODO(AlonH): Consider duplicating function instead of using generics (check performance).
/// Evaluate the quotient for the OODS point over the whole domain.
pub fn get_oods_quotient<F: ExtensionOf<BaseField>, EF: ExtensionOf<BaseField> + ExtensionOf<F>>(
    oods_point: CirclePoint<EF>,
    oods_value: EF,
    eval: &CircleEvaluation<F>,
) -> Vec<EF> {
    let mut values = Vec::with_capacity(eval.domain.size());
    for p_ind in eval.domain.iter_indices() {
        values.push(eval_oods_quotient_point(
            oods_point,
            oods_value,
            EvalByEvaluation::new(p_ind, eval),
        ));
    }
    values
}

/// Returns the mask values for the OODS point.
pub fn get_oods_values(
    mask: Mask,
    oods_point: CirclePoint<QM31>,
    trace_domains: &[CanonicCoset],
    trace_polys: &[CirclePoly<BaseField>],
) -> PointMapping<QM31> {
    let mut oods_evals = Vec::with_capacity(trace_polys.len());
    let mut oods_conjugate_evals = Vec::with_capacity(trace_polys.len());
    for poly in trace_polys {
        oods_evals.push(EvalByPoly {
            point: oods_point,
            poly,
        });
        oods_conjugate_evals.push(EvalByPoly {
            point: -oods_point,
            poly,
        });
    }
    mask.get_evaluation(trace_domains, &oods_evals[..], &oods_conjugate_evals[..])
}
