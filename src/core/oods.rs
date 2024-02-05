use super::air::Mask;
use super::circle::{CirclePoint, CirclePointIndex};
use super::constraints::{point_vanishing, EvalByEvaluation, EvalByPoly, PolyOracle};
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::fields::ExtensionOf;
use super::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly, PointMapping};

/// Evaluates the OODS quotient polynomial on a single point.
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
/// Returns the OODS quotient polynomial evaluation over the whole domain.
pub fn get_oods_quotient<F: ExtensionOf<BaseField>, EF: ExtensionOf<BaseField> + ExtensionOf<F>>(
    oods_point: CirclePoint<EF>,
    oods_value: EF,
    eval: &CircleEvaluation<F>,
) -> CircleEvaluation<EF> {
    let mut values = Vec::with_capacity(eval.domain.size());
    for p_ind in eval.domain.iter_indices() {
        values.push(eval_oods_quotient_point(
            oods_point,
            oods_value,
            EvalByEvaluation::new(p_ind, eval),
        ));
    }
    CircleEvaluation::new(eval.domain, values)
}

/// Returns the evaluations of the trace mask at the OODS point.
pub fn get_oods_values(
    mask: &Mask,
    oods_point: CirclePoint<SecureField>,
    trace_domains: &[CanonicCoset],
    trace_polys: &[CirclePoly<BaseField>],
) -> PointMapping<SecureField> {
    let mut oods_evals = Vec::with_capacity(trace_polys.len());
    for poly in trace_polys {
        oods_evals.push(EvalByPoly {
            point: oods_point,
            poly,
        });
    }
    mask.get_evaluation(trace_domains, &oods_evals[..])
}

// TODO(AlonH): Consider refactoring and using this function in `get_oods_values`.
/// Returns the OODS evaluation points for the mask.
pub fn get_oods_points(
    mask: &Mask,
    oods_point: CirclePoint<SecureField>,
    trace_domains: &[CanonicCoset],
) -> Vec<CirclePoint<SecureField>> {
    let mask_offsets = mask.get_point_indices(trace_domains);
    mask_offsets
        .iter()
        .map(|offset| oods_point + offset.to_point().into_ef())
        .collect()
}
