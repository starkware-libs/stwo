use super::air::Mask;
use super::channel::Blake2sChannel;
use super::circle::{CirclePoint, CirclePointIndex};
use super::constraints::{point_vanishing, EvalByEvaluation, EvalByPoly, PolyOracle};
use super::fields::m31::BaseField;
use super::fields::qm31::QM31;
use super::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly, PointMapping};

/// Evaluates the OODS boundary polynomial at the trace point.
pub fn eval_mask_quotient_point(
    trace: impl PolyOracle<BaseField>,
    oods_point: CirclePoint<QM31>,
    oods_value: QM31,
) -> QM31 {
    let num = trace.get_at(CirclePointIndex(0)) - oods_value;
    let denom: QM31 = point_vanishing(oods_point, trace.point().into_ef());
    num / denom
}

/// Evaluate the quotient for the OODS point over the whole domain.
pub fn get_mask_quotient(
    point: CirclePoint<QM31>,
    value: QM31,
    eval: &CircleEvaluation<BaseField>,
) -> Vec<QM31> {
    let mut values = Vec::with_capacity(eval.domain.size());
    for p_ind in eval.domain.iter_indices() {
        values.push(eval_mask_quotient_point(
            EvalByEvaluation {
                offset: p_ind,
                eval,
            },
            point,
            value,
        ));
    }
    values
}

/// Returns the mask values for the OODS point.
pub fn get_oods_values(
    mask: Mask,
    channel: &mut Blake2sChannel,
    trace_domains: &[CanonicCoset],
    trace_polys: &[CirclePoly<BaseField>],
) -> PointMapping<QM31> {
    let oods_point = CirclePoint::<QM31>::get_random_point(channel);
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
