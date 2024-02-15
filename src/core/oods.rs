use std::iter::zip;

use super::air::Component;
use super::circle::{CirclePoint, CirclePointIndex};
use super::constraints::{
    complex_conjugate_line, pair_vanishing, point_vanishing, EvalByEvaluation, PolyOracle,
};
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::fields::ComplexConjugate;
use super::fri::CirclePolyDegreeBound;
use super::poly::circle::CircleEvaluation;
use super::poly::{BitReversedOrder, NaturalOrder};

/// Evaluates the OODS quotient polynomial on a single point.
pub fn eval_oods_quotient_point(
    oods_point: CirclePoint<SecureField>,
    oods_value: SecureField,
    eval: impl PolyOracle<SecureField>,
) -> SecureField {
    let num = eval.get_at(CirclePointIndex(0)) - oods_value;
    let denom: SecureField = point_vanishing(oods_point, eval.point());
    num / denom
}

/// Evaluates the pair OODS quotient polynomial on a single point.
/// See `get_pair_oods_quotient` for more details.
pub fn eval_pair_oods_quotient_point(
    oods_point: CirclePoint<SecureField>,
    oods_value: SecureField,
    eval: impl PolyOracle<BaseField>,
) -> SecureField {
    let num = eval.get_at(CirclePointIndex(0))
        - complex_conjugate_line(oods_point, oods_value, eval.point());
    let denom = pair_vanishing(
        oods_point,
        oods_point.complex_conjugate(),
        eval.point().into_ef(),
    );
    num / denom
}

/// Returns the OODS quotient polynomial evaluation over the whole domain.
/// Note the resulting polynomial's highest monomial might increase by one.
pub fn get_oods_quotient(
    oods_point: CirclePoint<SecureField>,
    oods_value: SecureField,
    eval: &CircleEvaluation<SecureField, BitReversedOrder>,
) -> CircleEvaluation<SecureField, NaturalOrder> {
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

/// Returns the pair OODS quotient (i.e quotienting out both the oods point and its complex
/// conjugate) polynomial evaluation over the whole domain. Used in case we don't want the highest
/// monomial of the resulting quotient polynomial to increase which might take it out of the fft
/// space.
pub fn get_pair_oods_quotient(
    oods_point: CirclePoint<SecureField>,
    oods_value: SecureField,
    eval: &CircleEvaluation<BaseField, BitReversedOrder>,
) -> CircleEvaluation<SecureField, NaturalOrder> {
    let mut values = Vec::with_capacity(eval.domain.size());
    for p_ind in eval.domain.iter_indices() {
        values.push(eval_pair_oods_quotient_point(
            oods_point,
            oods_value,
            EvalByEvaluation::new(p_ind, eval),
        ));
    }
    CircleEvaluation::new(eval.domain, values)
}

pub fn quotient_log_bounds(component: impl Component) -> Vec<CirclePolyDegreeBound> {
    zip(
        component.mask().iter(),
        &component.trace_log_degree_bounds(),
    )
    .flat_map(|(trace_points, trace_bound)| {
        trace_points
            .iter()
            .map(|_| CirclePolyDegreeBound::new(*trace_bound))
            .collect::<Vec<_>>()
    })
    .collect()
}
