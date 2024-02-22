use std::iter::zip;

use itertools::enumerate;

use super::air::Component;
use super::backend::cpu::CPUCircleEvaluation;
use super::backend::Backend;
use super::circle::CirclePoint;
use super::constraints::{complex_conjugate_line, pair_vanishing, point_vanishing};
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::fields::ComplexConjugate;
use super::fri::CirclePolyDegreeBound;
use super::poly::circle::CircleEvaluation;
use super::poly::{BitReversedOrder, NaturalOrder};
use super::utils::bit_reverse_index;

/// Evaluates the OODS quotient polynomial on a single point.
pub fn eval_oods_quotient_at_point(
    point: CirclePoint<BaseField>,
    value: SecureField,
    oods_point: CirclePoint<SecureField>,
    oods_value: SecureField,
) -> SecureField {
    let num = value - oods_value;
    let denom: SecureField = point_vanishing(oods_point, point);
    num / denom
}

/// Evaluates the pair OODS quotient polynomial on a single point.
/// See `get_pair_oods_quotient` for more details.
pub fn eval_pair_oods_quotient_at_point(
    point: CirclePoint<BaseField>,
    value: BaseField,
    oods_point: CirclePoint<SecureField>,
    oods_value: SecureField,
) -> SecureField {
    let num = value - complex_conjugate_line(oods_point, oods_value, point);
    let denom = pair_vanishing(oods_point, oods_point.complex_conjugate(), point.into_ef());
    num / denom
}

/// Returns the OODS quotient polynomial evaluation over the whole domain.
/// Note the resulting polynomial's highest monomial might increase by one.
pub fn get_oods_quotient(
    oods_point: CirclePoint<SecureField>,
    oods_value: SecureField,
    eval: &CPUCircleEvaluation<SecureField, BitReversedOrder>,
) -> CPUCircleEvaluation<SecureField, NaturalOrder> {
    let mut values = Vec::with_capacity(eval.domain.size());
    for (i, point) in enumerate(eval.domain.iter()) {
        let index = bit_reverse_index(i, eval.domain.log_size());
        values.push(eval_oods_quotient_at_point(
            point,
            eval.values[index],
            oods_point,
            oods_value,
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
    eval: &CPUCircleEvaluation<BaseField, BitReversedOrder>,
) -> CPUCircleEvaluation<SecureField, NaturalOrder> {
    let mut values = Vec::with_capacity(eval.domain.size());
    for (i, point) in enumerate(eval.domain.iter()) {
        let index = bit_reverse_index(i, eval.domain.log_size());
        values.push(eval_pair_oods_quotient_at_point(
            point,
            eval.values[index],
            oods_point,
            oods_value,
        ));
    }
    CircleEvaluation::new(eval.domain, values)
}

pub fn quotient_log_bounds<B: Backend>(component: impl Component<B>) -> Vec<CirclePolyDegreeBound> {
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
