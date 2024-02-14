use super::air::Mask;
use super::backend::{Backend, CPUBackend};
use super::circle::{CirclePoint, CirclePointIndex};
use super::constraints::{
    complex_conjugate_line, pair_vanishing, point_vanishing, EvalByEvaluation, EvalByPoly,
    PolyOracle,
};
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::fields::ComplexConjugate;
use super::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly, PointMapping};
use super::poly::{BitReversedOrder, NaturalOrder};

type B = CPUBackend;

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
    eval: &CircleEvaluation<B, SecureField, BitReversedOrder>,
) -> CircleEvaluation<B, SecureField, NaturalOrder> {
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
    eval: &CircleEvaluation<B, BaseField, BitReversedOrder>,
) -> CircleEvaluation<B, SecureField, NaturalOrder> {
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

/// Returns the evaluations of the trace mask at the OODS point.
pub fn get_oods_values<B: Backend>(
    mask: &Mask,
    oods_point: CirclePoint<SecureField>,
    trace_polys: &[CirclePoly<B, BaseField>],
) -> PointMapping<SecureField> {
    let mut oods_evals = Vec::with_capacity(trace_polys.len());
    for poly in trace_polys {
        oods_evals.push(EvalByPoly {
            point: oods_point,
            poly,
        });
    }
    let trace_domains = trace_polys
        .iter()
        .map(|poly| CanonicCoset::new(poly.log_size()))
        .collect::<Vec<_>>();
    mask.get_evaluation(&trace_domains, &oods_evals[..])
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
