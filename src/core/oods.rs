use itertools::enumerate;
use num_traits::Zero;

use super::backend::cpu::{
    CPUCircleEvaluation, LOG_N_SECUREFIELD_IN_CACHE, N_SECUREFIELD_IN_CACHE,
};
use super::circle::CirclePoint;
use super::constraints::{complex_conjugate_line, pair_vanishing, point_vanishing};
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::fields::{ComplexConjugate, FieldExpOps};
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

    // Fallback to the non-chunked version if the domain is not big enough.
    if eval.domain.log_size() < LOG_N_SECUREFIELD_IN_CACHE as u32 {
        for (i, point) in enumerate(eval.domain.iter()) {
            let index = bit_reverse_index(i, eval.domain.log_size());
            values.push(eval_pair_oods_quotient_at_point(
                point,
                eval.values[index],
                oods_point,
                oods_value,
            ));
        }
        return CircleEvaluation::new(eval.domain, values);
    }
    let denom = oods_point.complex_conjugate().y - oods_point.y;
    let denom_inv = denom.inverse();
    for chunk in eval
        .domain
        .iter()
        .enumerate()
        .array_chunks::<N_SECUREFIELD_IN_CACHE>()
    {
        // Cached values for complex conjugate lines calculation.
        let mul = (oods_value.complex_conjugate() - oods_value) * denom_inv;
        let add = oods_value + mul * (-oods_point.y);

        let numerators: [SecureField; N_SECUREFIELD_IN_CACHE] = std::array::from_fn(|i| {
            let (i,point) = chunk[i];
            let idx = bit_reverse_index(i, eval.domain.log_size());
            let value = eval.values[idx];
            let complex_conjugate_line = add + (mul * (point.y));
            value - complex_conjugate_line
        });
        let denominators: [SecureField; N_SECUREFIELD_IN_CACHE] = std::array::from_fn(|i| {
            let point = chunk[i].1;
            pair_vanishing(oods_point, oods_point.complex_conjugate(), point.into_ef())
        });
        let mut inversed_denominators = [SecureField::zero(); N_SECUREFIELD_IN_CACHE];
        SecureField::batch_inverse(&denominators, &mut inversed_denominators);

        for i in 0..N_SECUREFIELD_IN_CACHE {
            values.push(numerators[i] * inversed_denominators[i]);
        }
    }
    CircleEvaluation::new(eval.domain, values)
}
