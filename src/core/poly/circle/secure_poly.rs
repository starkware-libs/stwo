use std::ops::Deref;

use super::CirclePoly;
use crate::core::air::evaluation::SECURE_EXTENSION_DEGREE;
use crate::core::backend::cpu::CPUCirclePoly;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Column;

pub type SecureArray<T> = [T; SECURE_EXTENSION_DEGREE];

pub struct SecureCirclePoly(pub SecureArray<CPUCirclePoly<BaseField>>);

impl SecureCirclePoly {
    pub fn eval_at_point(&self, point: CirclePoint<SecureField>) -> SecureField {
        to_single_value(self.eval_columns_at_point(point))
    }

    pub fn eval_columns_at_point(
        &self,
        point: CirclePoint<SecureField>,
    ) -> SecureArray<SecureField> {
        [
            self[0].eval_at_point(point),
            self[1].eval_at_point(point),
            self[2].eval_at_point(point),
            self[3].eval_at_point(point),
        ]
    }

    // TODO(AlonH): Remove this temporary function.
    pub fn to_circle_poly(&self) -> CPUCirclePoly<SecureField> {
        let coeffs_len = self[0].coeffs.len();
        let mut coeffs = Vec::<SecureField>::zeros(coeffs_len);
        #[allow(clippy::needless_range_loop)]
        for index in 0..coeffs_len {
            coeffs[index] =
                SecureField::from_m31_array(std::array::from_fn(|i| self[i].coeffs[index]));
        }
        CirclePoly::new(coeffs)
    }
}

impl Deref for SecureCirclePoly {
    type Target = SecureArray<CPUCirclePoly<BaseField>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub fn to_single_value(values: SecureArray<SecureField>) -> SecureField {
    let mut res = values[0];
    res += values[1] * SecureField::from_u32_unchecked(0, 1, 0, 0);
    res += values[2] * SecureField::from_u32_unchecked(0, 0, 1, 0);
    res += values[3] * SecureField::from_u32_unchecked(0, 0, 0, 1);
    res
}
