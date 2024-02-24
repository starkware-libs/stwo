use std::ops::Deref;

use crate::core::air::evaluation::SECURE_EXTENSION_DEGREE;
use crate::core::backend::cpu::CPUCirclePoly;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;

pub struct SecureCirclePoly(pub [CPUCirclePoly; SECURE_EXTENSION_DEGREE]);

impl SecureCirclePoly {
    pub fn eval_at_point(&self, point: CirclePoint<SecureField>) -> SecureField {
        combine_secure_value(self.eval_columns_at_point(point))
    }

    pub fn eval_columns_at_point(
        &self,
        point: CirclePoint<SecureField>,
    ) -> [SecureField; SECURE_EXTENSION_DEGREE] {
        [
            self[0].eval_at_point(point),
            self[1].eval_at_point(point),
            self[2].eval_at_point(point),
            self[3].eval_at_point(point),
        ]
    }
}

impl Deref for SecureCirclePoly {
    type Target = [CPUCirclePoly; SECURE_EXTENSION_DEGREE];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub fn combine_secure_value(value: [SecureField; SECURE_EXTENSION_DEGREE]) -> SecureField {
    let mut res = value[0];
    res += value[1] * SecureField::from_u32_unchecked(0, 1, 0, 0);
    res += value[2] * SecureField::from_u32_unchecked(0, 0, 1, 0);
    res += value[3] * SecureField::from_u32_unchecked(0, 0, 0, 1);
    res
}
