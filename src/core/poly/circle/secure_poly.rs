use std::ops::Deref;

use crate::core::air::evaluation::SECURE_EXTENSION_DEGREE;
use crate::core::backend::cpu::CPUCirclePoly;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;

pub struct SecureCirclePoly(pub [CPUCirclePoly<BaseField>; SECURE_EXTENSION_DEGREE]);

impl SecureCirclePoly {
    pub fn eval_at_point(&self, point: CirclePoint<SecureField>) -> SecureField {
        let mut res = self.0[0].eval_at_point(point);
        res += self.0[1].eval_at_point(point) * SecureField::from_u32_unchecked(0, 1, 0, 0);
        res += self.0[2].eval_at_point(point) * SecureField::from_u32_unchecked(0, 0, 1, 0);
        res += self.0[3].eval_at_point(point) * SecureField::from_u32_unchecked(0, 0, 0, 1);
        res
    }
}

impl Deref for SecureCirclePoly {
    type Target = [CPUCirclePoly<BaseField>; SECURE_EXTENSION_DEGREE];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
