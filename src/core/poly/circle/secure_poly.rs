use std::ops::Deref;

use crate::core::backend::cpu::CPUCirclePoly;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;

pub struct SecureCirclePoly(pub [CPUCirclePoly; SECURE_EXTENSION_DEGREE]);

impl SecureCirclePoly {
    pub fn eval_at_point(&self, point: CirclePoint<SecureField>) -> SecureField {
        Self::eval_from_partial_evals(self.eval_columns_at_point(point))
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

    /// Evaluates the polynomial at a point, given evaluations of its composing base field
    /// polynomials at that point.
    pub fn eval_from_partial_evals(evals: [SecureField; SECURE_EXTENSION_DEGREE]) -> SecureField {
        let mut res = evals[0];
        res += evals[1] * SecureField::from_u32_unchecked(0, 1, 0, 0);
        res += evals[2] * SecureField::from_u32_unchecked(0, 0, 1, 0);
        res += evals[3] * SecureField::from_u32_unchecked(0, 0, 0, 1);
        res
    }
}

impl Deref for SecureCirclePoly {
    type Target = [CPUCirclePoly; SECURE_EXTENSION_DEGREE];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
