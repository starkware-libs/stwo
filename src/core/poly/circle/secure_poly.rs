use std::ops::Deref;

use super::CircleDomain;
use crate::core::backend::cpu::{CPUCircleEvaluation, CPUCirclePoly};
use crate::core::backend::CPUBackend;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::{SecureColumn, SECURE_EXTENSION_DEGREE};
use crate::core::fields::FieldOps;
use crate::core::poly::BitReversedOrder;

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

    pub fn log_size(&self) -> u32 {
        self[0].log_size()
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

pub struct SecureEvaluation<B: FieldOps<BaseField>> {
    pub domain: CircleDomain,
    pub values: SecureColumn<B>,
}
impl<B: FieldOps<BaseField>> Deref for SecureEvaluation<B> {
    type Target = SecureColumn<B>;

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl SecureEvaluation<CPUBackend> {
    // TODO(spapini): Remove when we no longer use CircleEvaluation<SecureField>.
    pub fn to_cpu(self) -> CPUCircleEvaluation<SecureField, BitReversedOrder> {
        CPUCircleEvaluation::new(self.domain, self.values.to_vec())
    }
}
