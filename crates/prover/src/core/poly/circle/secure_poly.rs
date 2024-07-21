use std::ops::{Deref, DerefMut};

use super::{CircleDomain, CircleEvaluation, CirclePoly, PolyOps};
use crate::core::backend::cpu::CpuCircleEvaluation;
use crate::core::backend::CpuBackend;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::{SecureColumnByCoords, SECURE_EXTENSION_DEGREE};
use crate::core::fields::FieldOps;
use crate::core::poly::BitReversedOrder;

pub struct SecureCirclePoly<B: FieldOps<BaseField>>(pub [CirclePoly<B>; SECURE_EXTENSION_DEGREE]);

impl<B: PolyOps> SecureCirclePoly<B> {
    pub fn eval_at_point(&self, point: CirclePoint<SecureField>) -> SecureField {
        SecureField::from_partial_evals(self.eval_columns_at_point(point))
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
}

impl<B: FieldOps<BaseField>> Deref for SecureCirclePoly<B> {
    type Target = [CirclePoly<B>; SECURE_EXTENSION_DEGREE];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone)]
pub struct SecureEvaluation<B: FieldOps<BaseField>> {
    pub domain: CircleDomain,
    pub values: SecureColumnByCoords<B>,
}
impl<B: FieldOps<BaseField>> Deref for SecureEvaluation<B> {
    type Target = SecureColumnByCoords<B>;

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<B: FieldOps<BaseField>> DerefMut for SecureEvaluation<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

impl SecureEvaluation<CpuBackend> {
    // TODO(spapini): Remove when we no longer use CircleEvaluation<SecureField>.
    pub fn to_cpu(self) -> CpuCircleEvaluation<SecureField, BitReversedOrder> {
        CpuCircleEvaluation::new(self.domain, self.values.to_vec())
    }
}

impl From<CircleEvaluation<CpuBackend, SecureField, BitReversedOrder>>
    for SecureEvaluation<CpuBackend>
{
    fn from(evaluation: CircleEvaluation<CpuBackend, SecureField, BitReversedOrder>) -> Self {
        Self {
            domain: evaluation.domain,
            values: evaluation.values.into_iter().collect(),
        }
    }
}
