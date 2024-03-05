use std::ops::Deref;

use super::CircleDomain;
use crate::core::air::evaluation::{SecureColumn, SECURE_EXTENSION_DEGREE};
use crate::core::backend::cpu::{CPUCircleEvaluation, CPUCirclePoly};
use crate::core::backend::CPUBackend;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::{BitReversedOrder, NaturalOrder};

pub struct SecureCirclePoly(pub [CPUCirclePoly<BaseField>; SECURE_EXTENSION_DEGREE]);

impl SecureCirclePoly {
    pub fn eval_at_point(&self, point: CirclePoint<SecureField>) -> SecureField {
        let mut res = self[0].eval_at_point(point);
        res += self[1].eval_at_point(point) * SecureField::from_u32_unchecked(0, 1, 0, 0);
        res += self[2].eval_at_point(point) * SecureField::from_u32_unchecked(0, 0, 1, 0);
        res += self[3].eval_at_point(point) * SecureField::from_u32_unchecked(0, 0, 0, 1);
        res
    }
}

impl Deref for SecureCirclePoly {
    type Target = [CPUCirclePoly<BaseField>; SECURE_EXTENSION_DEGREE];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct SecureCircleEvaluation<EvalOrder = NaturalOrder> {
    pub domain: CircleDomain,
    pub values: SecureColumn<CPUBackend>,
    _eval_order: std::marker::PhantomData<EvalOrder>,
}

impl<EvalOrder> SecureCircleEvaluation<EvalOrder> {
    pub fn new(domain: CircleDomain, values: SecureColumn<CPUBackend>) -> Self {
        Self {
            domain,
            values,
            _eval_order: std::marker::PhantomData,
        }
    }
}

impl SecureCircleEvaluation<BitReversedOrder> {
    pub fn from_evaluations(
        evaluations: &[CPUCircleEvaluation<BaseField, BitReversedOrder>],
    ) -> Self {
        assert_eq!(evaluations.len(), 4);
        let domain = evaluations[0].domain;
        let values = SecureColumn {
            cols: std::array::from_fn(|i| evaluations[i].values.clone()),
        };
        Self::new(domain, values)
    }
}
