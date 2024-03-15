use std::ops::Deref;

use super::m31::BaseField;
use super::qm31::SecureField;
use super::{ExtensionOf, FieldOps};
use crate::core::backend::cpu::{CPUCircleEvaluation, CPUCirclePoly};
use crate::core::backend::{CPUBackend, Col, Column};
use crate::core::circle::CirclePoint;
use crate::core::poly::circle::CircleDomain;
use crate::core::poly::BitReversedOrder;
use crate::core::utils::IteratorMutExt;

pub const SECURE_EXTENSION_DEGREE: usize =
    <SecureField as ExtensionOf<BaseField>>::EXTENSION_DEGREE;

pub struct SecureColumn<B: FieldOps<BaseField>> {
    pub cols: [Col<B, BaseField>; SECURE_EXTENSION_DEGREE],
}
impl SecureColumn<CPUBackend> {
    pub fn set(&mut self, index: usize, value: SecureField) {
        self.cols
            .iter_mut()
            .map(|c| &mut c[index])
            .assign(value.to_m31_array());
    }
}
impl<B: FieldOps<BaseField>> SecureColumn<B> {
    pub fn zeros(len: usize) -> Self {
        Self {
            cols: std::array::from_fn(|_| Col::<B, BaseField>::zeros(len)),
        }
    }

    pub fn len(&self) -> usize {
        self.cols[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.cols[0].is_empty()
    }

    pub fn at(&self, index: usize) -> SecureField {
        SecureField::from_m31_array(std::array::from_fn(|i| self.cols[i].at(index)))
    }

    // TODO(spapini): Remove when we no longer use CircleEvaluation<SecureField>.
    pub fn to_cpu(&self) -> Vec<SecureField> {
        (0..self.len()).map(|i| self.at(i)).collect()
    }
}

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

    pub fn log_size(&self) -> u32 {
        self[0].log_size()
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
impl<B: FieldOps<BaseField>> SecureEvaluation<B> {
    // TODO(spapini): Remove when we no longer use CircleEvaluation<SecureField>.
    pub fn to_cpu_circle_eval(self) -> CPUCircleEvaluation<SecureField, BitReversedOrder> {
        CPUCircleEvaluation::new(self.domain, self.values.to_cpu())
    }
}

pub fn combine_secure_value(value: [SecureField; SECURE_EXTENSION_DEGREE]) -> SecureField {
    let mut res = value[0];
    res += value[1] * SecureField::from_u32_unchecked(0, 1, 0, 0);
    res += value[2] * SecureField::from_u32_unchecked(0, 0, 1, 0);
    res += value[3] * SecureField::from_u32_unchecked(0, 0, 0, 1);
    res
}
