use super::m31::BaseField;
use super::qm31::SecureField;
use super::{ExtensionOf, FieldOps};
use crate::core::backend::{CPUBackend, Col, Column};
use crate::core::utils::IteratorMutExt;

pub const SECURE_EXTENSION_DEGREE: usize =
    <SecureField as ExtensionOf<BaseField>>::EXTENSION_DEGREE;

/// An array of `SECURE_EXTENSION_DEGREE` base field columns, that represents a column of secure
/// field elements.
#[derive(Clone, Debug)]
pub struct SecureColumn<B: FieldOps<BaseField>> {
    pub columns: [Col<B, BaseField>; SECURE_EXTENSION_DEGREE],
}
impl SecureColumn<CPUBackend> {
    pub fn set(&mut self, index: usize, value: SecureField) {
        self.columns
            .iter_mut()
            .map(|c| &mut c[index])
            .assign(value.to_m31_array());
    }

    // TODO(spapini): Remove when we no longer use CircleEvaluation<SecureField>.
    pub fn to_vec(&self) -> Vec<SecureField> {
        (0..self.len()).map(|i| self.at(i)).collect()
    }
}
impl<B: FieldOps<BaseField>> SecureColumn<B> {
    pub fn at(&self, index: usize) -> SecureField {
        SecureField::from_m31_array(std::array::from_fn(|i| self.columns[i].at(index)))
    }

    pub fn zeros(len: usize) -> Self {
        Self {
            columns: std::array::from_fn(|_| Col::<B, BaseField>::zeros(len)),
        }
    }

    pub fn len(&self) -> usize {
        self.columns[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.columns[0].is_empty()
    }
}
