use super::m31::BaseField;
use super::qm31::SecureField;
use super::{ExtensionOf, FieldOps};
use crate::core::backend::{Col, Column, CpuBackend};
use crate::core::utils::IteratorMutExt;

pub const SECURE_EXTENSION_DEGREE: usize =
    <SecureField as ExtensionOf<BaseField>>::EXTENSION_DEGREE;

/// An array of `SECURE_EXTENSION_DEGREE` base field columns, that represents a column of secure
/// field elements.
#[derive(Clone, Debug)]
pub struct SecureColumn<B: FieldOps<BaseField>> {
    pub columns: [Col<B, BaseField>; SECURE_EXTENSION_DEGREE],
}
impl SecureColumn<CpuBackend> {
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

    pub fn to_cpu(&self) -> SecureColumn<CpuBackend> {
        SecureColumn {
            columns: self.columns.clone().map(|c| c.to_cpu()),
        }
    }
}

pub struct SecureColumnIter<'a> {
    column: &'a SecureColumn<CpuBackend>,
    index: usize,
}
impl Iterator for SecureColumnIter<'_> {
    type Item = SecureField;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.column.len() {
            let value = self.column.at(self.index);
            self.index += 1;
            Some(value)
        } else {
            None
        }
    }
}
impl<'a> IntoIterator for &'a SecureColumn<CpuBackend> {
    type Item = SecureField;
    type IntoIter = SecureColumnIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        SecureColumnIter {
            column: self,
            index: 0,
        }
    }
}
impl FromIterator<SecureField> for SecureColumn<CpuBackend> {
    fn from_iter<I: IntoIterator<Item = SecureField>>(iter: I) -> Self {
        let mut columns = std::array::from_fn(|_| vec![]);
        for value in iter.into_iter() {
            let vals = value.to_m31_array();
            for j in 0..SECURE_EXTENSION_DEGREE {
                columns[j].push(vals[j]);
            }
        }
        SecureColumn { columns }
    }
}
impl From<SecureColumn<CpuBackend>> for Vec<SecureField> {
    fn from(column: SecureColumn<CpuBackend>) -> Self {
        column.into_iter().collect()
    }
}
