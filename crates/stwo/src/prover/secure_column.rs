use std::array;
use std::iter::zip;

use itertools::zip_eq;

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use crate::prover::backend::{Col, Column, ColumnOps, CpuBackend};

/// A column major array of `SECURE_EXTENSION_DEGREE` base field columns, that represents a column
/// of secure field element coordinates.
#[derive(Clone, Debug)]
pub struct SecureColumnByCoords<B: ColumnOps<BaseField>> {
    pub columns: [Col<B, BaseField>; SECURE_EXTENSION_DEGREE],
}
impl SecureColumnByCoords<CpuBackend> {
    // TODO(first): Remove.
    pub fn to_vec(&self) -> Vec<SecureField> {
        (0..self.len()).map(|i| self.at(i)).collect()
    }
}
impl<B: ColumnOps<BaseField>> SecureColumnByCoords<B> {
    pub fn at(&self, index: usize) -> SecureField {
        SecureField::from_m31_array(std::array::from_fn(|i| self.columns[i].at(index)))
    }

    pub fn zeros(len: usize) -> Self {
        Self {
            columns: std::array::from_fn(|_| Col::<B, BaseField>::zeros(len)),
        }
    }

    /// # Safety
    pub unsafe fn uninitialized(len: usize) -> Self {
        Self {
            columns: std::array::from_fn(|_| Col::<B, BaseField>::uninitialized(len)),
        }
    }

    /// Creates a secure column from a base field column. Each base field element is embedded as
    /// a secure field element with its value in the first coordinate and zeros in the remaining
    /// coordinates.
    pub fn from_base_field_col(column: &Col<B, BaseField>) -> Self {
        let columns = array::from_fn(|i| {
            if i == 0 {
                column.clone()
            } else {
                Col::<B, BaseField>::zeros(column.len())
            }
        });
        Self { columns }
    }

    pub fn len(&self) -> usize {
        self.columns[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.columns[0].is_empty()
    }

    pub fn to_cpu(&self) -> SecureColumnByCoords<CpuBackend> {
        SecureColumnByCoords {
            columns: self.columns.clone().map(|c| c.to_cpu()),
        }
    }

    pub fn set(&mut self, index: usize, value: SecureField) {
        for (column, value) in zip_eq(&mut self.columns, value.to_m31_array()) {
            column.set(index, value);
        }
    }
}

pub struct SecureColumnByCoordsIter<'a> {
    column: &'a SecureColumnByCoords<CpuBackend>,
    index: usize,
}
impl Iterator for SecureColumnByCoordsIter<'_> {
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
impl<'a> IntoIterator for &'a SecureColumnByCoords<CpuBackend> {
    type Item = SecureField;
    type IntoIter = SecureColumnByCoordsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        SecureColumnByCoordsIter {
            column: self,
            index: 0,
        }
    }
}
impl FromIterator<SecureField> for SecureColumnByCoords<CpuBackend> {
    fn from_iter<I: IntoIterator<Item = SecureField>>(iter: I) -> Self {
        let values = iter.into_iter();
        let (lower_bound, _) = values.size_hint();
        let mut columns = array::from_fn(|_| Vec::with_capacity(lower_bound));

        for value in values {
            let coords = value.to_m31_array();
            zip(&mut columns, coords).for_each(|(col, coord)| col.push(coord));
        }

        SecureColumnByCoords { columns }
    }
}
impl From<SecureColumnByCoords<CpuBackend>> for Vec<SecureField> {
    fn from(column: SecureColumnByCoords<CpuBackend>) -> Self {
        column.into_iter().collect()
    }
}
