use super::m31::BaseField;
use super::qm31::SecureField;
use super::{ExtensionOf, FieldOps};
use crate::core::backend::{CPUBackend, Col, Column};
use crate::core::utils::IteratorMutExt;

pub const SECURE_EXTENSION_DEGREE: usize =
    <SecureField as ExtensionOf<BaseField>>::EXTENSION_DEGREE;

/// An array of `SECURE_EXTENSION_DEGREE` base field columns, that represents a column of secure
/// field elements.
/// `SecureColumn` is a list of SecureField elements.
#[derive(Clone, Debug)]
pub struct SecureColumn<B: FieldOps<BaseField>> {
    pub columns: [Col<B, BaseField>; SECURE_EXTENSION_DEGREE],
}
impl SecureColumn<CPUBackend> {
    /// Sets the value of the SecureField at the specified index in each column of the SecureColumn.
    ///
    /// # Arguments
    ///
    /// * `self` - The `SecureColumn` instance.
    /// * `index` - The index at which to set the value.
    /// * `value` - The `SecureField` value to set at the specified index.
    ///
    /// # Returns
    ///
    /// A new `SecureColumn` instance.
    ///
    /// # Example
    ///
    /// ```
    ///     use stwo_prover::core::fields::secure_column::SecureColumn;
    ///     use stwo_prover::core::backend::{CPUBackend};
    ///     use stwo_prover::core::fields::cm31::CM31;
    ///     use stwo_prover::core::fields::qm31::QM31;
    ///     use stwo_prover::core::fields::m31::{BaseField, M31};
    /// 
    ///     let a = M31(5);
    ///     let b = M31(6);
    ///     let c = M31(7);
    ///     let d = M31(8);
    ///     let array: [M31; 4] = [a, b, c, d];
    /// 
    ///     let qm = QM31::from_m31_array(array);
    ///
    ///     let mut secure_col = SecureColumn::<CPUBackend> {
    ///            columns: std::array::from_fn(|i| {
    ///                 vec![BaseField::from_u32_unchecked(i as u32); 4]
    ///         }),
    ///     };
    ///     println!("secure_column value: {:?}", secure_col);  
    /// 
    ///     secure_col.set(2, qm);
    ///     println!("secure_column value: {:?}", secure_col); 
    /// ```
    pub fn set(&mut self, index: usize, value: SecureField) {
        self.columns
            .iter_mut()
            .map(|c| &mut c[index])
            .assign(value.to_m31_array());
    }

    // TODO(spapini): Remove when we no longer use CircleEvaluation<SecureField>.
    /// Converts a `SecureColumn` with a CPU backend into a vector of `SecureField` elements.
    ///
    /// # Arguments
    ///
    /// * `self` - The `SecureColumn` instance.
    ///
    /// # Returns
    ///
    /// A Vector of `SecureField` elements.
    ///
    /// # Example
    ///
    /// ```
    ///     use stwo_prover::core::fields::secure_column::SecureColumn;
    ///     use stwo_prover::core::backend::{CPUBackend};
    ///     use stwo_prover::core::fields::m31::{BaseField};
    /// 
    ///     let secure_column = SecureColumn::<CPUBackend> {
    ///            columns: std::array::from_fn(|i| {
    ///                 vec![BaseField::from_u32_unchecked(i as u32); 2]
    ///         }),
    ///     };
    /// 
    ///     println!("vector of secure_column: {:?}", secure_column.to_vec());
    /// ```
    pub fn to_vec(&self) -> Vec<SecureField> {
        (0..self.len()).map(|i| self.at(i)).collect()
    }
}
impl<B: FieldOps<BaseField>> SecureColumn<B> {
    /// Retrieves the SecureField value at the specified index from the SecureColumn.
    ///
    /// # Arguments
    ///
    /// * `self` - The `SecureColumn` instance.
    /// * `index` - The index at which to set the value.
    ///
    /// # Returns
    ///
    /// A new `QM31` instance.
    ///
    /// # Example
    ///
    /// ```
    ///     use stwo_prover::core::fields::secure_column::SecureColumn;
    ///     use stwo_prover::core::backend::{CPUBackend};
    ///     use stwo_prover::core::fields::m31::{BaseField};
    /// 
    ///     let secure_col = SecureColumn::<CPUBackend> {
    ///            columns: std::array::from_fn(|i| {
    ///                 vec![BaseField::from_u32_unchecked(i as u32); 4]
    ///         }),
    ///     };
    ///     println!("secure_column value: {:?}", secure_col);  
    ///     println!("secure_column at index: {:?}", secure_col.at(2));
    /// ```
    pub fn at(&self, index: usize) -> SecureField {
        SecureField::from_m31_array(std::array::from_fn(|i| self.columns[i].at(index)))
    }

    /// Creates a new SecureColumn with all elements as zero.
    ///
    /// # Arguments
    ///
    /// * `len` - The len of each column in the`SecureColumn`.
    ///
    /// # Returns
    ///
    /// A new `SecureColumn` instance.
    ///
    /// # Example
    ///
    /// ```
    ///     use stwo_prover::core::fields::secure_column::SecureColumn;
    ///     use stwo_prover::core::backend::{CPUBackend};
    /// 
    ///     let secure_col = SecureColumn::<CPUBackend>::zeros(3);
    ///     println!("secure_column value: {:?}", secure_col);  
    /// ```
    pub fn zeros(len: usize) -> Self {
        Self {
            columns: std::array::from_fn(|_| Col::<B, BaseField>::zeros(len)),
        }
    }

    /// Returns the length of the columns in the `SecureColumn`.
    ///
    /// # Arguments
    ///
    /// * `self` - The `SecureColumn` instance.
    ///
    /// # Returns
    ///
    /// The length in `usize` of a column. All columns should have the same length.
    ///
    /// # Example
    ///
    /// ```
    ///     use stwo_prover::core::fields::secure_column::SecureColumn;
    ///     use stwo_prover::core::backend::{CPUBackend};
    /// 
    ///     let secure_col = SecureColumn::<CPUBackend>::zeros(3);
    ///     println!("length of secure_column: {:?}", secure_col.len()); 
    /// ```
    pub fn len(&self) -> usize {
        self.columns[0].len()
    }

    /// Checks if the `SecureColumn` is empty.
    ///
    /// # Arguments
    ///
    /// * `self` - The `SecureColumn` instance.
    ///
    /// # Returns
    ///
    /// Returns `true` if the SecureColumn is empty, otherwise `false`.
    ///
    /// # Example
    ///
    /// ```
    ///     use stwo_prover::core::fields::secure_column::SecureColumn;
    ///     use stwo_prover::core::backend::{CPUBackend};
    /// 
    ///     let secure_col = SecureColumn::<CPUBackend>::zeros(3);
    ///     println!("Is secure_column empty: {:?}", secure_col.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.columns[0].is_empty()
    }
    
    /// Converts the SecureColumn to use the CPU backend.
    ///
    /// Returns the length of the columns in the `SecureColumn`.
    ///
    /// # Arguments
    ///
    /// * `self` - The `SecureColumn` instance.
    ///
    /// # Returns
    ///
    /// Returns a new `SecureColumn` using the CPU backend containing the same data.
    ///
    /// # Example
    ///
    /// ```
    ///     use stwo_prover::core::fields::secure_column::SecureColumn;
    ///     use stwo_prover::core::backend::{CPUBackend};
    ///     use stwo_prover::core::backend::simd::{SimdBackend};
    /// 
    ///     let secure_col = SecureColumn::<CPUBackend>::zeros(1);
    ///     println!("secure_column SimdBackend: {:?}", secure_col);
    /// 
    ///     println!("secure_column to CPU: {:?}", secure_col.to_cpu());
    /// ```
    pub fn to_cpu(&self) -> SecureColumn<CPUBackend> {
        SecureColumn {
            columns: self.columns.clone().map(|c| c.to_cpu()),
        }
    }
}

/// Iterator over the columns of a SecureColumn with a CPU backend.
///
/// # Fields
///
/// * `column` - A reference to the SecureColumn being iterated over.
/// * `index` - The current index of the iteration.
pub struct SecureColumnIter<'a> {
    column: &'a SecureColumn<CPUBackend>,
    index: usize,
}

/// Implementation of the `Iterator` trait for `SecureColumnIter`.
impl Iterator for SecureColumnIter<'_> {
    type Item = SecureField;

    /// Converts the SecureColumn to use the CPU backend.
    ///
    /// # Arguments
    ///
    /// * `self` - The `SecureColumn` instance.
    ///
    /// # Returns
    ///
    /// Returns a new `SecureColumn` using the CPU backend containing the same data.
    ///
    /// # Example
    ///
    /// ```text
    ///     use stwo_prover::core::fields::secure_column::{SecureColumn, SecureColumnIter};
    ///     use stwo_prover::core::backend::{CPUBackend};
    /// 
    ///     let secure_column = SecureColumn::<CPUBackend> {
    ///         columns: std::array::from_fn(|i| {
    ///         vec![BaseField::from_u32_unchecked(i as u32); 4]
    ///        }),
    /// 
    ///     let mut sec_iter = SecureColumnIter {
    ///         column: &secure_column,
    ///         index: 0,
    ///     };
    /// 
    ///     println!("next secure field: {:?}", sec_iter.next());
    /// ```    
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
/// Implementation of the `IntoIterator` trait for `SecureColumn<CPUBackend>`.
impl<'a> IntoIterator for &'a SecureColumn<CPUBackend> {
    type Item = SecureField;
    type IntoIter = SecureColumnIter<'a>;

    /// Converts the `SecureColumn` into an iterator.
    ///
    /// # Returns
    ///
    /// Returns an iterator over the `SecureColumn`.
    ///
    /// # Example
    ///
    /// ```text
    ///     use stwo_prover::core::fields::secure_column::{SecureColumn, SecureColumnIter};
    ///     use stwo_prover::core::backend::{CPUBackend};
    /// 
    ///     let secure_column = SecureColumn::<CPUBackend> {
    ///         columns: std::array::from_fn(|i| {
    ///         vec![BaseField::from_u32_unchecked(i as u32); 4]
    ///        }),
    ///     
    ///     let mut sec_iter = secure_column.into_iter();
    /// 
    ///     println!("next secure field: {:?}", sec_iter.next());
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        SecureColumnIter {
            column: self,
            index: 0,
        }
    }
}
/// Implementation of the `FromIterator` trait for `SecureColumn<CPUBackend>`.
impl FromIterator<SecureField> for SecureColumn<CPUBackend> {
    /// Creates a new SecureColumn with all elements as zero.
    ///
    /// # Arguments
    ///
    /// * `iter` - An iterator yielding `SecureField` elements.
    ///
    /// # Returns
    ///
    /// Returns a new `SecureColumn` constructed from the elements of the iterator.
    ///
    /// # Example
    ///
    /// ```
    ///     use stwo_prover::core::fields::secure_column::SecureColumn;
    ///     use stwo_prover::core::backend::{CPUBackend};
    ///     use stwo_prover::core::fields::m31::{BaseField};
    /// 
    ///     let secure_column = SecureColumn::<CPUBackend> {
    ///         columns: std::array::from_fn(|i| {
    ///             vec![BaseField::from_u32_unchecked(i as u32); 2]
    ///         }),
    ///     };
    ///
    ///     let sec_iter = secure_column.into_iter();
    ///     let secure_col = SecureColumn::<CPUBackend>::from_iter(sec_iter);
    ///     println!("secure_column: {:?}", secure_col);
    /// ```
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
/// Implements conversion from a `SecureColumn<CPUBackend>` into a vector of `SecureField` elements.
impl From<SecureColumn<CPUBackend>> for Vec<SecureField> {
    fn from(column: SecureColumn<CPUBackend>) -> Self {
        column.into_iter().collect()
    }
}
