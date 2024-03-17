use std::iter::zip;
use std::ops::{Deref, DerefMut};

use itertools::zip_eq;

use crate::core::ColumnVec;

/// A container that holds an element for each commitment tree.
#[derive(Debug, Clone)]
pub struct TreeVec<T>(pub Vec<T>);
impl<T> TreeVec<T> {
    pub fn new() -> TreeVec<T> {
        TreeVec(Vec::new())
    }
    /// Creates a [TreeColumns] instance by mapping each element of the [TreeVec] to a [ColumnVec].
    pub fn to_cols<'a, U: 'a, F: Fn(&'a T) -> ColumnVec<U>>(&'a self, f: F) -> TreeColumns<U> {
        TreeColumns(TreeVec(self.0.iter().map(f).collect()))
    }
    pub fn map<U, F: Fn(T) -> U>(self, f: F) -> TreeVec<U> {
        TreeVec(self.0.into_iter().map(f).collect())
    }
    pub fn zip<U>(self, other: impl Into<TreeVec<U>>) -> TreeVec<(T, U)> {
        let other = other.into();
        assert_eq!(self.0.len(), other.len());
        TreeVec(zip(self.0, other.0).collect())
    }
    pub fn as_ref(&self) -> TreeVec<&T> {
        TreeVec(self.0.iter().collect())
    }
}
// Converts &TreeVec<T> to TreeVec<&T>.
impl<'a, T> From<&'a TreeVec<T>> for TreeVec<&'a T> {
    fn from(val: &'a TreeVec<T>) -> Self {
        val.as_ref()
    }
}
impl<T> Deref for TreeVec<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> DerefMut for TreeVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl<T> Default for TreeVec<T> {
    fn default() -> Self {
        TreeVec(Vec::new())
    }
}

/// A container that holds an element for each column of each commitment tree.
#[derive(Debug, Clone)]
pub struct TreeColumns<T>(pub TreeVec<ColumnVec<T>>);
impl<T> TreeColumns<T> {
    pub fn new(values: Vec<ColumnVec<T>>) -> Self {
        TreeColumns(TreeVec(values))
    }
    pub fn map<U, F: FnMut(T) -> U>(self, mut f: F) -> TreeColumns<U> {
        TreeColumns(TreeVec(
            self.0
                 .0
                .into_iter()
                .map(|column| column.into_iter().map(&mut f).collect())
                .collect(),
        ))
    }
    /// Zips two [TreeColumns] with the same structure (number of columns in each tree).
    /// The resulting [TreeColumns] has the same structure, with each value being a tuple of the
    /// corresponding values from the input [TreeColumns].
    pub fn zip_cols<U>(self, other: impl Into<TreeColumns<U>>) -> TreeColumns<(T, U)> {
        let other = other.into();
        assert_eq!(self.0.len(), other.0.len());
        TreeColumns(TreeVec(
            self.0
                 .0
                .into_iter()
                .zip(other.0 .0)
                .map(|(column1, column2)| zip_eq(column1, column2).collect())
                .collect(),
        ))
    }
    pub fn as_ref(&self) -> TreeColumns<&T> {
        TreeColumns(TreeVec(
            self.0
                 .0
                .iter()
                .map(|column| column.iter().collect())
                .collect(),
        ))
    }
    /// Flattens the [TreeColumns] into a single [ColumnVec] with all the columns combined.
    pub fn flatten(self) -> ColumnVec<T> {
        self.0 .0.into_iter().flatten().collect()
    }
}
impl<'a, T> From<&'a TreeColumns<T>> for TreeColumns<&'a T> {
    fn from(val: &'a TreeColumns<T>) -> Self {
        val.as_ref()
    }
}
impl<T> TreeColumns<Vec<T>> {
    /// Flattens a [TreeColumns] of [Vec]s into a single [Vec] with all the elements combined.
    pub fn flatten_all(self) -> Vec<T> {
        self.flatten().into_iter().flatten().collect()
    }

    // TODO(spapini): Remove after accumulating oods quotients by size.
    /// Flattens a [TreeColumns] of [Vec]s into a single [Vec] with all the elements combined, in
    /// reverse order.
    pub fn flatten_all_rev(self) -> Vec<T> {
        self.flatten().into_iter().flatten().rev().collect()
    }
}
