use std::ops::{Deref, DerefMut};

use itertools::zip_eq;
use serde::{Deserialize, Serialize};

use crate::core::ColumnVec;

/// A container that holds an element for each commitment tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeVec<T>(pub Vec<T>);

impl<T> TreeVec<T> {
    pub fn new(vec: Vec<T>) -> TreeVec<T> {
        TreeVec(vec)
    }
    pub fn map<U, F: Fn(T) -> U>(self, f: F) -> TreeVec<U> {
        TreeVec(self.0.into_iter().map(f).collect())
    }
    pub fn zip<U>(self, other: impl Into<TreeVec<U>>) -> TreeVec<(T, U)> {
        let other = other.into();
        TreeVec(self.0.into_iter().zip(other.0).collect())
    }
    pub fn zip_eq<U>(self, other: impl Into<TreeVec<U>>) -> TreeVec<(T, U)> {
        let other = other.into();
        TreeVec(zip_eq(self.0, other.0).collect())
    }
    pub fn as_ref(&self) -> TreeVec<&T> {
        TreeVec(self.iter().collect())
    }
    pub fn as_mut(&mut self) -> TreeVec<&mut T> {
        TreeVec(self.iter_mut().collect())
    }
}

/// Converts `&TreeVec<T>` to `TreeVec<&T>`.
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

impl<T> TreeVec<ColumnVec<T>> {
    pub fn map_cols<U, F: FnMut(T) -> U>(self, mut f: F) -> TreeVec<ColumnVec<U>> {
        TreeVec(
            self.0
                .into_iter()
                .map(|column| column.into_iter().map(&mut f).collect())
                .collect(),
        )
    }
    /// Zips two [`TreeVec<ColumVec<T>>`] with the same structure (number of columns in each tree).
    /// The resulting [`TreeVec<ColumVec<T>>`] has the same structure, with each value being a tuple
    /// of the corresponding values from the input [`TreeVec<ColumVec<T>>`].
    pub fn zip_cols<U>(
        self,
        other: impl Into<TreeVec<ColumnVec<U>>>,
    ) -> TreeVec<ColumnVec<(T, U)>> {
        let other = other.into();
        TreeVec(
            zip_eq(self.0, other.0)
                .map(|(column1, column2)| zip_eq(column1, column2).collect())
                .collect(),
        )
    }
    pub fn as_cols_ref(&self) -> TreeVec<ColumnVec<&T>> {
        TreeVec(self.iter().map(|column| column.iter().collect()).collect())
    }
    /// Flattens the [`TreeVec<ColumVec<T>>`] into a single [`ColumnVec`] with all the columns
    /// combined.
    pub fn flatten(self) -> ColumnVec<T> {
        self.0.into_iter().flatten().collect()
    }
}

impl<'a, T> From<&'a TreeVec<ColumnVec<T>>> for TreeVec<ColumnVec<&'a T>> {
    fn from(val: &'a TreeVec<ColumnVec<T>>) -> Self {
        val.as_cols_ref()
    }
}

impl<T> TreeVec<ColumnVec<Vec<T>>> {
    /// Flattens a [`TreeVec<ColumVec<T>>`] of [Vec]s into a single [Vec] with all the elements
    /// combined.
    pub fn flatten_cols(self) -> Vec<T> {
        self.0.into_iter().flatten().flatten().collect()
    }
}
