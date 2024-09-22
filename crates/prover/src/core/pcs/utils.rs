use std::collections::BTreeSet;
use std::ops::{Deref, DerefMut};

use itertools::zip_eq;
use serde::{Deserialize, Serialize};

use super::TreeSubspan;
use crate::core::ColumnVec;

/// A container that holds an element for each commitment tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeVec<T>(pub Vec<T>);

impl<T> TreeVec<T> {
    pub const fn new(vec: Vec<T>) -> TreeVec<T> {
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

/// Converts `&TreeVec<&Vec<T>>` to `TreeVec<Vec<&T>>`.
impl<'a, T> From<&'a TreeVec<&'a Vec<T>>> for TreeVec<Vec<&'a T>> {
    fn from(val: &'a TreeVec<&'a Vec<T>>) -> Self {
        TreeVec(val.iter().map(|vec| vec.iter().collect()).collect())
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

    /// Appends the columns of another [`TreeVec<ColumVec<T>>`] to this one.
    pub fn append_cols(&mut self, mut other: TreeVec<ColumnVec<T>>) {
        let n_trees = self.0.len().max(other.0.len());
        self.0.resize_with(n_trees, Default::default);
        for (self_col, other_col) in self.0.iter_mut().zip(other.0.iter_mut()) {
            self_col.append(other_col);
        }
    }

    /// Concatenates the columns of multiple [`TreeVec<ColumVec<T>>`] into a single
    /// [`TreeVec<ColumVec<T>>`].
    pub fn concat_cols(
        trees: impl Iterator<Item = TreeVec<ColumnVec<T>>>,
    ) -> TreeVec<ColumnVec<T>> {
        let mut result = TreeVec::default();
        for tree in trees {
            result.append_cols(tree);
        }
        result
    }

    /// Extracts a sub-tree based on the specified locations.
    ///
    /// # Panics
    ///
    /// If two or more locations have the same tree index.
    pub fn sub_tree(&self, locations: &[TreeSubspan]) -> TreeVec<ColumnVec<&T>> {
        let tree_indicies: BTreeSet<usize> = locations.iter().map(|l| l.tree_index).collect();
        assert_eq!(tree_indicies.len(), locations.len());
        let max_tree_index = tree_indicies.iter().max().unwrap_or(&0);
        let mut res = TreeVec(vec![Vec::new(); max_tree_index + 1]);

        for &location in locations {
            // TODO(andrew): Throwing error here might be better instead.
            let chunk = self.get_chunk(location).unwrap();
            res[location.tree_index] = chunk;
        }

        res
    }

    fn get_chunk(&self, location: TreeSubspan) -> Option<ColumnVec<&T>> {
        let tree = self.0.get(location.tree_index)?;
        let chunk = tree.get(location.col_start..location.col_end)?;
        Some(chunk.iter().collect())
    }
}

impl<T> TreeVec<&ColumnVec<T>> {
    pub fn map_cols<U, F: FnMut(&T) -> U>(self, mut f: F) -> TreeVec<ColumnVec<U>> {
        TreeVec(
            self.0
                .into_iter()
                .map(|column| column.iter().map(&mut f).collect())
                .collect(),
        )
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
