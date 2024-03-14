use std::iter::zip;
use std::ops::{Deref, DerefMut};

use crate::core::ColumnVec;

/// A container that holds an element for each commitment tree.
pub struct TreeVec<T>(pub Vec<T>);
impl<T> TreeVec<T> {
    pub fn new() -> TreeVec<T> {
        TreeVec(Vec::new())
    }
    pub fn to_cols<'a, U: 'a, F: Fn(&'a T) -> Vec<U>>(&'a self, f: F) -> TreeColumns<U> {
        TreeColumns(TreeVec(self.0.iter().map(f).collect()))
    }
    pub fn map<U, F: Fn(T) -> U>(self, f: F) -> TreeVec<U> {
        TreeVec(self.0.into_iter().map(f).collect())
    }
    pub fn zip<U>(self, other: impl Into<TreeVec<U>>) -> TreeVec<(T, U)> {
        TreeVec(zip(self.0, other.into().0).collect())
    }
    pub fn as_ref(&self) -> TreeVec<&T> {
        TreeVec(self.0.iter().collect())
    }
}
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
pub struct TreeColumns<T>(pub TreeVec<ColumnVec<T>>);
impl<T> TreeColumns<T> {
    pub fn new(values: Vec<Vec<T>>) -> Self {
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
    pub fn zip_cols<U>(self, other: impl Into<TreeColumns<U>>) -> TreeColumns<(T, U)> {
        TreeColumns(TreeVec(
            self.0
                 .0
                .into_iter()
                .zip(other.into().0 .0)
                .map(|(column1, column2)| zip(column1, column2).collect())
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
    pub fn flatten(self) -> Vec<T> {
        self.0 .0.into_iter().flatten().collect()
    }
}
impl<'a, T> From<&'a TreeColumns<T>> for TreeColumns<&'a T> {
    fn from(val: &'a TreeColumns<T>) -> Self {
        val.as_ref()
    }
}
impl<T> TreeColumns<Vec<T>> {
    pub fn flatten_all(self) -> Vec<T> {
        self.flatten().into_iter().flatten().collect()
    }
    pub fn flatten_all_rev(self) -> Vec<T> {
        self.flatten().into_iter().flatten().rev().collect()
    }
}
