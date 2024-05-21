use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut, Index};

use self::fields::m31::BaseField;

pub mod air;
pub mod backend;
pub mod channel;
pub mod circle;
pub mod constraints;
pub mod fft;
pub mod fields;
pub mod fri;
pub mod lookups;
pub mod pcs;
pub mod poly;
pub mod proof_of_work;
pub mod prover;
pub mod queries;
#[cfg(test)]
pub mod test_utils;
pub mod utils;
pub mod vcs;

/// A vector in which each element relates (by index) to a column in the trace.
pub type ColumnVec<T> = Vec<T>;

/// A vector of [ColumnVec]s. Each [ColumnVec] relates (by index) to a component in the air.
#[derive(Debug, Clone)]
pub struct ComponentVec<T>(pub Vec<ColumnVec<T>>);

impl<T> ComponentVec<T> {
    pub fn flatten(self) -> ColumnVec<T> {
        self.0.into_iter().flatten().collect()
    }
}

impl<T> ComponentVec<ColumnVec<T>> {
    pub fn flatten_cols(self) -> Vec<T> {
        self.0.into_iter().flatten().flatten().collect()
    }
}

impl<T> Default for ComponentVec<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T> Deref for ComponentVec<T> {
    type Target = Vec<ColumnVec<T>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for ComponentVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Default)]
pub struct InteractionElements(BTreeMap<String, BaseField>);

impl InteractionElements {
    pub fn new(elements: BTreeMap<String, BaseField>) -> Self {
        Self(elements)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Index<&str> for InteractionElements {
    type Output = BaseField;

    fn index(&self, index: &str) -> &Self::Output {
        // TODO(AlonH): Return an error if the key is not found.
        &self.0[index]
    }
}
