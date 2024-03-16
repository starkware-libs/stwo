use std::ops::{Deref, DerefMut};

pub mod air;
pub mod backend;
pub mod channel;
pub mod circle;
pub mod commitment_scheme;
pub mod constraints;
pub mod fft;
pub mod fields;
pub mod fri;
pub mod lookups;
pub mod oods;
pub mod poly;
pub mod proof_of_work;
pub mod prover;
pub mod queries;
pub mod utils;

/// A vector in which each element relates (by index) to a column in the trace.
pub type ColumnVec<T> = Vec<T>;

/// A vector of [ColumnVec]s. Each [ColumnVec] relates (by index) to a component in the air.
pub struct ComponentVec<T>(pub Vec<ColumnVec<T>>);

impl<T: Copy> ComponentVec<ColumnVec<T>> {
    pub fn flatten(&self) -> Vec<T> {
        self.iter().flatten().flatten().copied().collect()
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
