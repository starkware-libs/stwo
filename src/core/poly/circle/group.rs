use crate::core::circle::{CirclePoint, CirclePointIndex, Coset};
use crate::core::fields::m31::BaseField;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub coset: Coset,
}

impl Group {
    pub fn new(log_size: u32) -> Self {
        assert!(log_size > 0);
        Self {
            coset: Coset::subgroup(log_size),
        }
    }

    /// Gets the full coset represented <G_n>, where G_n is a generator of order 2^n.
    pub fn coset(&self) -> Coset {
        self.coset
    }

    /// Returns the log size of the group.
    pub fn log_size(&self) -> u32 {
        self.coset.log_size
    }

    /// Returns the size of the group.
    pub fn size(&self) -> usize {
        self.coset.size()
    }

    pub fn initial_index(&self) -> CirclePointIndex {
        self.coset.initial_index
    }

    pub fn step_size(&self) -> CirclePointIndex {
        self.coset.step_size
    }

    pub fn index_at(&self, index: usize) -> CirclePointIndex {
        self.coset.index_at(index)
    }

    pub fn at(&self, i: usize) -> CirclePoint<BaseField> {
        self.coset.at(i)
    }
}
