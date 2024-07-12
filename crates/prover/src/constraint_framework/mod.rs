/// ! This module contains helpers to express and use constraints for components.
mod assert;
pub mod constant_cols;
mod domain;
mod info;
pub mod logup;
mod point;

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Neg, Sub};

pub use assert::{assert_constraints, AssertEvaluator};
pub use domain::DomainEvaluator;
pub use info::InfoEvaluator;
use num_traits::{One, Zero};
pub use point::PointEvaluator;

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::fields::FieldExpOps;

/// A trait for evaluating expressions at some point or row.
pub trait EvalAtRow {
    // TODO(spapini): Use a better trait for these, like 'Algebra' or something.
    /// The base field type.
    type F: FieldExpOps
        + Copy
        + Debug
        + AddAssign<Self::F>
        + AddAssign<BaseField>
        + Add<Self::F, Output = Self::F>
        + Sub<Self::F, Output = Self::F>
        + Mul<BaseField, Output = Self::F>
        + Add<SecureField, Output = Self::EF>
        + Mul<SecureField, Output = Self::EF>
        + From<BaseField>;

    /// The extension field type.
    type EF: One
        + Copy
        + Debug
        + Zero
        + Neg<Output = Self::EF>
        + Add<SecureField, Output = Self::EF>
        + Sub<SecureField, Output = Self::EF>
        + Mul<SecureField, Output = Self::EF>
        + Add<Self::F, Output = Self::EF>
        + Mul<Self::F, Output = Self::EF>
        + Sub<Self::EF, Output = Self::EF>
        + Mul<Self::EF, Output = Self::EF>;

    /// Returns the next mask value.
    fn next_mask(&mut self) -> Self::F {
        self.next_interaction_mask(0, [0])[0]
    }

    /// Returns the next mask values for a given interaction.
    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N];

    /// Adds a constraint to the component.
    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF>;

    /// Combines 4 base field values into a single extension field value.
    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF;
}
