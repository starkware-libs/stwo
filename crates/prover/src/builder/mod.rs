mod assert;
mod domain;
mod info;
mod point;

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Sub};

pub use assert::AssertEvaluator;
pub use domain::DomainEvaluator;
pub use info::InfoEvaluator;
use num_traits::{One, Zero};
pub use point::PointEvaluator;

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;

pub trait EvalAtRow {
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
    type EF: One
        + Copy
        + Debug
        + Zero
        + Add<SecureField, Output = Self::EF>
        + Sub<SecureField, Output = Self::EF>
        + Mul<SecureField, Output = Self::EF>
        + Add<Self::F, Output = Self::EF>
        + Mul<Self::F, Output = Self::EF>
        + Sub<Self::EF, Output = Self::EF>
        + Mul<Self::EF, Output = Self::EF>;

    fn next_mask(&mut self) -> Self::F {
        self.next_interaction_mask(0, [0])[0]
    }
    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N];
    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF>;
    fn combine_ef(values: [Self::F; 4]) -> Self::EF;
}
