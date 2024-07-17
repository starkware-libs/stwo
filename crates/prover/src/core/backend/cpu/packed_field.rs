use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use num_traits::{One, Zero};

use crate::core::fields::Field;

/// Unified interface for operations commonly performed on scalar and packed (SIMD) types.
///
/// # Safety
///
/// Should only be implemented on types that store a power of two many items.
pub unsafe trait PackedField:
    Sized
    + Copy
    + Debug
    + Zero
    + One
    + AddAssign
    + SubAssign
    + MulAssign
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
{
    type Field: Field;

    fn deinterleave(self, other: Self) -> (Self, Self);

    fn interleave(self, other: Self) -> (Self, Self);

    fn double(self) -> Self;

    fn pointwise_sum(self) -> Self::Field;
}

unsafe impl<F: Field> PackedField for F {
    type Field = F;

    fn deinterleave(self, other: Self) -> (Self, Self) {
        (self, other)
    }

    fn interleave(self, other: Self) -> (Self, Self) {
        (self, other)
    }

    fn pointwise_sum(self) -> Self::Field {
        self
    }

    fn double(self) -> Self {
        F::double(&self)
    }
}
