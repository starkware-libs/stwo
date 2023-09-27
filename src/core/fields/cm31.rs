use crate::core::fields::m31::M31;
use std::ops::Add;
use std::ops::Mul;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
// Complex extension field of M31.
pub struct CM31(M31, M31);

impl Add for CM31 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl Mul for CM31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(
            self.0 * rhs.0 - self.1 * rhs.1,
            self.0 * rhs.1 + self.1 * rhs.0,
        )
    }
}

#[cfg(test)]
use crate::core::fields::m31::P;

#[test]
fn test_addition() {
    let x = CM31(M31::from_u32_unchecked(1), M31::from_u32_unchecked(2));
    let y = CM31(M31::from_u32_unchecked(4), M31::from_u32_unchecked(5));
    assert_eq!(
        x + y,
        CM31(M31::from_u32_unchecked(5), M31::from_u32_unchecked(7))
    );
}

#[test]
fn test_multiplication() {
    let x = CM31(M31::from_u32_unchecked(1), M31::from_u32_unchecked(2));
    let y = CM31(M31::from_u32_unchecked(4), M31::from_u32_unchecked(5));
    assert_eq!(
        x * y,
        CM31(M31::from_u32_unchecked(P - 6), M31::from_u32_unchecked(13))
    );
}
