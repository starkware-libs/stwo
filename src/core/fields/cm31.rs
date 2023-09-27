use crate::core::fields::m31::M31;
use std::fmt::Display;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Neg;
use std::ops::Sub;
use std::ops::SubAssign;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Complex extension field of M31.
/// Equivalent to M31\[x\] over (x^2 + 1) as the irreducible polynomial.
/// Represented as (a, b) of a + bi.
pub struct CM31(M31, M31);

impl Display for CM31 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} + {}i", self.0, self.1)
    }
}

impl Add for CM31 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl AddAssign for CM31 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Neg for CM31 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0, -self.1)
    }
}

impl Sub for CM31 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl SubAssign for CM31 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for CM31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i.
        Self(
            self.0 * rhs.0 - self.1 * rhs.1,
            self.0 * rhs.1 + self.1 * rhs.0,
        )
    }
}

impl MulAssign for CM31 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
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

#[test]
fn test_negation() {
    let x = CM31(M31::from_u32_unchecked(1), M31::from_u32_unchecked(2));
    assert_eq!(
        -x,
        CM31(
            M31::from_u32_unchecked(P - 1),
            M31::from_u32_unchecked(P - 2)
        )
    );
}

#[test]
fn test_subtraction() {
    let x = CM31(M31::from_u32_unchecked(1), M31::from_u32_unchecked(2));
    let y = CM31(M31::from_u32_unchecked(4), M31::from_u32_unchecked(5));
    assert_eq!(
        x - y,
        CM31(
            M31::from_u32_unchecked(P - 3),
            M31::from_u32_unchecked(P - 3)
        )
    );
}
