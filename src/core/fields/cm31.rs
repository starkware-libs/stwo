use crate::core::fields::m31::M31;
use crate::impl_field;
use num_traits::{Num, One, Zero};
use std::fmt::Display;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

pub const P2: u64 = 4611686014132420609; // (2 ** 31 - 1) ** 2

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Complex extension field of M31.
/// Equivalent to M31\[x\] over (x^2 + 1) as the irreducible polynomial.
/// Represented as (a, b) of a + bi.
pub struct CM31(M31, M31);

impl_field!(CM31, P2);

impl CM31 {
    pub const fn from_u32_unchecked(a: u32, b: u32) -> CM31 {
        Self(M31::from_u32_unchecked(a), M31::from_u32_unchecked(b))
    }
}

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

impl One for CM31 {
    fn one() -> Self {
        Self(M31::one(), M31::zero())
    }
}

impl Add<M31> for CM31 {
    type Output = Self;

    fn add(self, rhs: M31) -> Self::Output {
        Self(self.0 + rhs, self.1)
    }
}

impl Mul<M31> for CM31 {
    type Output = Self;

    fn mul(self, rhs: M31) -> Self::Output {
        Self(self.0 * rhs, self.1 * rhs)
    }
}

impl Zero for CM31 {
    fn zero() -> Self {
        Self(M31::zero(), M31::zero())
    }

    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fields::m31::P;

    #[test]
    fn test_addition() {
        let x = CM31::from_u32_unchecked(1, 2);
        let y = CM31::from_u32_unchecked(4, 5);
        assert_eq!(x + y, CM31::from_u32_unchecked(5, 7));
    }

    #[test]
    fn test_multiplication() {
        let x = CM31::from_u32_unchecked(1, 2);
        let y = CM31::from_u32_unchecked(4, 5);
        assert_eq!(x * y, CM31::from_u32_unchecked(P - 6, 13));
    }

    #[test]
    fn test_negation() {
        let x = CM31::from_u32_unchecked(1, 2);
        assert_eq!(-x, CM31::from_u32_unchecked(P - 1, P - 2));
    }

    #[test]
    fn test_subtraction() {
        let x = CM31::from_u32_unchecked(1, 2);
        let y = CM31::from_u32_unchecked(4, 5);
        assert_eq!(x - y, CM31::from_u32_unchecked(P - 3, P - 3));
    }

    #[test]
    fn test_division() {
        let x = CM31::from_u32_unchecked(P - 6, 13);
        let y = CM31::from_u32_unchecked(4, 5);
        assert_eq!(x / y, CM31::from_u32_unchecked(1, 2));
    }
}
