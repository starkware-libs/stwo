use crate::core::fields::m31::M31;
use num_traits::{Num, One, Zero};

use crate::core::fields::cm31::CM31;
use crate::{impl_extension_field, impl_field};
use std::fmt::Display;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

pub const P4: u128 = 21267647892944572736998860269687930881; // (2 ** 31 - 1) ** 4
pub const R: CM31 = CM31::from_u32_unchecked(1, 2);

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Extension field of CM31.
/// Equivalent to CM31\[x\] over (x^2 - 1 - 2i) as the irreducible polynomial.
/// Represented as ((a, b), (c, d)) of (a + bi) + (c + di)u.
pub struct QM31(CM31, CM31);

impl_field!(QM31, P4);
impl_extension_field!(QM31, CM31);

impl QM31 {
    pub const fn from_u32_unchecked(a: u32, b: u32, c: u32, d: u32) -> Self {
        Self(
            CM31::from_u32_unchecked(a, b),
            CM31::from_u32_unchecked(c, d),
        )
    }
}

impl Display for QM31 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} + {}u", self.0, self.1)
    }
}

impl Mul for QM31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // (a + bu) * (c + du) = (ac + rbd) + (ad + bc)u.
        Self(
            self.0 * rhs.0 + R * self.1 * rhs.1,
            self.0 * rhs.1 + self.1 * rhs.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fields::m31::P;

    #[test]
    fn test_addition() {
        let x = QM31::from_u32_unchecked(1, 2, 3, 4);
        let y = QM31::from_u32_unchecked(4, 5, 6, 7);
        let m = M31::from_u32_unchecked(8);
        let q = QM31::from_u32_unchecked(8, 0, 0, 0);
        assert_eq!(x + y, QM31::from_u32_unchecked(5, 7, 9, 11));
        assert_eq!(y + m, y + q);
    }

    #[test]
    fn test_multiplication() {
        let x = QM31::from_u32_unchecked(1, 2, 3, 4);
        let y = QM31::from_u32_unchecked(4, 5, 6, 7);
        let m = M31::from_u32_unchecked(8);
        let q = QM31::from_u32_unchecked(8, 0, 0, 0);
        assert_eq!(x * y, QM31::from_u32_unchecked(P - 106, 38, P - 16, 50));
        assert_eq!(y * m, y * q);
    }

    #[test]
    fn test_negation() {
        let x = QM31::from_u32_unchecked(1, 2, 3, 4);
        assert_eq!(-x, QM31::from_u32_unchecked(P - 1, P - 2, P - 3, P - 4));
    }

    #[test]
    fn test_subtraction() {
        let x = QM31::from_u32_unchecked(1, 2, 3, 4);
        let y = QM31::from_u32_unchecked(4, 5, 6, 7);
        assert_eq!(x - y, QM31::from_u32_unchecked(P - 3, P - 3, P - 3, P - 3));
    }

    #[test]
    fn test_division() {
        let x = QM31::from_u32_unchecked(P - 106, 38, P - 16, 50);
        let y = QM31::from_u32_unchecked(4, 5, 6, 7);
        assert_eq!(x / y, QM31::from_u32_unchecked(1, 2, 3, 4));
    }
}
