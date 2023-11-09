use crate::core::fields::cm31::CM31;
use std::fmt::Display;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub const P4: u128 = 21267647892944572736998860269687930881; // (2 ** 31 - 1) ** 4
pub const R: CM31 = CM31::from_u32_unchecked(1, 2);

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Extension field of CM31.
/// Equivalent to CM31\[x\] over (x^2 - 1 - 2i) as the irreducible polynomial.
/// Represented as ((a, b), (c, d)) of (a + bi) + (c + di)u.
pub struct QM31(CM31, CM31);

impl QM31 {
    pub fn square(&self) -> Self {
        (*self) * (*self)
    }

    pub fn double(&self) -> Self {
        (*self) + (*self)
    }

    pub fn pow(&self, exp: u128) -> Self {
        let mut res = Self::one();
        let mut base = *self;
        let mut exp = exp;
        while exp > 0 {
            if exp & 1 == 1 {
                res *= base;
            }
            base = base.square();
            exp >>= 1;
        }
        res
    }

    pub fn one() -> Self {
        Self(CM31::one(), CM31::zero())
    }

    pub fn zero() -> Self {
        Self(CM31::zero(), CM31::zero())
    }

    pub const fn from_u32_unchecked(a: u32, b: u32, c: u32, d: u32) -> Self {
        Self(
            CM31::from_u32_unchecked(a, b),
            CM31::from_u32_unchecked(c, d),
        )
    }

    pub fn inverse(&self) -> Self {
        assert!(*self != Self::zero(), "division by zero");
        self.pow(P4 - 2)
    }
}

impl Display for QM31 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} + {}u", self.0, self.1)
    }
}

impl Add for QM31 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl AddAssign for QM31 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Neg for QM31 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0, -self.1)
    }
}

impl Sub for QM31 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl SubAssign for QM31 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
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

impl MulAssign for QM31 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for QM31 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl DivAssign for QM31 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
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
        assert_eq!(x + y, QM31::from_u32_unchecked(5, 7, 9, 11));
    }

    #[test]
    fn test_multiplication() {
        let x = QM31::from_u32_unchecked(1, 2, 3, 4);
        let y = QM31::from_u32_unchecked(4, 5, 6, 7);
        assert_eq!(x * y, QM31::from_u32_unchecked(P - 106, 38, P - 16, 50));
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
