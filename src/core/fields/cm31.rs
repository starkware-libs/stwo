use crate::core::circle::CirclePoint;
use crate::core::fields::m31::M31;
use crate::core::fields::m31::P;
use std::fmt::Display;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::DivAssign;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Neg;
use std::ops::Sub;
use std::ops::SubAssign;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Complex extension field of M31.
pub struct CM31(M31, M31);

impl CM31 {
    pub fn square(&self) -> Self {
        (*self) * (*self)
    }

    pub fn double(&self) -> CM31 {
        (*self) + (*self)
    }

    pub fn pow(&self, exp: u64) -> Self {
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

    pub fn one() -> CM31 {
        Self(M31::one(), M31::zero())
    }

    pub fn zero() -> CM31 {
        Self(M31::zero(), M31::zero())
    }

    pub const fn from_u32_unchecked(a: u32, b: u32) -> CM31 {
        Self(M31::from_u32_unchecked(a), M31::from_u32_unchecked(b))
    }

    pub fn inverse(&self) -> CM31 {
        assert!(*self != Self::zero(), "division by zero");
        self.pow((P as u64).pow(2) - 2)
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

impl Div for CM31 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl DivAssign for CM31 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl From<CirclePoint> for CM31 {
    fn from(p: CirclePoint) -> Self {
        Self(p.x, p.y)
    }
}

#[cfg(test)]
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

#[test]
fn test_division() {
    let x = CM31(M31::from_u32_unchecked(P - 6), M31::from_u32_unchecked(13));
    let y = CM31(M31::from_u32_unchecked(4), M31::from_u32_unchecked(5));
    assert_eq!(
        x / y,
        CM31(
            M31::from_u32_unchecked(1),
            M31::from_u32_unchecked(2)
        )
    );
}
