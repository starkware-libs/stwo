use std::fmt::Display;
use std::iter::{Product, Sum};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use num_traits::{Num, One, Zero};
use prover_research::core::fields::cm31::CM31;
use prover_research::core::fields::m31::BaseField;
use prover_research::core::fields::{ComplexConjugate, ExtensionOf, Field};

use crate::m31::{FastBaseField, MODULUS};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct C31(pub FastBaseField, pub FastBaseField);

impl From<CM31> for C31 {
    fn from(value: CM31) -> Self {
        Self(value.0.into(), value.1.into())
    }
}

impl From<BaseField> for C31 {
    fn from(value: BaseField) -> Self {
        Self(value.into(), FastBaseField::zero())
    }
}

impl From<C31> for CM31 {
    fn from(value: C31) -> Self {
        CM31(value.0.into(), value.1.into())
    }
}

// impl From<C31> for SecureField {
//     fn from(value: C31) -> SecureField {
//         CM31::from(value).into()
//     }
// }

impl ExtensionOf<FastBaseField> for C31 {}

impl Display for C31 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} + {}i", self.0, self.1)
    }
}

impl Mul for C31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i.
        Self(
            self.0 * rhs.0 - self.1 * rhs.1,
            self.0 * rhs.1 + self.1 * rhs.0,
        )
    }
}

impl Add for C31 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl Neg for C31 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0, -self.1)
    }
}

impl Sub for C31 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl One for C31 {
    fn one() -> Self {
        Self(FastBaseField::one(), FastBaseField::zero())
    }
}

impl Zero for C31 {
    fn zero() -> Self {
        Self(FastBaseField::zero(), FastBaseField::zero())
    }

    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl Add<FastBaseField> for C31 {
    type Output = Self;

    fn add(self, rhs: FastBaseField) -> Self::Output {
        Self(self.0 + rhs, self.1)
    }
}

impl Add<C31> for FastBaseField {
    type Output = C31;

    fn add(self, rhs: C31) -> Self::Output {
        rhs + self
    }
}

impl Sub<FastBaseField> for C31 {
    type Output = Self;

    fn sub(self, rhs: FastBaseField) -> Self::Output {
        Self(self.0 - rhs, self.1)
    }
}

impl Sub<C31> for FastBaseField {
    type Output = C31;

    fn sub(self, rhs: C31) -> Self::Output {
        -rhs + self
    }
}

impl Mul<FastBaseField> for C31 {
    type Output = Self;

    fn mul(self, rhs: FastBaseField) -> Self::Output {
        Self(self.0 * rhs, self.1 * rhs)
    }
}

impl Mul<C31> for FastBaseField {
    type Output = C31;

    fn mul(self, rhs: C31) -> Self::Output {
        rhs * self
    }
}

impl Div<FastBaseField> for C31 {
    type Output = Self;

    fn div(self, rhs: FastBaseField) -> Self::Output {
        Self(self.0 / rhs, self.1 / rhs)
    }
}

impl Div<C31> for FastBaseField {
    type Output = C31;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: C31) -> Self::Output {
        rhs.inverse() * self
    }
}

impl ComplexConjugate for C31 {
    fn complex_conjugate(&self) -> Self {
        Self(self.0, -self.1)
    }
}

impl From<FastBaseField> for C31 {
    fn from(x: FastBaseField) -> Self {
        Self(x, FastBaseField::zero())
    }
}

impl AddAssign<FastBaseField> for C31 {
    fn add_assign(&mut self, rhs: FastBaseField) {
        *self = *self + rhs;
    }
}

impl SubAssign<FastBaseField> for C31 {
    fn sub_assign(&mut self, rhs: FastBaseField) {
        *self = *self - rhs;
    }
}

impl MulAssign<FastBaseField> for C31 {
    fn mul_assign(&mut self, rhs: FastBaseField) {
        *self = *self * rhs;
    }
}

impl DivAssign<FastBaseField> for C31 {
    fn div_assign(&mut self, rhs: FastBaseField) {
        *self = *self / rhs;
    }
}

impl Rem<FastBaseField> for C31 {
    type Output = Self;

    fn rem(self, _rhs: FastBaseField) -> Self::Output {
        unimplemented!("Rem is not implemented for {}", stringify!(C31));
    }
}

impl RemAssign<FastBaseField> for C31 {
    fn rem_assign(&mut self, _rhs: FastBaseField) {
        unimplemented!("RemAssign is not implemented for {}", stringify!(C31));
    }
}

impl Sum for C31 {
    fn sum<I>(mut iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        let first = iter.next().unwrap_or_else(<C31>::zero);
        iter.fold(first, |a, b| a + b)
    }
}

impl Product for C31 {
    fn product<I>(mut iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        let first = iter.next().unwrap_or_else(<C31>::one);
        iter.fold(first, |a, b| a * b)
    }
}

impl<'a> Sum<&'a C31> for C31 {
    fn sum<I>(mut iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        let first = iter.next().copied().unwrap_or_else(<C31>::zero);
        iter.fold(first, |a, &b| a + b)
    }
}

impl<'a> Product<&'a C31> for C31 {
    fn product<I>(mut iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        let first = iter.next().copied().unwrap_or_else(<C31>::one);
        iter.fold(first, |a, &b| a * b)
    }
}

impl Num for C31 {
    type FromStrRadixErr = ();

    fn from_str_radix(_str: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        unimplemented!(
            "Num::from_str_radix is not implemented for {}",
            stringify!(C31)
        );
    }
}

impl Field for C31 {
    fn double(&self) -> Self {
        Self(self.0.double(), self.1.double())
    }

    fn inverse(&self) -> Self {
        assert!(*self != Self::zero(), "0 has no inverse");
        self.pow(((MODULUS as u64).pow(2) - 2) as u128)
    }
}

impl AddAssign for C31 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for C31 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for C31 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for C31 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl DivAssign for C31 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Rem for C31 {
    type Output = Self;

    fn rem(self, _rhs: Self) -> Self::Output {
        unimplemented!("Rem is not implemented for {}", stringify!(C31));
    }
}

impl RemAssign for C31 {
    fn rem_assign(&mut self, _rhs: Self) {
        unimplemented!("RemAssign is not implemented for {}", stringify!(C31));
    }
}
