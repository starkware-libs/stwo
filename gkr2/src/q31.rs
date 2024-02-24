use std::fmt::Display;
use std::iter::{Product, Sum};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use num_traits::{Num, One, Zero};
use prover_research::core::fields::m31::BaseField;
use prover_research::core::fields::qm31::{SecureField, QM31};
use prover_research::core::fields::{ComplexConjugate, ExtensionOf, Field};

use crate::c31::C31;
use crate::m31::{FastBaseField, MODULUS};

/// Can be safely transmuted from [`SecureField`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FastSecureField(C31, C31);

impl From<SecureField> for FastSecureField {
    fn from(value: SecureField) -> Self {
        Self(value.0.into(), value.1.into())
    }
}

impl From<BaseField> for FastSecureField {
    fn from(value: BaseField) -> Self {
        Self(value.into(), C31::zero())
    }
}

impl From<FastSecureField> for SecureField {
    fn from(value: FastSecureField) -> Self {
        QM31(value.0.into(), value.1.into())
    }
}

impl ExtensionOf<FastBaseField> for FastSecureField {}

impl Display for FastSecureField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) + ({})u", self.0, self.1)
    }
}

pub const R: C31 = C31(FastBaseField(1), FastBaseField(2));

impl Mul for FastSecureField {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // = (x + i * y) * (1 + 2 * i)
        // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i.
        // = (x - 2y) + (2x + y)i
        // = (x - y.double()) + (x.double + )

        // (a + bu) * (c + du) = (ac + rbd) + (ad + bc)u.
        // let tmp = self.1 * rhs.1;

        Self(
            // self.0 * rhs.0 + C31(tmp.0 - tmp.1.double(), tmp.0.double() + tmp.1),
            self.0 * rhs.0 + R * self.1 * rhs.1,
            self.0 * rhs.1 + self.1 * rhs.0,
        )
    }
}

impl Add for FastSecureField {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl Neg for FastSecureField {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0, -self.1)
    }
}

impl Sub for FastSecureField {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl One for FastSecureField {
    fn one() -> Self {
        Self(C31::one(), C31::zero())
    }
}

impl Zero for FastSecureField {
    fn zero() -> Self {
        Self(C31::zero(), C31::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero() && self.1.is_zero()
    }
}

impl Add<FastBaseField> for FastSecureField {
    type Output = Self;

    fn add(self, rhs: FastBaseField) -> Self::Output {
        Self(self.0 + rhs, self.1)
    }
}

impl Add<FastSecureField> for FastBaseField {
    type Output = FastSecureField;

    fn add(self, rhs: FastSecureField) -> Self::Output {
        rhs + self
    }
}

impl Sub<FastBaseField> for FastSecureField {
    type Output = Self;

    fn sub(self, rhs: FastBaseField) -> Self::Output {
        Self(self.0 - rhs, self.1)
    }
}

impl Sub<FastSecureField> for FastBaseField {
    type Output = FastSecureField;

    fn sub(self, rhs: FastSecureField) -> Self::Output {
        -rhs + self
    }
}

impl Mul<FastBaseField> for FastSecureField {
    type Output = Self;

    fn mul(self, rhs: FastBaseField) -> Self::Output {
        Self(self.0 * rhs, self.1 * rhs)
    }
}

impl Mul<FastSecureField> for FastBaseField {
    type Output = FastSecureField;

    fn mul(self, rhs: FastSecureField) -> Self::Output {
        rhs * self
    }
}

impl Div<FastBaseField> for FastSecureField {
    type Output = Self;

    fn div(self, rhs: FastBaseField) -> Self::Output {
        Self(self.0 / rhs, self.1 / rhs)
    }
}

impl Div<FastSecureField> for FastBaseField {
    type Output = FastSecureField;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: FastSecureField) -> Self::Output {
        rhs.inverse() * self
    }
}

impl ComplexConjugate for FastSecureField {
    fn complex_conjugate(&self) -> Self {
        Self(self.0, -self.1)
    }
}

impl From<FastBaseField> for FastSecureField {
    fn from(x: FastBaseField) -> Self {
        Self(x.into(), C31::zero())
    }
}

impl AddAssign<FastBaseField> for FastSecureField {
    fn add_assign(&mut self, rhs: FastBaseField) {
        *self = *self + rhs;
    }
}

impl SubAssign<FastBaseField> for FastSecureField {
    fn sub_assign(&mut self, rhs: FastBaseField) {
        *self = *self - rhs;
    }
}

impl MulAssign<FastBaseField> for FastSecureField {
    fn mul_assign(&mut self, rhs: FastBaseField) {
        *self = *self * rhs;
    }
}

impl DivAssign<FastBaseField> for FastSecureField {
    fn div_assign(&mut self, rhs: FastBaseField) {
        *self = *self / rhs;
    }
}

impl Rem<FastBaseField> for FastSecureField {
    type Output = Self;

    fn rem(self, _rhs: FastBaseField) -> Self::Output {
        unimplemented!("Rem is not implemented for {}", stringify!(SecureField));
    }
}

impl RemAssign<FastBaseField> for FastSecureField {
    fn rem_assign(&mut self, _rhs: FastBaseField) {
        unimplemented!(
            "RemAssign is not implemented for {}",
            stringify!(SecureField)
        );
    }
}

impl Sum for FastSecureField {
    fn sum<I>(mut iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        let first = iter.next().unwrap_or_else(<FastSecureField>::zero);
        iter.fold(first, |a, b| a + b)
    }
}

impl Product for FastSecureField {
    fn product<I>(mut iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        let first = iter.next().unwrap_or_else(<FastSecureField>::one);
        iter.fold(first, |a, b| a * b)
    }
}

impl<'a> Sum<&'a FastSecureField> for FastSecureField {
    fn sum<I>(mut iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        let first = iter.next().copied().unwrap_or_else(<FastSecureField>::zero);
        iter.fold(first, |a, &b| a + b)
    }
}

impl<'a> Product<&'a FastSecureField> for FastSecureField {
    fn product<I>(mut iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        let first = iter.next().copied().unwrap_or_else(<FastSecureField>::one);
        iter.fold(first, |a, &b| a * b)
    }
}

impl Num for FastSecureField {
    type FromStrRadixErr = ();

    fn from_str_radix(_str: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        unimplemented!(
            "Num::from_str_radix is not implemented for {}",
            stringify!(SecureField)
        );
    }
}

impl Field for FastSecureField {
    fn double(&self) -> Self {
        Self(self.0.double(), self.1.double())
    }

    fn inverse(&self) -> Self {
        assert!(*self != Self::zero(), "0 has no inverse");
        self.pow((MODULUS as u128).pow(4) - 2)
    }
}

impl AddAssign for FastSecureField {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for FastSecureField {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for FastSecureField {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for FastSecureField {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl DivAssign for FastSecureField {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Rem for FastSecureField {
    type Output = Self;

    fn rem(self, _rhs: Self) -> Self::Output {
        unimplemented!("Rem is not implemented for {}", stringify!(SecureField));
    }
}

impl RemAssign for FastSecureField {
    fn rem_assign(&mut self, _rhs: Self) {
        unimplemented!(
            "RemAssign is not implemented for {}",
            stringify!(SecureField)
        );
    }
}
