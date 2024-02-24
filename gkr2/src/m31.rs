use std::fmt::Display;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use num_traits::{Num, One, Zero};
use prover_research::core::fields::m31::BaseField;
use prover_research::core::fields::qm31::SecureField;
use prover_research::core::fields::{ComplexConjugate, Field};

pub const MODULUS: u32 = (1 << 31) - 1;

#[derive(Debug, Clone, Copy)]
pub struct FastBaseField(pub u32);

impl From<BaseField> for FastBaseField {
    fn from(value: BaseField) -> Self {
        Self(value.0)
    }
}

impl From<FastBaseField> for BaseField {
    fn from(value: FastBaseField) -> Self {
        BaseField::from_u32_unchecked(value.into_integer())
    }
}

impl From<FastBaseField> for SecureField {
    fn from(value: FastBaseField) -> Self {
        BaseField::from(value).into()
    }
}

impl FastBaseField {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);

    pub fn into_integer(self) -> u32 {
        if self.is_zero() {
            0
        } else {
            self.0
        }
    }
}

impl Display for FastBaseField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.into_integer())
    }
}

impl Field for FastBaseField {
    fn double(&self) -> Self {
        Self(((self.0 << 1) & MODULUS) + (self.0 >> 30))
    }

    fn inverse(&self) -> Self {
        self.pow((MODULUS - 2).into())
    }
}

impl ComplexConjugate for FastBaseField {
    fn complex_conjugate(&self) -> Self {
        todo!()
    }
}

impl PartialEq for FastBaseField {
    fn eq(&self, other: &Self) -> bool {
        self.into_integer() == other.into_integer()
    }
}

impl Eq for FastBaseField {}

impl PartialOrd for FastBaseField {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.into_integer().partial_cmp(&other.into_integer())
    }
}

impl Ord for FastBaseField {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.into_integer().cmp(&other.into_integer())
    }
}

impl Zero for FastBaseField {
    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        self.0 == 0 || self.0 == MODULUS
    }
}

impl One for FastBaseField {
    fn one() -> Self {
        Self(1)
    }

    fn is_one(&self) -> bool {
        self.0 == 1
    }
}

impl Num for FastBaseField {
    type FromStrRadixErr = ();

    fn from_str_radix(_str: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        unimplemented!(
            "Num::from_str_radix is not implemented for {}",
            stringify!($field_name)
        );
    }
}

impl Add for FastBaseField {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.0 += rhs.0;
        self.0 = (self.0 & MODULUS) + (self.0 >> 31);
        self
    }
}

impl AddAssign for FastBaseField {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for FastBaseField {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self {
        self.0 += MODULUS - rhs.0;
        self.0 = (self.0 & MODULUS) + (self.0 >> 31);
        self
    }
}

impl SubAssign for FastBaseField {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for FastBaseField {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let t = self.0 as u64 * (rhs.0 << 1) as u64;
        // we want to truncate here
        #[allow(clippy::cast_possible_truncation)]
        let t0 = t as u32 >> 1;
        let t1 = (t >> 32) as u32;
        let x = t0 + t1;
        Self((x & MODULUS) + (x >> 31))
    }
}

impl MulAssign for FastBaseField {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for FastBaseField {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

impl DivAssign for FastBaseField {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Neg for FastBaseField {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(MODULUS - self.0)
    }
}

impl Rem for FastBaseField {
    type Output = Self;

    fn rem(self, _rhs: Self) -> Self::Output {
        unimplemented!("Rem is not implemented for {}", stringify!(FastBaseField));
    }
}

impl RemAssign for FastBaseField {
    fn rem_assign(&mut self, _rhs: Self) {
        unimplemented!("RemAssign is not implemented for {}", stringify!(Self));
    }
}
