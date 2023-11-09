use crate::impl_field;
use num_traits::{Num, One, Zero};
use std::fmt::Display;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

pub const P: u32 = 2147483647; // 2 ** 31 - 1
pub const K_BITS: u32 = 31;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct M31(u32);
pub type Field = M31;

impl_field!(M31, u32, P);

impl M31 {
    pub fn sqrt(&self) -> Option<Self> {
        let result = self.pow((1 << 29) as u32);
        (result.square() == *self).then_some(result)
    }

    pub fn reduce(val: u64) -> Self {
        Self((((((val >> K_BITS) + val + 1) >> K_BITS) + val) & (P as u64)) as u32)
    }

    pub const fn from_u32_unchecked(arg: u32) -> Self {
        Self(arg)
    }
}

impl Display for M31 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Add for M31 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::reduce((self.0 as u64) + (rhs.0 as u64))
    }
}

impl Neg for M31 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::reduce(P as u64 - (self.0 as u64))
    }
}

impl Sub for M31 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::reduce((self.0 as u64) + (P as u64) - (rhs.0 as u64))
    }
}

impl Mul for M31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::reduce((self.0 as u64) * (rhs.0 as u64))
    }
}

impl One for M31 {
    fn one() -> Self {
        Self(1)
    }
}

impl Zero for M31 {
    fn zero() -> Self {
        Self(0)
    }

    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn mul_p(a: u32, b: u32) -> u32 {
        ((a as u64 * b as u64) % P as u64) as u32
    }

    fn add_p(a: u32, b: u32) -> u32 {
        (a + b) % P
    }

    fn neg_p(a: u32) -> u32 {
        if a == 0 {
            0
        } else {
            P - a
        }
    }

    #[test]
    fn test_basic_ops() {
        let mut rng = rand::thread_rng();
        for _ in 0..10000 {
            let x: u32 = rng.gen::<u32>() % P;
            let y: u32 = rng.gen::<u32>() % P;
            assert_eq!(
                M31::from_u32_unchecked(add_p(x, y)),
                M31::from_u32_unchecked(x) + M31::from_u32_unchecked(y)
            );
            assert_eq!(
                M31::from_u32_unchecked(mul_p(x, y)),
                M31::from_u32_unchecked(x) * M31::from_u32_unchecked(y)
            );
            assert_eq!(
                M31::from_u32_unchecked(neg_p(x)),
                -M31::from_u32_unchecked(x)
            );
        }
    }
}
