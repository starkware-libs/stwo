use std::fmt::Display;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use bytemuck::{Pod, Zeroable};

use super::{ComplexConjugate, FieldExpOps};
use crate::impl_field;

pub const MODULUS_BITS: u32 = 31;
pub const N_BYTES_FELT: usize = 4;
pub const P: u32 = 2147483647; // 2 ** 31 - 1

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct M31(pub u32);
pub type BaseField = M31;

impl_field!(M31, P);

impl M31 {
    pub fn sqrt(&self) -> Option<Self> {
        let result = self.pow(1 << 29);
        (result.square() == *self).then_some(result)
    }

    /// Assumes that `val` is in the range [0, 2 * `P`) and returns `val` % `P`.
    pub fn partial_reduce(val: u32) -> Self {
        Self(val.checked_sub(P).unwrap_or(val))
    }

    /// Assumes that `val` is in the range [0, `P`.pow(2)) and returns `val` % `P`.
    pub fn reduce(val: u64) -> Self {
        Self((((((val >> MODULUS_BITS) + val + 1) >> MODULUS_BITS) + val) & (P as u64)) as u32)
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
        Self::partial_reduce(self.0 + rhs.0)
    }
}

impl Neg for M31 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::partial_reduce(P - self.0)
    }
}

impl Sub for M31 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::partial_reduce(self.0 + P - rhs.0)
    }
}

impl Mul for M31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::reduce((self.0 as u64) * (rhs.0 as u64))
    }
}

impl ComplexConjugate for M31 {
    fn complex_conjugate(&self) -> Self {
        *self
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

impl From<usize> for M31 {
    fn from(value: usize) -> Self {
        M31::reduce(value.try_into().unwrap())
    }
}

impl From<u32> for M31 {
    fn from(value: u32) -> Self {
        M31::reduce(value.into())
    }
}

impl From<i32> for M31 {
    fn from(value: i32) -> Self {
        M31::reduce(value.try_into().unwrap())
    }
}

#[cfg(test)]
#[macro_export]
macro_rules! m31 {
    ($m:expr) => {
        $crate::core::fields::m31::M31::from_u32_unchecked($m)
    };
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::{M31, P};
    use crate::core::fields::IntoSlice;

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
            assert_eq!(m31!(add_p(x, y)), m31!(x) + m31!(y));
            assert_eq!(m31!(mul_p(x, y)), m31!(x) * m31!(y));
            assert_eq!(m31!(neg_p(x)), -m31!(x));
        }
    }

    #[test]
    fn test_into_slice() {
        let mut rng = rand::thread_rng();
        let x = (0..100)
            .map(|_| m31!(rng.gen::<u32>()))
            .collect::<Vec<M31>>();

        let slice = M31::into_slice(&x);

        for i in 0..100 {
            assert_eq!(
                x[i],
                m31!(u32::from_le_bytes(
                    slice[i * 4..(i + 1) * 4].try_into().unwrap()
                ))
            );
        }
    }
}
