use std::fmt::Display;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use serde::{Deserialize, Serialize};

use super::IntoSlice;
use crate::impl_field;

pub const K_BITS: u32 = 31;
pub const N_BYTES_FELT: usize = 4;
pub const P: u32 = 2147483647; // 2 ** 31 - 1

#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct M31(u32);
pub type BaseField = M31;

impl_field!(M31, P);

impl M31 {
    pub fn sqrt(&self) -> Option<Self> {
        let result = self.pow(1 << 29);
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
impl From<u16> for M31 {
    fn from(value: u16) -> Self {
        M31::from_u32_unchecked(value.into())
    }
}

impl From<i32> for M31 {
    fn from(value: i32) -> Self {
        M31::reduce(value.try_into().unwrap())
    }
}

// TODO(Ohad): Do not compile on non-le targets.
unsafe impl IntoSlice<u8> for M31 {
    fn into_slice(sl: &[Self]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(sl.as_ptr() as *const u8, std::mem::size_of_val(sl)) }
    }
}

#[cfg(test)]
#[macro_export]
macro_rules! m31 {
    ($m:expr) => {
        M31::from_u32_unchecked($m)
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
            .map(|_| M31::from(rng.gen::<u32>()))
            .collect::<Vec<M31>>();

        let slice = <M31 as IntoSlice<u8>>::into_slice(&x);

        for i in 0..100 {
            assert_eq!(
                x[i],
                M31::from_u32_unchecked(u32::from_le_bytes(
                    slice[i * 4..(i + 1) * 4].try_into().unwrap()
                ))
            );
        }
    }
}
