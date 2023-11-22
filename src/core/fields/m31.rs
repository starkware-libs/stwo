use std::fmt::Display;
use std::io::Write;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use super::byte_translate::{ByteTranslate, ByteTranslateError};
use crate::impl_field;

pub const P: u32 = 2147483647; // 2 ** 31 - 1
pub const K_BITS: u32 = 31;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

impl ByteTranslate for M31 {
    const LENGTH_BYTES: usize = 4;

    fn to_le_bytes(self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }

    fn to_be_bytes(self) -> Vec<u8> {
        self.0.to_be_bytes().to_vec()
    }

    fn write_le_bytes(&self, mut dst: &mut [u8]) {
        assert!(
            dst.len() >= Self::LENGTH_BYTES,
            "write_le_bytes failed, dst.len() = {}",
            dst.len()
        );
        match dst.write(self.0.to_le_bytes().as_ref()) {
            Ok(_) => {}
            Err(_) => panic!("write_le_bytes failed"),
        }
    }

    fn write_be_bytes(&self, mut dst: &mut [u8]) {
        assert!(
            dst.len() >= Self::LENGTH_BYTES,
            "write_be_bytes failed, dst.len() = {}",
            dst.len()
        );
        match dst.write(self.0.to_be_bytes().as_ref()) {
            Ok(_) => {}
            Err(_) => panic!("write_be_bytes failed"),
        }
    }

    fn read_le_bytes(src: &[u8]) -> Result<Self, ByteTranslateError> {
        if src.len() < Self::LENGTH_BYTES {
            return Err(ByteTranslateError::InvalidLength);
        }
        Ok(Self(u32::from_le_bytes(
            src[..Self::LENGTH_BYTES].try_into().unwrap(),
        )))
    }

    fn read_be_bytes(src: &[u8]) -> Result<Self, ByteTranslateError> {
        if src.len() < Self::LENGTH_BYTES {
            return Err(ByteTranslateError::InvalidLength);
        }
        Ok(Self(u32::from_be_bytes(
            src[..Self::LENGTH_BYTES].try_into().unwrap(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::{M31, P};
    use crate::core::fields::byte_translate::ByteTranslate;

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

    #[test]
    pub fn test_byte_translate() {
        let mut rng = rand::thread_rng();
        for _ in 0..10000 {
            let x = M31::from_u32_unchecked(rng.gen::<u32>() % P);
            let mut dst_le = [0u8; M31::LENGTH_BYTES];
            let mut dst_be = [0u8; M31::LENGTH_BYTES];

            x.write_le_bytes(&mut dst_le);
            x.write_be_bytes(&mut dst_be);
            let x_le_bytes = x.to_le_bytes();
            let x_be_bytes = x.to_be_bytes();

            assert_eq!(x_le_bytes, dst_le);
            assert_eq!(x_be_bytes, dst_be);
            assert_eq!(x, M31::read_le_bytes(&x_le_bytes).unwrap());
            assert_eq!(x, M31::read_be_bytes(&x_be_bytes).unwrap());
        }
    }
}
