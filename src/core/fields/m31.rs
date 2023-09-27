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

pub const P: u32 = 2147483647; // 2 ** 31 - 1

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct M31(u32);
pub type Field = M31;

impl M31 {
    pub fn square(&self) -> Self {
        (*self) * (*self)
    }

    pub fn double(&self) -> M31 {
        (*self) + (*self)
    }

    pub fn sqrt(&self) -> Option<M31> {
        let result = self.pow((1 << 29) as u32);
        (result.square() == *self).then_some(result)
    }

    pub fn pow(&self, exp: u32) -> Self {
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
    pub fn reduce(val: u64) -> Self {
        Self((((((val >> 31) + val + 1) >> 31) + val) & (P as u64)) as u32)
    }

    pub fn one() -> M31 {
        Self(1)
    }

    pub fn zero() -> M31 {
        Self(0)
    }

    pub const fn from_u32_unchecked(arg: u32) -> M31 {
        Self(arg)
    }

    pub fn inverse(&self) -> M31 {
        assert!(*self != Self::zero(), "division by zero");
        self.pow(P - 2)
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

impl AddAssign for M31 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
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
impl SubAssign for M31 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for M31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::reduce((self.0 as u64) * (rhs.0 as u64))
    }
}

impl MulAssign for M31 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for M31 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl DivAssign for M31 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
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
    fn test_basic_operations() {
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
