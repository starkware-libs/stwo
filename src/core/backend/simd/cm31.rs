use std::array;
use std::ops::{Add, Mul, MulAssign, Sub};

use num_traits::{One, Zero};

use super::m31::{PackedBaseField, N_LANES};
use crate::core::fields::cm31::{CM31, P2};
use crate::core::fields::FieldExpOps;

/// SIMD implementation of [`CM31`].
#[derive(Copy, Clone, Debug)]
pub struct PackedCM31(pub [PackedBaseField; 2]);

impl PackedCM31 {
    pub fn broadcast(value: CM31) -> Self {
        Self([
            PackedBaseField::broadcast(value.0),
            PackedBaseField::broadcast(value.1),
        ])
    }

    pub fn a(&self) -> PackedBaseField {
        self.0[0]
    }

    pub fn b(&self) -> PackedBaseField {
        self.0[1]
    }

    pub fn to_array(&self) -> [CM31; N_LANES] {
        let a = self.a().to_array();
        let b = self.b().to_array();
        array::from_fn(|i| CM31(a[i], b[i]))
    }

    pub fn from_array(values: [CM31; N_LANES]) -> Self {
        Self([
            PackedBaseField::from_array(values.map(|v| v.0)),
            PackedBaseField::from_array(values.map(|v| v.1)),
        ])
    }

    pub fn interleave(self, other: Self) -> (Self, Self) {
        let Self([a_evens, b_evens]) = self;
        let Self([a_odds, b_odds]) = other;
        let (a_lhs, a_rhs) = a_evens.interleave(a_odds);
        let (b_lhs, b_rhs) = b_evens.interleave(b_odds);
        (Self([a_lhs, b_lhs]), Self([a_rhs, b_rhs]))
    }

    pub fn deinterleave(self, other: Self) -> (Self, Self) {
        let Self([a_self, b_self]) = self;
        let Self([a_other, b_other]) = other;
        let (a_evens, a_odds) = a_self.deinterleave(a_other);
        let (b_evens, b_odds) = b_self.deinterleave(b_other);
        (Self([a_evens, b_evens]), Self([a_odds, b_odds]))
    }

    pub fn double(self) -> Self {
        let Self([a, b]) = self;
        Self([a.double(), b.double()])
    }
}

impl Add for PackedCM31 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self([self.a() + rhs.a(), self.b() + rhs.b()])
    }
}

impl Sub for PackedCM31 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self([self.a() - rhs.a(), self.b() - rhs.b()])
    }
}

impl Mul for PackedCM31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // Compute using Karatsuba.
        let ac = self.a() * rhs.a();
        let bd = self.b() * rhs.b();
        // Computes (a + b) * (c + d).
        let ab_t_cd = (self.a() + self.b()) * (rhs.a() + rhs.b());
        // (ac - bd) + (ad + bc)i.
        Self([ac - bd, ab_t_cd - ac - bd])
    }
}

impl Zero for PackedCM31 {
    fn zero() -> Self {
        Self([PackedBaseField::zero(), PackedBaseField::zero()])
    }

    fn is_zero(&self) -> bool {
        self.a().is_zero() && self.b().is_zero()
    }
}

impl One for PackedCM31 {
    fn one() -> Self {
        Self([PackedBaseField::one(), PackedBaseField::zero()])
    }
}

impl MulAssign for PackedCM31 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl FieldExpOps for PackedCM31 {
    fn inverse(&self) -> Self {
        assert!(!self.is_zero(), "0 has no inverse");
        self.pow((P2 - 2) as u128)
    }
}

impl Add<PackedBaseField> for PackedCM31 {
    type Output = Self;

    fn add(self, rhs: PackedBaseField) -> Self::Output {
        Self([self.a() + rhs, self.b()])
    }
}

impl Sub<PackedBaseField> for PackedCM31 {
    type Output = Self;

    fn sub(self, rhs: PackedBaseField) -> Self::Output {
        let Self([a, b]) = self;
        Self([a - rhs, b])
    }
}

impl Mul<PackedBaseField> for PackedCM31 {
    type Output = Self;

    fn mul(self, rhs: PackedBaseField) -> Self::Output {
        let Self([a, b]) = self;
        Self([a * rhs, b * rhs])
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::core::backend::simd::cm31::PackedCM31;

    #[test]
    fn addition_works() {
        let mut rng = StdRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();

        let res = PackedCM31::from_array(lhs) + PackedCM31::from_array(rhs);

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] + rhs[i]));
    }

    #[test]
    fn subtraction_works() {
        let mut rng = StdRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();

        let res = PackedCM31::from_array(lhs) - PackedCM31::from_array(rhs);

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] - rhs[i]));
    }

    #[test]
    fn multiplication_works() {
        let mut rng = StdRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();

        let res = PackedCM31::from_array(lhs) * PackedCM31::from_array(rhs);

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] * rhs[i]));
    }

    #[test]
    fn negation_works() {
        let mut rng = StdRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();

        let res = PackedCM31::from_array(lhs) * PackedCM31::from_array(rhs);

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] * rhs[i]));
    }
}
