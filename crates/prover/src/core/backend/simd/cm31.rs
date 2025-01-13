use std::array;
use std::ops::{Add, Mul, MulAssign, Neg, Sub};

use bytemuck::{Pod, Zeroable};
use num_traits::{One, Zero};

use super::m31::{PackedM31, N_LANES};
use super::PACKED_CM31_BATCH_INVERSE_CHUNK_SIZE;
use crate::core::fields::cm31::CM31;
use crate::core::fields::{batch_inverse_chunked, FieldExpOps};

/// SIMD implementation of [`CM31`].
#[derive(Copy, Clone, Debug)]
pub struct PackedCM31(pub [PackedM31; 2]);

impl PackedCM31 {
    /// Constructs a new instance with all vector elements set to `value`.
    pub fn broadcast(value: CM31) -> Self {
        Self([PackedM31::broadcast(value.0), PackedM31::broadcast(value.1)])
    }

    /// Returns all `a` values such that each vector element is represented as `a + bi`.
    pub const fn a(&self) -> PackedM31 {
        self.0[0]
    }

    /// Returns all `b` values such that each vector element is represented as `a + bi`.
    pub const fn b(&self) -> PackedM31 {
        self.0[1]
    }

    pub fn to_array(&self) -> [CM31; N_LANES] {
        let a = self.a().to_array();
        let b = self.b().to_array();
        array::from_fn(|i| CM31(a[i], b[i]))
    }

    pub fn from_array(values: [CM31; N_LANES]) -> Self {
        Self([
            PackedM31::from_array(values.map(|v| v.0)),
            PackedM31::from_array(values.map(|v| v.1)),
        ])
    }

    /// Interleaves two vectors.
    pub fn interleave(self, other: Self) -> (Self, Self) {
        let Self([a_evens, b_evens]) = self;
        let Self([a_odds, b_odds]) = other;
        let (a_lhs, a_rhs) = a_evens.interleave(a_odds);
        let (b_lhs, b_rhs) = b_evens.interleave(b_odds);
        (Self([a_lhs, b_lhs]), Self([a_rhs, b_rhs]))
    }

    /// Deinterleaves two vectors.
    pub fn deinterleave(self, other: Self) -> (Self, Self) {
        let Self([a_self, b_self]) = self;
        let Self([a_other, b_other]) = other;
        let (a_evens, a_odds) = a_self.deinterleave(a_other);
        let (b_evens, b_odds) = b_self.deinterleave(b_other);
        (Self([a_evens, b_evens]), Self([a_odds, b_odds]))
    }

    /// Doubles each element in the vector.
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
        Self([PackedM31::zero(), PackedM31::zero()])
    }

    fn is_zero(&self) -> bool {
        self.a().is_zero() && self.b().is_zero()
    }
}

unsafe impl Pod for PackedCM31 {}

unsafe impl Zeroable for PackedCM31 {
    fn zeroed() -> Self {
        unsafe { core::mem::zeroed() }
    }
}

impl One for PackedCM31 {
    fn one() -> Self {
        Self([PackedM31::one(), PackedM31::zero()])
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
        // 1 / (a + bi) = (a - bi) / (a^2 + b^2).
        Self([self.a(), -self.b()]) * (self.a().square() + self.b().square()).inverse()
    }

    fn batch_inverse(column: &[Self]) -> Vec<Self> {
        batch_inverse_chunked(column, PACKED_CM31_BATCH_INVERSE_CHUNK_SIZE)
    }
}

impl Add<PackedM31> for PackedCM31 {
    type Output = Self;

    fn add(self, rhs: PackedM31) -> Self::Output {
        Self([self.a() + rhs, self.b()])
    }
}

impl Sub<PackedM31> for PackedCM31 {
    type Output = Self;

    fn sub(self, rhs: PackedM31) -> Self::Output {
        let Self([a, b]) = self;
        Self([a - rhs, b])
    }
}

impl Mul<PackedM31> for PackedCM31 {
    type Output = Self;

    fn mul(self, rhs: PackedM31) -> Self::Output {
        let Self([a, b]) = self;
        Self([a * rhs, b * rhs])
    }
}

impl Neg for PackedCM31 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let Self([a, b]) = self;
        Self([-a, -b])
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::backend::simd::cm31::PackedCM31;

    #[test]
    fn addition_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();
        let packed_lhs = PackedCM31::from_array(lhs);
        let packed_rhs = PackedCM31::from_array(rhs);

        let res = packed_lhs + packed_rhs;

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] + rhs[i]));
    }

    #[test]
    fn subtraction_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();
        let packed_lhs = PackedCM31::from_array(lhs);
        let packed_rhs = PackedCM31::from_array(rhs);

        let res = packed_lhs - packed_rhs;

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] - rhs[i]));
    }

    #[test]
    fn multiplication_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();
        let packed_lhs = PackedCM31::from_array(lhs);
        let packed_rhs = PackedCM31::from_array(rhs);

        let res = packed_lhs * packed_rhs;

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] * rhs[i]));
    }

    #[test]
    fn negation_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let values = rng.gen();
        let packed_values = PackedCM31::from_array(values);

        let res = -packed_values;

        assert_eq!(res.to_array(), values.map(|v| -v));
    }
}
