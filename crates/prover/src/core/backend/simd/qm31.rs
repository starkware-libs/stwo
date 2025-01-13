use std::array;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use bytemuck::{Pod, Zeroable};
use num_traits::{One, Zero};
use rand::distributions::{Distribution, Standard};

use super::cm31::PackedCM31;
use super::m31::{PackedM31, N_LANES};
use super::PACKED_QM31_BATCH_INVERSE_CHUNK_SIZE;
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::QM31;
use crate::core::fields::{batch_inverse_chunked, FieldExpOps};

pub type PackedSecureField = PackedQM31;

/// SIMD implementation of [`QM31`].
#[derive(Copy, Clone, Debug)]
pub struct PackedQM31(pub [PackedCM31; 2]);

impl PackedQM31 {
    /// Constructs a new instance with all vector elements set to `value`.
    pub fn broadcast(value: QM31) -> Self {
        Self([
            PackedCM31::broadcast(value.0),
            PackedCM31::broadcast(value.1),
        ])
    }

    /// Returns all `a` values such that each vector element is represented as `a + bu`.
    pub const fn a(&self) -> PackedCM31 {
        self.0[0]
    }

    /// Returns all `b` values such that each vector element is represented as `a + bu`.
    pub const fn b(&self) -> PackedCM31 {
        self.0[1]
    }

    pub fn to_array(&self) -> [QM31; N_LANES] {
        let a = self.a().to_array();
        let b = self.b().to_array();
        array::from_fn(|i| QM31(a[i], b[i]))
    }

    pub fn from_array(values: [QM31; N_LANES]) -> Self {
        let a = values.map(|v| v.0);
        let b = values.map(|v| v.1);
        Self([PackedCM31::from_array(a), PackedCM31::from_array(b)])
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
        let Self([a_lhs, b_lhs]) = self;
        let Self([a_rhs, b_rhs]) = other;
        let (a_evens, a_odds) = a_lhs.deinterleave(a_rhs);
        let (b_evens, b_odds) = b_lhs.deinterleave(b_rhs);
        (Self([a_evens, b_evens]), Self([a_odds, b_odds]))
    }

    /// Sums all the elements in the vector.
    pub fn pointwise_sum(self) -> QM31 {
        self.to_array().into_iter().sum()
    }

    /// Doubles each element in the vector.
    pub fn double(self) -> Self {
        let Self([a, b]) = self;
        Self([a.double(), b.double()])
    }

    /// Returns vectors `a, b, c, d` such that element `i` is represented as
    /// `QM31(a_i, b_i, c_i, d_i)`.
    pub const fn into_packed_m31s(self) -> [PackedM31; 4] {
        let Self([PackedCM31([a, b]), PackedCM31([c, d])]) = self;
        [a, b, c, d]
    }

    /// Creates an instance from vectors `a, b, c, d` such that element `i`
    /// is represented as `QM31(a_i, b_i, c_i, d_i)`.
    pub const fn from_packed_m31s([a, b, c, d]: [PackedM31; 4]) -> Self {
        Self([PackedCM31([a, b]), PackedCM31([c, d])])
    }
}

impl Add for PackedQM31 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self([self.a() + rhs.a(), self.b() + rhs.b()])
    }
}

impl Sub for PackedQM31 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self([self.a() - rhs.a(), self.b() - rhs.b()])
    }
}

impl Mul for PackedQM31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // Compute using Karatsuba.
        //   (a + ub) * (c + ud) =
        //   (ac + (2+i)bd) + (ad + bc)u =
        //   ac + 2bd + ibd + (ad + bc)u.
        let ac = self.a() * rhs.a();
        let bd = self.b() * rhs.b();
        let bd_times_1_plus_i = PackedCM31([bd.a() - bd.b(), bd.a() + bd.b()]);
        // Computes ac + bd.
        let ac_p_bd = ac + bd;
        // Computes ad + bc.
        let ad_p_bc = (self.a() + self.b()) * (rhs.a() + rhs.b()) - ac_p_bd;
        // ac + 2bd + ibd =
        // ac + bd + bd + ibd
        let l = PackedCM31([
            ac_p_bd.a() + bd_times_1_plus_i.a(),
            ac_p_bd.b() + bd_times_1_plus_i.b(),
        ]);
        Self([l, ad_p_bc])
    }
}

impl Zero for PackedQM31 {
    fn zero() -> Self {
        Self([PackedCM31::zero(), PackedCM31::zero()])
    }

    fn is_zero(&self) -> bool {
        self.a().is_zero() && self.b().is_zero()
    }
}

impl One for PackedQM31 {
    fn one() -> Self {
        Self([PackedCM31::one(), PackedCM31::zero()])
    }
}

impl AddAssign for PackedQM31 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl MulAssign for PackedQM31 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl FieldExpOps for PackedQM31 {
    fn inverse(&self) -> Self {
        assert!(!self.is_zero(), "0 has no inverse");
        // (a + bu)^-1 = (a - bu) / (a^2 - (2+i)b^2).
        let b2 = self.b().square();
        let ib2 = PackedCM31([-b2.b(), b2.a()]);
        let denom = self.a().square() - (b2 + b2 + ib2);
        let denom_inverse = denom.inverse();
        Self([self.a() * denom_inverse, -self.b() * denom_inverse])
    }

    fn batch_inverse(column: &[Self]) -> Vec<Self> {
        batch_inverse_chunked(column, PACKED_QM31_BATCH_INVERSE_CHUNK_SIZE)
    }
}

impl Add<PackedM31> for PackedQM31 {
    type Output = Self;

    fn add(self, rhs: PackedM31) -> Self::Output {
        Self([self.a() + rhs, self.b()])
    }
}

impl Mul<PackedM31> for PackedQM31 {
    type Output = Self;

    fn mul(self, rhs: PackedM31) -> Self::Output {
        let Self([a, b]) = self;
        Self([a * rhs, b * rhs])
    }
}

impl Mul<PackedCM31> for PackedQM31 {
    type Output = Self;

    fn mul(self, rhs: PackedCM31) -> Self::Output {
        let Self([a, b]) = self;
        Self([a * rhs, b * rhs])
    }
}

impl Sub<PackedM31> for PackedQM31 {
    type Output = Self;

    fn sub(self, rhs: PackedM31) -> Self::Output {
        let Self([a, b]) = self;
        Self([a - rhs, b])
    }
}

impl Add<QM31> for PackedQM31 {
    type Output = Self;

    fn add(self, rhs: QM31) -> Self::Output {
        self + PackedQM31::broadcast(rhs)
    }
}

impl Sub<QM31> for PackedQM31 {
    type Output = Self;

    fn sub(self, rhs: QM31) -> Self::Output {
        self - PackedQM31::broadcast(rhs)
    }
}

impl Mul<QM31> for PackedQM31 {
    type Output = Self;

    fn mul(self, rhs: QM31) -> Self::Output {
        self * PackedQM31::broadcast(rhs)
    }
}

impl Mul<M31> for PackedQM31 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: M31) -> Self::Output {
        self * PackedM31::broadcast(rhs)
    }
}

impl Add<M31> for PackedQM31 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: M31) -> Self::Output {
        self + PackedM31::broadcast(rhs)
    }
}

impl SubAssign for PackedQM31 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

unsafe impl Pod for PackedQM31 {}

unsafe impl Zeroable for PackedQM31 {
    fn zeroed() -> Self {
        unsafe { core::mem::zeroed() }
    }
}

impl Sum for PackedQM31 {
    fn sum<I>(mut iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        let first = iter.next().unwrap_or_else(Self::zero);
        iter.fold(first, |a, b| a + b)
    }
}

impl<'a> Sum<&'a Self> for PackedQM31 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.copied().sum()
    }
}

impl Neg for PackedQM31 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let Self([a, b]) = self;
        Self([-a, -b])
    }
}

impl Distribution<PackedQM31> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> PackedQM31 {
        PackedQM31::from_array(rng.gen())
    }
}

impl From<PackedM31> for PackedQM31 {
    fn from(value: PackedM31) -> Self {
        PackedQM31::from_packed_m31s([
            value,
            PackedM31::zero(),
            PackedM31::zero(),
            PackedM31::zero(),
        ])
    }
}

impl From<QM31> for PackedQM31 {
    fn from(value: QM31) -> Self {
        PackedQM31::broadcast(value)
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::backend::simd::qm31::PackedQM31;

    #[test]
    fn addition_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();
        let packed_lhs = PackedQM31::from_array(lhs);
        let packed_rhs = PackedQM31::from_array(rhs);

        let res = packed_lhs + packed_rhs;

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] + rhs[i]));
    }

    #[test]
    fn subtraction_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();
        let packed_lhs = PackedQM31::from_array(lhs);
        let packed_rhs = PackedQM31::from_array(rhs);

        let res = packed_lhs - packed_rhs;

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] - rhs[i]));
    }

    #[test]
    fn multiplication_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();
        let packed_lhs = PackedQM31::from_array(lhs);
        let packed_rhs = PackedQM31::from_array(rhs);

        let res = packed_lhs * packed_rhs;

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] * rhs[i]));
    }

    #[test]
    fn negation_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let values = rng.gen();
        let packed_values = PackedQM31::from_array(values);

        let res = -packed_values;

        assert_eq!(res.to_array(), values.map(|v| -v));
    }
}
