use std::array;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use bytemuck::{Pod, Zeroable};
use num_traits::{One, Zero};

use super::cm31::PackedCM31;
use super::m31::{PackedBaseField, N_LANES};
use crate::core::fields::qm31::{P4, QM31};
use crate::core::fields::FieldExpOps;

/// SIMD implementation of [`QM31`].
#[derive(Copy, Clone, Debug)]
pub struct PackedSecureField(pub [PackedCM31; 2]);

impl PackedSecureField {
    pub fn zero() -> Self {
        Self([PackedCM31::zero(), PackedCM31::zero()])
    }

    pub fn broadcast(value: QM31) -> Self {
        Self([
            PackedCM31::broadcast(value.0),
            PackedCM31::broadcast(value.1),
        ])
    }

    pub fn a(&self) -> PackedCM31 {
        self.0[0]
    }

    pub fn b(&self) -> PackedCM31 {
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

    pub fn interleave(self, other: Self) -> (Self, Self) {
        let Self([a_evens, b_evens]) = self;
        let Self([a_odds, b_odds]) = other;
        let (a_lhs, a_rhs) = a_evens.interleave(a_odds);
        let (b_lhs, b_rhs) = b_evens.interleave(b_odds);
        (Self([a_lhs, b_lhs]), Self([a_rhs, b_rhs]))
    }

    pub fn deinterleave(self, other: Self) -> (Self, Self) {
        let Self([a_lhs, b_lhs]) = self;
        let Self([a_rhs, b_rhs]) = other;
        let (a_evens, a_odds) = a_lhs.deinterleave(a_rhs);
        let (b_evens, b_odds) = b_lhs.deinterleave(b_rhs);
        (Self([a_evens, b_evens]), Self([a_odds, b_odds]))
    }

    /// Sums all the elements in the packed M31 element.
    pub fn pointwise_sum(self) -> QM31 {
        self.to_array().into_iter().sum()
    }

    pub fn double(self) -> Self {
        let Self([a, b]) = self;
        Self([a.double(), b.double()])
    }
}

impl Add for PackedSecureField {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self([self.a() + rhs.a(), self.b() + rhs.b()])
    }
}

impl Sub for PackedSecureField {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self([self.a() - rhs.a(), self.b() - rhs.b()])
    }
}

impl Mul for PackedSecureField {
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

impl Zero for PackedSecureField {
    fn zero() -> Self {
        Self([PackedCM31::zero(), PackedCM31::zero()])
    }

    fn is_zero(&self) -> bool {
        self.a().is_zero() && self.b().is_zero()
    }
}

impl One for PackedSecureField {
    fn one() -> Self {
        Self([PackedCM31::one(), PackedCM31::zero()])
    }
}

impl AddAssign for PackedSecureField {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl MulAssign for PackedSecureField {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl FieldExpOps for PackedSecureField {
    fn inverse(&self) -> Self {
        // TODO(andrew): Use a better multiplication tree. Also for other constant powers in the
        // code.
        assert!(!self.is_zero(), "0 has no inverse");
        self.pow(P4 - 2)
    }
}

impl Add<PackedBaseField> for PackedSecureField {
    type Output = Self;

    fn add(self, rhs: PackedBaseField) -> Self::Output {
        Self([self.a() + rhs, self.b()])
    }
}

impl Mul<PackedBaseField> for PackedSecureField {
    type Output = Self;

    fn mul(self, rhs: PackedBaseField) -> Self::Output {
        let Self([a, b]) = self;
        Self([a * rhs, b * rhs])
    }
}

impl Sub<PackedBaseField> for PackedSecureField {
    type Output = Self;

    fn sub(self, rhs: PackedBaseField) -> Self::Output {
        let Self([a, b]) = self;
        Self([a - rhs, b])
    }
}

impl SubAssign for PackedSecureField {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

unsafe impl Pod for PackedSecureField {}

unsafe impl Zeroable for PackedSecureField {
    fn zeroed() -> Self {
        unsafe { core::mem::zeroed() }
    }
}

impl Sum for PackedSecureField {
    fn sum<I>(mut iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        let first = iter.next().unwrap_or_else(Self::zero);
        iter.fold(first, |a, b| a + b)
    }
}

impl<'a> Sum<&'a Self> for PackedSecureField {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.copied().sum()
    }
}

impl Neg for PackedSecureField {
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

    use crate::core::backend::simd::qm31::PackedSecureField;

    #[test]
    fn addition_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();

        let res = PackedSecureField::from_array(lhs) + PackedSecureField::from_array(rhs);

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] + rhs[i]));
    }

    #[test]
    fn subtraction_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();

        let res = PackedSecureField::from_array(lhs) - PackedSecureField::from_array(rhs);

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] - rhs[i]));
    }

    #[test]
    fn multiplication_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();

        let res = PackedSecureField::from_array(lhs) * PackedSecureField::from_array(rhs);

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] * rhs[i]));
    }

    #[test]
    fn negation_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let values = rng.gen();

        let res = -PackedSecureField::from_array(values);

        assert_eq!(res.to_array(), values.map(|v| -v));
    }
}
