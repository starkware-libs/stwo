use std::ops::{Add, AddAssign, Mul, MulAssign, Sub};

use bytemuck::{Pod, Zeroable};
use num_traits::{One, Zero};

use super::cm31::PackedCM31;
use super::m31::K_BLOCK_SIZE;
use super::PackedBaseField;
use crate::core::fields::qm31::QM31;
use crate::core::fields::FieldExpOps;

/// AVX implementation for an extension of CM31.
/// See [crate::core::fields::qm31::QM31] for more information.
#[derive(Copy, Clone, Debug)]
pub struct PackedSecureField(pub [PackedCM31; 2]);
impl PackedSecureField {
    pub fn zero() -> Self {
        Self([
            PackedCM31([PackedBaseField::zero(); 2]),
            PackedCM31([PackedBaseField::zero(); 2]),
        ])
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
    pub fn to_array(&self) -> [QM31; K_BLOCK_SIZE] {
        std::array::from_fn(|i| QM31(self.a().to_array()[i], self.b().to_array()[i]))
    }

    pub fn from_array(array: [QM31; K_BLOCK_SIZE]) -> Self {
        let a = PackedBaseField::from_array(std::array::from_fn(|i| array[i].0 .0));
        let b = PackedBaseField::from_array(std::array::from_fn(|i| array[i].0 .1));
        let c = PackedBaseField::from_array(std::array::from_fn(|i| array[i].1 .0));
        let d = PackedBaseField::from_array(std::array::from_fn(|i| array[i].1 .1));
        Self([PackedCM31([a, b]), PackedCM31([c, d])])
    }

    // Multiply packed QM31 by packed M31.
    pub fn mul_packed_m31(&self, rhs: PackedBaseField) -> PackedSecureField {
        Self::from_packed_m31s(self.to_packed_m31s().map(|m31| m31 * rhs))
    }

    /// Sums all the elements in the packed M31 element.
    pub fn pointwise_sum(self) -> QM31 {
        self.to_array().into_iter().sum()
    }

    pub fn to_packed_m31s(&self) -> [PackedBaseField; 4] {
        [self.a().a(), self.a().b(), self.b().a(), self.b().b()]
    }

    pub fn from_packed_m31s(array: [PackedBaseField; 4]) -> Self {
        Self([
            PackedCM31([array[0], array[1]]),
            PackedCM31([array[2], array[3]]),
        ])
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
        assert!(!self.is_zero(), "0 has no inverse");
        // (a + bu)^-1 = (a - bu) / (a^2 - (2+i)b^2).
        let b2 = self.b().square();
        let ib2 = PackedCM31([-b2.b(), b2.a()]);
        let denom = self.a().square() - (b2 + b2 + ib2);
        let denom_inverse = denom.inverse();
        Self([self.a() * denom_inverse, -self.b() * denom_inverse])
    }
}

impl Add<PackedBaseField> for PackedSecureField {
    type Output = Self;
    fn add(self, rhs: PackedBaseField) -> Self::Output {
        Self([self.a() + rhs, self.b()])
    }
}
impl Sub<PackedBaseField> for PackedSecureField {
    type Output = Self;
    fn sub(self, rhs: PackedBaseField) -> Self::Output {
        Self([self.a() - rhs, self.b()])
    }
}
impl Mul<PackedBaseField> for PackedSecureField {
    type Output = Self;
    fn mul(self, rhs: PackedBaseField) -> Self::Output {
        Self([self.a() * rhs, self.b() * rhs])
    }
}

unsafe impl Pod for PackedSecureField {}
unsafe impl Zeroable for PackedSecureField {
    fn zeroed() -> Self {
        unsafe { core::mem::zeroed() }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    #[test]
    fn test_qm31avx512_basic_ops() {
        let mut rng = SmallRng::seed_from_u64(0);
        let x = PackedSecureField::from_array(rng.gen());
        let y = PackedSecureField::from_array(rng.gen());
        let sum = x + y;
        let diff = x - y;
        let prod = x * y;
        for i in 0..16 {
            assert_eq!(sum.to_array()[i], x.to_array()[i] + y.to_array()[i]);
            assert_eq!(diff.to_array()[i], x.to_array()[i] - y.to_array()[i]);
            assert_eq!(prod.to_array()[i], x.to_array()[i] * y.to_array()[i]);
        }
    }

    #[test]
    fn test_from_array() {
        let mut rng = SmallRng::seed_from_u64(0);
        let x_arr = std::array::from_fn(|_| rng.gen());

        let packed = PackedSecureField::from_array(x_arr);
        let to_arr = packed.to_array();

        assert_eq!(to_arr, x_arr);
    }
}
