use std::ops::{Add, Mul, MulAssign, Sub};

use num_traits::{One, Zero};

use super::m31::{PackedBaseField, K_BLOCK_SIZE};
use crate::core::fields::cm31::{CM31, P2};
use crate::core::fields::FieldExpOps;

/// AVX implementation for the complex extension field of M31.
/// See [crate::core::fields::cm31::CM31] for more information.
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
    pub fn to_array(&self) -> [CM31; K_BLOCK_SIZE] {
        std::array::from_fn(|i| CM31(self.0[0].to_array()[i], self.0[1].to_array()[i]))
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
        Self([self.a() - rhs, self.b()])
    }
}
impl Mul<PackedBaseField> for PackedCM31 {
    type Output = Self;
    fn mul(self, rhs: PackedBaseField) -> Self::Output {
        Self([self.a() * rhs, self.b() * rhs])
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::core::fields::m31::{M31, P};

    #[test]
    fn test_cm31avx512_basic_ops() {
        let rng = &mut StdRng::seed_from_u64(0);
        let x = PackedCM31([
            PackedBaseField::from_array(std::array::from_fn(|_| M31::from(rng.gen::<u32>() % P))),
            PackedBaseField::from_array(std::array::from_fn(|_| M31::from(rng.gen::<u32>() % P))),
        ]);
        let y = PackedCM31([
            PackedBaseField::from_array(std::array::from_fn(|_| M31::from(rng.gen::<u32>() % P))),
            PackedBaseField::from_array(std::array::from_fn(|_| M31::from(rng.gen::<u32>() % P))),
        ]);
        let sum = x + y;
        let diff = x - y;
        let prod = x * y;
        for i in 0..16 {
            assert_eq!(sum.to_array()[i], x.to_array()[i] + y.to_array()[i]);
            assert_eq!(diff.to_array()[i], x.to_array()[i] - y.to_array()[i]);
            assert_eq!(prod.to_array()[i], x.to_array()[i] * y.to_array()[i]);
        }
    }
}
