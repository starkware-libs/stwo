use std::ops::{Add, Mul, Sub};

use super::m31::{PackedBaseField, K_BLOCK_SIZE};
use crate::core::fields::cm31::CM31;

#[derive(Copy, Clone)]
pub struct PackedCM31(pub [PackedBaseField; 2]);
impl PackedCM31 {
    pub fn to_array(&self) -> [CM31; K_BLOCK_SIZE] {
        std::array::from_fn(|i| CM31(self.0[0].to_array()[i], self.0[1].to_array()[i]))
    }
}
impl Add for PackedCM31 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}
impl Sub for PackedCM31 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}
impl Mul for PackedCM31 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let ac = self.0[0] * rhs.0[0];
        let bd = self.0[1] * rhs.0[1];
        let m = (self.0[0] + self.0[1]) * (rhs.0[0] + rhs.0[1]);
        Self([ac - bd, m - ac - bd])
    }
}

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
