use std::ops::{Add, Mul, Sub};

use super::cm31::PackedCM31;
use super::m31::K_BLOCK_SIZE;
use crate::core::fields::qm31::QM31;

#[derive(Copy, Clone)]
pub struct PackedQM31(pub [PackedCM31; 2]);
impl PackedQM31 {
    pub fn to_array(&self) -> [QM31; K_BLOCK_SIZE] {
        std::array::from_fn(|i| QM31(self.0[0].to_array()[i], self.0[1].to_array()[i]))
    }
}
impl Add for PackedQM31 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}
impl Sub for PackedQM31 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}
impl Mul for PackedQM31 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        // (a+ub)*(c+ud) = (ac+(1+2i)bd)+(ad+bc)u =
        // ac+bd+2ibd+(ad+bc)u.
        // Use karatsuba to compute multiplication.
        let ac = self.0[0] * rhs.0[0];
        let bd = self.0[1] * rhs.0[1];
        let bd2 = bd + bd;
        let ac_p_bd = ac + bd;
        let m = (self.0[0] + self.0[1]) * (rhs.0[0] + rhs.0[1]) - ac_p_bd;
        // ac+bd+2ibd =
        // ac+bd-Im(2bd)+iRe(2bd)
        let l = PackedCM31([ac_p_bd.0[0] - bd2.0[1], ac_p_bd.0[1] + bd2.0[0]]);
        Self([l, m])
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::core::backend::avx512::m31::PackedBaseField;
    use crate::core::fields::m31::{M31, P};

    #[test]
    fn test_qm31avx512_basic_ops() {
        let rng = &mut StdRng::seed_from_u64(0);
        let x = PackedQM31([
            PackedCM31([
                PackedBaseField::from_array(std::array::from_fn(|_| {
                    M31::from(rng.gen::<u32>() % P)
                })),
                PackedBaseField::from_array(std::array::from_fn(|_| {
                    M31::from(rng.gen::<u32>() % P)
                })),
            ]),
            PackedCM31([
                PackedBaseField::from_array(std::array::from_fn(|_| {
                    M31::from(rng.gen::<u32>() % P)
                })),
                PackedBaseField::from_array(std::array::from_fn(|_| {
                    M31::from(rng.gen::<u32>() % P)
                })),
            ]),
        ]);
        let y = PackedQM31([
            PackedCM31([
                PackedBaseField::from_array(std::array::from_fn(|_| {
                    M31::from(rng.gen::<u32>() % P)
                })),
                PackedBaseField::from_array(std::array::from_fn(|_| {
                    M31::from(rng.gen::<u32>() % P)
                })),
            ]),
            PackedCM31([
                PackedBaseField::from_array(std::array::from_fn(|_| {
                    M31::from(rng.gen::<u32>() % P)
                })),
                PackedBaseField::from_array(std::array::from_fn(|_| {
                    M31::from(rng.gen::<u32>() % P)
                })),
            ]),
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
