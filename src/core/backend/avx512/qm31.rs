use std::ops::{Add, Mul, Sub};

use super::cm31::PackedCM31;
use super::m31::K_BLOCK_SIZE;
use super::PackedBaseField;
use crate::core::fields::qm31::QM31;

/// AVX implementation for an extension of CM31.
/// See [crate::core::fields::qm31::QM31] for more information.
#[derive(Copy, Clone)]
pub struct PackedQM31(pub [PackedCM31; 2]);
impl PackedQM31 {
    pub fn a(&self) -> PackedCM31 {
        self.0[0]
    }
    pub fn b(&self) -> PackedCM31 {
        self.0[1]
    }
    pub fn to_array(&self) -> [QM31; K_BLOCK_SIZE] {
        std::array::from_fn(|i| QM31(self.a().to_array()[i], self.b().to_array()[i]))
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
        //   (ac + (1+2i)bd) + (ad + bc)u =
        //   ac + bd + 2ibd + (ad + bc)u.
        let ac = self.a() * rhs.a();
        let bd = self.b() * rhs.b();
        let bd2 = bd + bd;
        // Computes ac + bd.
        let ac_p_bd = ac + bd;
        // Computes ad + bc.
        let ad_p_bc = (self.a() + self.b()) * (rhs.a() + rhs.b()) - ac_p_bd;
        // ac + bd + 2ibd =
        // ac + bd -Im(2bd) + iRe(2bd)
        let l = PackedCM31([ac_p_bd.a() - bd2.b(), ac_p_bd.b() + bd2.a()]);
        Self([l, ad_p_bc])
    }
}

impl Mul<PackedBaseField> for PackedQM31 {
    type Output = Self;

    fn mul(self, rhs: PackedBaseField) -> Self {
        Self([self.0[0] * rhs, self.0[1] * rhs])
    }
}

impl Add<PackedBaseField> for PackedQM31 {
    type Output = Self;

    fn add(self, rhs: PackedBaseField) -> Self {
        Self([self.0[0] + rhs, self.0[1]])
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
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
