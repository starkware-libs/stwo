use std::ops::{Add, AddAssign, Mul, MulAssign, Sub};

use bytemuck::{Pod, Zeroable};
use num_traits::{One, Zero};

use super::cm31::PackedCM31;
use super::m31::K_BLOCK_SIZE;
use super::PackedBaseField;
use crate::core::fields::qm31::{P4, QM31};
use crate::core::fields::FieldExpOps;

/// AVX implementation for an extension of CM31.
/// See [crate::core::fields::qm31::QM31] for more information.
#[derive(Copy, Clone)]
pub struct PackedQM31(pub [PackedCM31; 2]);
impl PackedQM31 {
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

    pub fn from_array(array: &[QM31; K_BLOCK_SIZE]) -> Self {
        let a = PackedBaseField::from_array(std::array::from_fn(|i| array[i].0 .0));
        let b = PackedBaseField::from_array(std::array::from_fn(|i| array[i].0 .1));
        let c = PackedBaseField::from_array(std::array::from_fn(|i| array[i].1 .0));
        let d = PackedBaseField::from_array(std::array::from_fn(|i| array[i].1 .1));
        Self([PackedCM31([a, b]), PackedCM31([c, d])])
    }

    // Multiply packed QM31 by packed M31.
    pub fn mul_packed_m31(&self, rhs: PackedBaseField) -> PackedQM31 {
        let a = self.0[0].0[0] * rhs;
        let b = self.0[0].0[1] * rhs;
        let c = self.0[1].0[0] * rhs;
        let d = self.0[1].0[1] * rhs;
        PackedQM31([PackedCM31([a, b]), PackedCM31([c, d])])
    }

    /// Sums all the elements in the packed M31 element.
    pub fn pointwise_sum(self) -> QM31 {
        self.to_array().into_iter().sum()
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
        // TODO(andrew): Use a better multiplication tree. Also for other constant powers in the
        // code.
        assert!(!self.is_zero(), "0 has no inverse");
        self.pow(P4 - 2)
    }
}

impl Add<PackedBaseField> for PackedQM31 {
    type Output = Self;
    fn add(self, rhs: PackedBaseField) -> Self::Output {
        Self([self.a() + rhs, self.b()])
    }
}
impl Sub<PackedBaseField> for PackedQM31 {
    type Output = Self;
    fn sub(self, rhs: PackedBaseField) -> Self::Output {
        Self([self.a() - rhs, self.b()])
    }
}
impl Mul<PackedBaseField> for PackedQM31 {
    type Output = Self;
    fn mul(self, rhs: PackedBaseField) -> Self::Output {
        Self([self.a() * rhs, self.b() * rhs])
    }
}

unsafe impl Pod for PackedQM31 {}
unsafe impl Zeroable for PackedQM31 {
    fn zeroed() -> Self {
        unsafe { core::mem::zeroed() }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::core::backend::avx512::m31::PackedBaseField;
    use crate::core::fields::cm31::CM31;
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

    #[test]
    fn test_from_array() {
        let rng = &mut StdRng::seed_from_u64(0);
        let x_arr = std::array::from_fn(|_| {
            QM31(
                CM31(
                    M31::from(rng.gen::<u32>() % P),
                    M31::from(rng.gen::<u32>() % P),
                ),
                CM31(
                    M31::from(rng.gen::<u32>() % P),
                    M31::from(rng.gen::<u32>() % P),
                ),
            )
        });

        let packed = PackedQM31::from_array(&x_arr);
        let to_arr = packed.to_array();

        assert_eq!(to_arr, x_arr);
    }
}
