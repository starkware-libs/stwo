use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};

use num_traits::{One, Zero};

use super::m31::{PackedM31, N_LANES};
use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::{pow2147483645, M31};
use crate::core::fields::qm31::QM31;
use crate::core::fields::FieldExpOps;

pub const LOG_N_VERY_PACKED_ELEMS: u32 = 1;
pub const N_VERY_PACKED_ELEMS: usize = 1 << LOG_N_VERY_PACKED_ELEMS;
pub type VeryPackedBaseField = VeryPackedM31;

/// Holds a vector of PackedM31 that is fetched together.
// TODO: Remove `pub` visibility
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct VeryPackedM31(pub [PackedM31; N_VERY_PACKED_ELEMS]);
#[derive(Copy, Clone, Debug)]
pub struct VeryPackedCM31(pub [VeryPackedM31; 2]);
#[derive(Copy, Clone, Debug)]
pub struct VeryPackedQM31(pub [VeryPackedCM31; 2]);
pub type VeryPackedSecureField = VeryPackedQM31;

impl VeryPackedM31 {
    /// Constructs a new instance with all vector elements set to `value`.
    pub fn broadcast(M31(value): M31) -> Self {
        Self(std::array::from_fn(|_| PackedM31::broadcast(M31(value))))
    }

    pub fn from_array(values: [M31; N_LANES * N_VERY_PACKED_ELEMS]) -> VeryPackedM31 {
        Self(std::array::from_fn(|i| {
            let start = i * N_LANES;
            let end = start + N_LANES;
            PackedM31::from_array(values[start..end].try_into().unwrap())
        }))
    }

    pub fn to_array(&self) -> [M31; N_LANES * N_VERY_PACKED_ELEMS] {
        // Safety: We are transmuting &[A; N_VERY_PACKED_ELEMS] into &[i32; N_LANES *
        // N_VERY_PACKED_ELEMS] because we know that A contains [i32; N_LANES] and the
        // memory layout is contiguous.
        unsafe {
            std::slice::from_raw_parts(self.0.as_ptr() as *const M31, N_LANES * N_VERY_PACKED_ELEMS)
                .try_into()
                .unwrap()
        }
    }
}

impl VeryPackedCM31 {
    pub fn broadcast(value: CM31) -> Self {
        Self([
            VeryPackedM31::broadcast(value.0),
            VeryPackedM31::broadcast(value.1),
        ])
    }

    pub fn a(&self) -> VeryPackedM31 {
        self.0[0]
    }

    pub fn b(&self) -> VeryPackedM31 {
        self.0[1]
    }
}

impl VeryPackedQM31 {
    pub fn broadcast(value: QM31) -> Self {
        Self([
            VeryPackedCM31::broadcast(value.0),
            VeryPackedCM31::broadcast(value.1),
        ])
    }
    pub fn a(&self) -> VeryPackedCM31 {
        self.0[0]
    }

    pub fn b(&self) -> VeryPackedCM31 {
        self.0[1]
    }

    pub fn from_very_packed_m31s([a, b, c, d]: [VeryPackedM31; 4]) -> Self {
        Self([VeryPackedCM31([a, b]), VeryPackedCM31([c, d])])
    }
}

impl From<M31> for VeryPackedM31 {
    fn from(v: M31) -> Self {
        Self::broadcast(v)
    }
}

impl From<VeryPackedM31> for VeryPackedQM31 {
    fn from(value: VeryPackedM31) -> Self {
        VeryPackedQM31::from_very_packed_m31s([
            value,
            VeryPackedM31::zero(),
            VeryPackedM31::zero(),
            VeryPackedM31::zero(),
        ])
    }
}

impl From<QM31> for VeryPackedQM31 {
    fn from(value: QM31) -> Self {
        VeryPackedQM31::broadcast(value)
    }
}

impl Add for VeryPackedM31 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}

impl Add for VeryPackedCM31 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self([self.a() + rhs.a(), self.b() + rhs.b()])
    }
}

impl Add<M31> for VeryPackedM31 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: M31) -> Self::Output {
        Self([self.0[0] + rhs, self.0[1] + rhs])
    }
}

impl Add<QM31> for VeryPackedM31 {
    type Output = VeryPackedQM31;

    #[inline(always)]
    fn add(self, rhs: QM31) -> Self::Output {
        VeryPackedQM31::from(self) + VeryPackedQM31::broadcast(rhs)
    }
}

impl Add for VeryPackedQM31 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self([self.a() + rhs.a(), self.b() + rhs.b()])
    }
}

impl Add<QM31> for VeryPackedQM31 {
    type Output = Self;

    fn add(self, rhs: QM31) -> Self::Output {
        self + Self::broadcast(rhs)
    }
}

impl Add<VeryPackedM31> for VeryPackedCM31 {
    type Output = Self;

    fn add(self, rhs: VeryPackedM31) -> Self::Output {
        Self([self.a() + rhs, self.b()])
    }
}

impl Add<VeryPackedM31> for VeryPackedQM31 {
    type Output = Self;

    fn add(self, rhs: VeryPackedM31) -> Self::Output {
        Self([self.a() + rhs, self.b()])
    }
}

impl Sub<QM31> for VeryPackedQM31 {
    type Output = Self;

    fn sub(self, rhs: QM31) -> Self::Output {
        self - Self::broadcast(rhs)
    }
}

impl Mul<QM31> for VeryPackedM31 {
    type Output = VeryPackedQM31;

    #[inline(always)]
    fn mul(self, rhs: QM31) -> Self::Output {
        VeryPackedQM31::from(self) * VeryPackedQM31::broadcast(rhs)
    }
}

impl Neg for VeryPackedM31 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self([self.0[0].neg(), self.0[1].neg()])
    }
}

impl Neg for VeryPackedCM31 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let Self([a, b]) = self;
        Self([-a, -b])
    }
}

impl Neg for VeryPackedQM31 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let Self([a, b]) = self;
        Self([-a, -b])
    }
}

impl FieldExpOps for VeryPackedM31 {
    fn inverse(&self) -> Self {
        assert!(!self.is_zero(), "0 has no inverse");
        pow2147483645(*self)
    }
}

impl<T> AddAssign<T> for VeryPackedM31
where
    Self: Add<T, Output = Self>,
{
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs;
    }
}

impl<T> MulAssign<T> for VeryPackedM31
where
    Self: Mul<T, Output = Self>,
{
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<T> AddAssign<T> for VeryPackedQM31
where
    Self: Add<T, Output = Self>,
{
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs;
    }
}

impl Sub for VeryPackedM31 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}

impl Sub for VeryPackedCM31 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self([self.a() - rhs.a(), self.b() - rhs.b()])
    }
}

impl Sub for VeryPackedQM31 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self([self.a() - rhs.a(), self.b() - rhs.b()])
    }
}

impl Zero for VeryPackedM31 {
    fn zero() -> Self {
        Self([PackedM31::zero(), PackedM31::zero()])
    }

    fn is_zero(&self) -> bool {
        self.0[0].is_zero() && self.0[1].is_zero()
    }
}

impl Zero for VeryPackedCM31 {
    fn zero() -> Self {
        Self([VeryPackedM31::zero(), VeryPackedM31::zero()])
    }

    fn is_zero(&self) -> bool {
        self.a().is_zero() && self.b().is_zero()
    }
}

impl Zero for VeryPackedQM31 {
    fn zero() -> Self {
        Self([VeryPackedCM31::zero(), VeryPackedCM31::zero()])
    }

    fn is_zero(&self) -> bool {
        self.a().is_zero() && self.b().is_zero()
    }
}

impl One for VeryPackedM31 {
    fn one() -> Self {
        Self([PackedM31::one(), PackedM31::one()])
    }
}

impl One for VeryPackedCM31 {
    fn one() -> Self {
        Self([VeryPackedM31::one(), VeryPackedM31::zero()])
    }
}

impl One for VeryPackedQM31 {
    fn one() -> Self {
        Self([VeryPackedCM31::one(), VeryPackedCM31::zero()])
    }
}

impl Mul for VeryPackedM31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self([self.0[0] * rhs.0[0], self.0[1] * rhs.0[1]])
    }
}

impl Mul<M31> for VeryPackedM31 {
    type Output = Self;

    fn mul(self, rhs: M31) -> Self::Output {
        Self([self.0[0] * rhs, self.0[1] * rhs])
    }
}

impl Mul for VeryPackedCM31 {
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

impl Mul for VeryPackedQM31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // Compute using Karatsuba.
        //   (a + ub) * (c + ud) =
        //   (ac + (2+i)bd) + (ad + bc)u =
        //   ac + 2bd + ibd + (ad + bc)u.
        let ac = self.a() * rhs.a();
        let bd = self.b() * rhs.b();
        let bd_times_1_plus_i = VeryPackedCM31([bd.a() - bd.b(), bd.a() + bd.b()]);
        // Computes ac + bd.
        let ac_p_bd = ac + bd;
        // Computes ad + bc.
        let ad_p_bc = (self.a() + self.b()) * (rhs.a() + rhs.b()) - ac_p_bd;
        // ac + 2bd + ibd =
        // ac + bd + bd + ibd
        let l = VeryPackedCM31([
            ac_p_bd.a() + bd_times_1_plus_i.a(),
            ac_p_bd.b() + bd_times_1_plus_i.b(),
        ]);
        Self([l, ad_p_bc])
    }
}

impl Mul<VeryPackedM31> for VeryPackedCM31 {
    type Output = Self;

    fn mul(self, rhs: VeryPackedM31) -> Self::Output {
        let Self([a, b]) = self;
        Self([a * rhs, b * rhs])
    }
}

impl Mul<VeryPackedM31> for VeryPackedQM31 {
    type Output = Self;

    fn mul(self, rhs: VeryPackedM31) -> Self::Output {
        let Self([a, b]) = self;
        Self([a * rhs, b * rhs])
    }
}

impl Mul<QM31> for VeryPackedQM31 {
    type Output = Self;

    fn mul(self, rhs: QM31) -> Self::Output {
        self * VeryPackedQM31::broadcast(rhs)
    }
}
