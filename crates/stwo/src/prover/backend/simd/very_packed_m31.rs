use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};

use bytemuck::{Pod, Zeroable};
use num_traits::{One, Zero};

use super::cm31::PackedCM31;
use super::m31::{PackedM31, N_LANES};
use super::qm31::PackedQM31;
use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::QM31;
use crate::core::fields::{batch_inverse_in_place, FieldExpOps};

pub const LOG_N_VERY_PACKED_ELEMS: u32 = 1;
pub const N_VERY_PACKED_ELEMS: usize = 1 << LOG_N_VERY_PACKED_ELEMS;

#[derive(Clone, Debug, Copy)]
#[repr(transparent)]
pub struct Vectorized<A: Copy, const N: usize>(pub [A; N]);

impl<A: Copy, const N: usize> Vectorized<A, N> {
    pub fn from_fn<F>(cb: F) -> Self
    where
        F: FnMut(usize) -> A,
    {
        Vectorized(std::array::from_fn(cb))
    }
}

impl<A: Copy, const N: usize> From<[A; N]> for Vectorized<A, N> {
    fn from(array: [A; N]) -> Self {
        Vectorized(array)
    }
}

unsafe impl<A: Copy, const N: usize> Zeroable for Vectorized<A, N> {
    fn zeroed() -> Self {
        unsafe { core::mem::zeroed() }
    }
}

unsafe impl<A: Pod, const N: usize> Pod for Vectorized<A, N> {}

pub type VeryPackedM31 = Vectorized<PackedM31, N_VERY_PACKED_ELEMS>;
pub type VeryPackedCM31 = Vectorized<PackedCM31, N_VERY_PACKED_ELEMS>;
pub type VeryPackedQM31 = Vectorized<PackedQM31, N_VERY_PACKED_ELEMS>;
pub type VeryPackedBaseField = VeryPackedM31;
pub type VeryPackedSecureField = VeryPackedQM31;

impl VeryPackedM31 {
    pub fn broadcast(value: M31) -> Self {
        Self::from_fn(|_| PackedM31::broadcast(value))
    }

    pub fn from_array(values: [M31; N_LANES * N_VERY_PACKED_ELEMS]) -> VeryPackedM31 {
        Self::from_fn(|i| {
            let start = i * N_LANES;
            let end = start + N_LANES;
            PackedM31::from_array(values[start..end].try_into().unwrap())
        })
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
        Self::from_fn(|_| PackedCM31::broadcast(value))
    }
}

impl VeryPackedQM31 {
    pub fn broadcast(value: QM31) -> Self {
        Self::from_fn(|_| PackedQM31::broadcast(value))
    }

    pub fn from_very_packed_m31s([a, b, c, d]: [VeryPackedM31; 4]) -> Self {
        Self::from_fn(|i| PackedQM31::from_packed_m31s([a.0[i], b.0[i], c.0[i], d.0[i]]))
    }

    pub fn into_very_packed_m31s(self) -> [VeryPackedM31; 4] {
        std::array::from_fn(|i| VeryPackedM31::from(self.0.map(|v| v.into_packed_m31s()[i])))
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

trait Scalar {}
impl Scalar for M31 {}
impl Scalar for CM31 {}
impl Scalar for QM31 {}
impl Scalar for PackedM31 {}
impl Scalar for PackedCM31 {}
impl Scalar for PackedQM31 {}

impl<A: Add<B> + Copy, B: Copy, const N: usize> Add<Vectorized<B, N>> for Vectorized<A, N>
where
    <A as Add<B>>::Output: Copy,
{
    type Output = Vectorized<A::Output, N>;

    fn add(self, other: Vectorized<B, N>) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i] + other.0[i])
    }
}

impl<A: Add<B> + Copy, B: Scalar + Copy, const N: usize> Add<B> for Vectorized<A, N>
where
    <A as Add<B>>::Output: Copy,
{
    type Output = Vectorized<A::Output, N>;

    fn add(self, other: B) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i] + other)
    }
}

impl<A: Sub<B> + Copy, B: Copy, const N: usize> Sub<Vectorized<B, N>> for Vectorized<A, N>
where
    <A as Sub<B>>::Output: Copy,
{
    type Output = Vectorized<A::Output, N>;

    fn sub(self, other: Vectorized<B, N>) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i] - other.0[i])
    }
}

impl<A: Sub<B> + Copy, B: Scalar + Copy, const N: usize> Sub<B> for Vectorized<A, N>
where
    <A as Sub<B>>::Output: Copy,
{
    type Output = Vectorized<A::Output, N>;

    fn sub(self, other: B) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i] - other)
    }
}

impl<A: Mul<B> + Copy, B: Copy, const N: usize> Mul<Vectorized<B, N>> for Vectorized<A, N>
where
    <A as Mul<B>>::Output: Copy,
{
    type Output = Vectorized<A::Output, N>;

    fn mul(self, other: Vectorized<B, N>) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i] * other.0[i])
    }
}

impl<A: Mul<B> + Copy, B: Scalar + Copy, const N: usize> Mul<B> for Vectorized<A, N>
where
    <A as Mul<B>>::Output: Copy,
{
    type Output = Vectorized<A::Output, N>;

    fn mul(self, other: B) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i] * other)
    }
}

impl<A: AddAssign<B> + Copy, B: Copy, const N: usize> AddAssign<Vectorized<B, N>>
    for Vectorized<A, N>
{
    fn add_assign(&mut self, other: Vectorized<B, N>) {
        for i in 0..N {
            self.0[i] += other.0[i];
        }
    }
}

impl<A: AddAssign<B> + Copy, B: Scalar + Copy, const N: usize> AddAssign<B> for Vectorized<A, N> {
    fn add_assign(&mut self, other: B) {
        for i in 0..N {
            self.0[i] += other;
        }
    }
}

impl<A: MulAssign<B> + Copy, B: Copy, const N: usize> MulAssign<Vectorized<B, N>>
    for Vectorized<A, N>
{
    fn mul_assign(&mut self, other: Vectorized<B, N>) {
        for i in 0..N {
            self.0[i] *= other.0[i];
        }
    }
}

impl<A: Neg + Copy, const N: usize> Neg for Vectorized<A, N>
where
    <A as Neg>::Output: Copy,
{
    type Output = Vectorized<A::Output, N>;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i].neg())
    }
}

impl<A: Zero + Copy, const N: usize> Zero for Vectorized<A, N> {
    fn zero() -> Self {
        Vectorized::from_fn(|_| A::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(A::is_zero)
    }
}

impl<A: One + Copy, const N: usize> One for Vectorized<A, N> {
    fn one() -> Self {
        Vectorized::from_fn(|_| A::one())
    }
}

impl<A: FieldExpOps + Zero + Copy, const N: usize> FieldExpOps for Vectorized<A, N> {
    fn inverse(&self) -> Self {
        let mut dst = [A::zero(); N];
        batch_inverse_in_place(&self.0, &mut dst);
        dst.into()
    }
}
