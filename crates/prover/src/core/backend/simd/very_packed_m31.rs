use num_traits::Zero;

use super::cm31::PackedCM31;
use super::m31::{PackedM31, N_LANES};
use super::qm31::PackedQM31;
use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::QM31;
use crate::math::vectorized::{Scalar, Vectorized};

pub const LOG_N_VERY_PACKED_ELEMS: u32 = 1;
pub const N_VERY_PACKED_ELEMS: usize = 1 << LOG_N_VERY_PACKED_ELEMS;

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

impl Scalar for M31 {}
impl Scalar for CM31 {}
impl Scalar for QM31 {}
impl Scalar for PackedM31 {}
impl Scalar for PackedCM31 {}
impl Scalar for PackedQM31 {}
