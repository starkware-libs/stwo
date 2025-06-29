use std::mem::MaybeUninit;
use std::ptr;

use stwo_prover::core::backend::web::webgpu::ByteSerialize;

use crate::poseidon::web::{
    ComputeCompositionPolynomialInput, ComputeCompositionPolynomialOutput, GpuExtendedColumn,
    GpuLookupElements, GpuOriginalColumn,
};
use crate::poseidon::PoseidonElements;

impl ByteSerialize for GpuExtendedColumn {}
impl ByteSerialize for GpuOriginalColumn {}
impl ByteSerialize for ComputeCompositionPolynomialOutput {}
impl ByteSerialize for ComputeCompositionPolynomialInput {}

#[allow(dead_code)]
impl ComputeCompositionPolynomialInput {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), std::mem::size_of::<Self>());
        unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const Self) }
    }
}

#[allow(dead_code)]
impl ComputeCompositionPolynomialOutput {
    /// Create directly on the heap without an intermediate stack copy.
    pub fn from_bytes_box(bytes: &[u8]) -> Box<Self> {
        assert_eq!(bytes.len(), core::mem::size_of::<Self>());

        let boxed_uninit = Box::<MaybeUninit<Self>>::new_uninit();
        let raw_ptr = Box::into_raw(boxed_uninit) as *mut Self as *mut u8;
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), raw_ptr, bytes.len());
            Box::from_raw(raw_ptr as *mut Self)
        }
    }
}

impl From<&PoseidonElements> for GpuLookupElements {
    fn from(value: &PoseidonElements) -> Self {
        GpuLookupElements {
            z: value.0.z.into(),
            alpha: value.0.alpha.into(),
            alpha_powers: value
                .0
                .alpha_powers
                .iter()
                .map(|&x| x.into())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }
}
