mod accumulation;
mod bit_reverse;
mod circle;
pub mod column;
pub mod error;
mod fri;
pub mod m31;
mod quotients;
// pub mod packedm31;

use std::fmt::Debug;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, ValidAsZeroBits};
// use error::Error;
use once_cell::sync::Lazy;

use self::m31::LoadBaseField;
use super::Backend;
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::QM31;

// TODO:: cleanup unwraps with error handling?
// (We can replace lazy statics with unsafe global references)
static DEVICE: Lazy<Arc<CudaDevice>> = Lazy::new(|| CudaDevice::new(0).unwrap().load());
// static MODULUS: Lazy<CudaSlice<u32>> =
//     Lazy::new(|| DEVICE.htod_copy([P; VECTOR_SIZE].to_vec()).unwrap());

type Device = Arc<CudaDevice>;

trait Load {
    fn load(self) -> Self;
}

impl Load for Device {
    fn load(self) -> Self {
        bit_reverse::load_bit_reverse_ptx(&self);
        LoadBaseField::load(&self);
        column::load_batch_inverse_ptx(&self);
        circle::load_circle(&self);
        self
    }
}

unsafe impl ValidAsZeroBits for M31 {}
unsafe impl ValidAsZeroBits for QM31 {}

#[derive(Copy, Clone, Debug)]
pub struct GpuBackend;

impl Backend for GpuBackend {}
