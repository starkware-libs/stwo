mod accumulation;
mod bit_reverse;
mod circle;
pub mod column;
pub mod error;
mod fri;
pub mod m31;
pub mod qm31;
mod quotients;
// pub mod packedm31;

use std::fmt::Debug;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
// use error::Error;
use once_cell::sync::Lazy;
use qm31::LoadSecureBaseField;

use self::m31::LoadBaseField;
use super::Backend;

static DEVICE: Lazy<Arc<CudaDevice>> = Lazy::new(|| CudaDevice::new(0).unwrap().load());

type Device = Arc<CudaDevice>;

trait Load {
    fn load(self) -> Self;
}

impl Load for Device {
    fn load(self) -> Self {
        bit_reverse::load_bit_reverse_ptx(&self);
        LoadBaseField::load(&self);
        LoadSecureBaseField::load(&self);
        self
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GpuBackend;

impl Backend for GpuBackend {}
