mod accumulation;
mod bit_reverse;
mod circle;
pub mod column;
pub mod error;
mod fri;
pub mod m31;
pub mod qm31;
mod quotients;

use std::fmt::Debug;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
// use error::Error;
use once_cell::sync::Lazy;

use super::Backend;

static DEVICE: Lazy<Arc<CudaDevice>> = Lazy::new(|| CudaDevice::new(0).unwrap().load());

type Device = Arc<CudaDevice>;

trait Load {
    fn load(self) -> Self;
}

impl Load for Device {
    fn load(self) -> Self {
        bit_reverse::load_bit_reverse_ptx(&self);
        m31::load_base_field(&self);
        qm31::load_secure_field(&self);
        self
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GpuBackend;

impl Backend for GpuBackend {}
