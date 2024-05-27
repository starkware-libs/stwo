pub mod error;
pub mod m31;

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};
// use error::Error;
use once_cell::sync::Lazy;

use self::m31::LoadPackedBaseField;
use crate::core::fields::m31::P;
const VECTOR_SIZE: usize = 16;

// TODO:: cleanup unwraps with error handling?
// (We can replace lazy statics with unsafe global references)
static DEVICE: Lazy<Arc<CudaDevice>> = Lazy::new(|| CudaDevice::new(0).unwrap().load());
static MODULUS: Lazy<CudaSlice<u32>> =
    Lazy::new(|| DEVICE.htod_copy([P; VECTOR_SIZE].to_vec()).unwrap());

type Device = Arc<CudaDevice>;

trait Load {
    fn load(self) -> Self;
}

impl Load for Device {
    fn load(self) -> Self {
        LoadPackedBaseField::load(&self);
        self
    }
}

#[cfg(test)]
mod test {}
