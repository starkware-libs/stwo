mod accumulation;
mod bit_reverse;
mod circle;
mod column;
pub mod error;
mod fri;
pub mod m31;
mod quotients;

use std::fmt::Debug;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};
use cudarc::nvrtc::compile_ptx;
// use error::Error;
use once_cell::sync::Lazy;

use self::m31::LoadPackedBaseField;
use super::Backend;
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
        let ptx_src = include_str!("bit_reverse.cu");
        let ptx = compile_ptx(ptx_src).unwrap();
        self.load_ptx(ptx, "bit_reverse", &["kernel"]).unwrap();
        self
    }
}

#[derive(Copy, Clone, Debug)]
struct GpuBackend;

impl Backend for GpuBackend {}
