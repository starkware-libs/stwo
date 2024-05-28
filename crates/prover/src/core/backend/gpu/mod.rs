pub mod error;
pub mod m31;
mod bit_reverse;
mod column;
mod accumulation;
mod circle;
mod fri;

use std::fmt::Debug;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};
use cudarc::nvrtc::compile_ptx;
// use error::Error;
use once_cell::sync::Lazy;

use self::m31::LoadPackedBaseField;
use super::{Backend};
use crate::core::fields::m31::{BaseField, P};
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{ColumnSampleBatch, QuotientOps};
use crate::core::poly::circle::{CircleDomain, CircleEvaluation, SecureEvaluation,
};
use crate::core::poly::BitReversedOrder;
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



impl QuotientOps for GpuBackend {
    fn accumulate_quotients(
        _domain: CircleDomain,
        _columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        _random_coeff: SecureField,
        _sample_batches: &[ColumnSampleBatch],
    ) -> SecureEvaluation<Self> {
        todo!()
    }
}

