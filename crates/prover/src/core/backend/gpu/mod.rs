pub mod error;
pub mod m31;
pub mod testing;

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
// use error::Error;
use once_cell::sync::Lazy;

use crate::core::fields::m31::P;

const VECTOR_SIZE: usize = 16;

// #[allow(unused_macros)]
// #[macro_export]
// macro_rules! device {
//     () => {
//         DEVICE.read().unwrap()
//     };
// }

// TODO:: cleanup unwraps with error handling?
// (We can replace lazy statics with unsafe global references)
static DEVICE: Lazy<Arc<CudaDevice>> = Lazy::new(|| CudaDevice::new(0).unwrap().load());
static M512P: Lazy<CudaSlice<u32>> =
    Lazy::new(|| DEVICE.htod_copy([P; VECTOR_SIZE].to_vec()).unwrap());

trait InstructionSet {
    fn load(self) -> Self;
    fn vector_512_add_32u(&self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32>;
    fn vector_512_min_32u(&self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32>;
    fn vector_512_sub_32u(&self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32>;
    fn vector_512_set_32u(&self, val: &CudaSlice<u32>) -> CudaSlice<u32>;
}
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Device(Arc<CudaDevice>);

#[allow(dead_code)]
impl InstructionSet for Arc<CudaDevice> {
    // Returns a new Device with functions and constants pre-loaded
    fn load(mut self) -> Self {
        // let mut device = CudaDevice::new(0).unwrap();
        self.load_vector_512_operations();
        self
    }

    fn vector_512_add_32u(&self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32> {
        let add_kernel = self
            .get_func("instruction_set_op", "vector_512_add_32u")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(VECTOR_SIZE as u32);
        let out = self.alloc_zeros::<u32>(VECTOR_SIZE).unwrap(); // unsafe optimization exists with just alloc

        unsafe { add_kernel.launch(cfg, (in1, in2, &out)) }.unwrap();

        out
    }

    // TODO:: Optimize using __vminu4 (might not be possible)?
    fn vector_512_min_32u(&self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32> {
        let min_kernel = self
            .get_func("instruction_set_op", "vector_512_min_32u")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(VECTOR_SIZE as u32);
        let out = self.alloc_zeros::<u32>(VECTOR_SIZE).unwrap();

        unsafe { min_kernel.launch(cfg, (in1, in2, &out)) }.unwrap();

        out
    }

    fn vector_512_sub_32u(&self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32> {
        let sub_kernel = self
            .get_func("instruction_set_op", "vector_512_sub_32u")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(VECTOR_SIZE as u32);
        let out = self.alloc_zeros::<u32>(VECTOR_SIZE).unwrap();

        unsafe { sub_kernel.launch(cfg, (in1, in2, &out)) }.unwrap();

        out
    }

    fn vector_512_set_32u(&self, val: &CudaSlice<u32>) -> CudaSlice<u32> {
        let sub_kernel = self
            .get_func("instruction_set_op", "vector_512_set_32u")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(VECTOR_SIZE as u32);
        let out = self.alloc_zeros::<u32>(VECTOR_SIZE).unwrap();

        unsafe { sub_kernel.launch(cfg, (val, &out)) }.unwrap();

        out
    }
}

trait Load {
    fn load_vector_512_operations(&mut self);
}

impl Load for Arc<CudaDevice> {
    // Note:: intrinsic Math operations are not computing properly...? GPU issue?
    fn load_vector_512_operations(&mut self) {
        let vector_512_operations = compile_ptx("
            extern \"C\" __global__ void vector_512_add_32u( unsigned int *in1,  unsigned int *in2, unsigned int *out) {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int VECTOR_SIZE = 16; 
                if (i < VECTOR_SIZE) {
                    out[i] = in1[i] + in2[i];
                    // out[i] = __vadd4(in1[i], in2[i]);
                }
            }

            extern \"C\" __global__ void vector_512_min_32u(const unsigned int *in1, const unsigned int *in2, unsigned int *out) {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int VECTOR_SIZE = 16; 
                if (i  < VECTOR_SIZE) {
                    out[i] = min(in1[i], in2[i]);
                    // out[i] = __vminu4(in1[i], in2[i]);
                }
            }

            extern \"C\" __global__ void vector_512_sub_32u(const unsigned int *in1, const unsigned int *in2, unsigned int *out) {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int VECTOR_SIZE = 16; 
                if (i < VECTOR_SIZE) {
                    // out[i] = __vsub4(in1[i], in2[i]);
                    out[i] = in1[i] -in2[i]; 
                }
            }

            extern \"C\" __global__ void vector_512_set_32u(const unsigned int *val, unsigned int *out) {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int VECTOR_SIZE = 16; 
                if (i < VECTOR_SIZE) {
                    out[i] = *val;
                }
            }


        ").unwrap();

        self.load_ptx(
            vector_512_operations,
            "instruction_set_op",
            &[
                "vector_512_add_32u",
                "vector_512_min_32u",
                "vector_512_sub_32u",
                "vector_512_set_32u",
            ],
        )
        .unwrap();
    }
}

pub trait Kernelize {
    fn load_to_kernel();
    fn dtoh();
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use crate::core::backend::gpu::InstructionSet;
    // use super::Device;
    use crate::core::backend::gpu::{DEVICE, VECTOR_SIZE};
    use crate::core::fields::m31::{M31, P};
    #[test]
    pub fn test_vector_512_operations() {
        let values = [
            0,
            1,
            2,
            10,
            (P - 1) / 2,
            (P + 1) / 2,
            P - 2,
            P - 1,
            0,
            1,
            2,
            10,
            (P - 1) / 2,
            (P + 1) / 2,
            P - 2,
            P - 1,
        ]
        .map(M31::from_u32_unchecked);

        // let mut device = Device::new_loaded();

        let in1 = DEVICE
            .htod_copy(values.iter().map(|m31| m31.0).collect_vec())
            .unwrap();
        let in2 = DEVICE
            .htod_copy(values.iter().map(|m31| m31.0).collect_vec())
            .unwrap();
        let in3 = DEVICE.htod_copy(vec![P]).unwrap();

        // Test vector_512_add_32u
        let out = DEVICE.vector_512_add_32u(&in1, &in2);
        let out_host: Vec<u32> = DEVICE.dtoh_sync_copy(&out).unwrap();
        assert!(out_host == values.iter().map(|v| v.0 * 2).collect_vec());

        // Test vector_512_min_32u
        let out = DEVICE.vector_512_min_32u(&in1, &out);
        let out_host: Vec<u32> = DEVICE.dtoh_sync_copy(&out).unwrap();
        let out1: Vec<u32> = DEVICE.dtoh_sync_copy(&in1).unwrap();
        assert!(out_host == out1);

        // Test vector_512_sub_32u
        let out = DEVICE.vector_512_sub_32u(&in1, &in2);
        let out_host: Vec<u32> = DEVICE.dtoh_sync_copy(&out).unwrap();
        assert!(out_host == values.iter().map(|v| v.0 - v.0).collect_vec());

        // Test vector_512_set_32u
        let out = DEVICE.vector_512_set_32u(&in3);
        let out_host: Vec<u32> = DEVICE.dtoh_sync_copy(&out).unwrap();
        assert!(out_host == [P; VECTOR_SIZE]);
    }
}
