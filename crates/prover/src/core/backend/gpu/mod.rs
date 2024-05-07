pub mod error;
pub mod m31;
pub mod testing;

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
// use error::Error;
use lazy_static::lazy_static;

use crate::core::fields::m31::P;

const VECTOR_SIZE: usize = 16;

// TODO:: cleanup unwraps with error handling?
// (We can replace lazy statics with unsafe global references)
lazy_static! {
    static ref DEVICE: Mutex<Device> = Mutex::new(Device::new_loaded());
    static ref M512P: Mutex<CudaSlice<u32>> = Mutex::new(
        DEVICE
            .lock()
            .unwrap()
            .0
            .htod_copy([P; VECTOR_SIZE].to_vec())
            .unwrap()
    );
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Device(Arc<CudaDevice>);

#[allow(dead_code)]
impl Device {
    // Returns a new Device with functions and constants pre-loaded
    fn new_loaded() -> Self {
        let mut device = Self(CudaDevice::new(0).unwrap());
        device.load_vector_512_operations();
        device
    }

    fn vector_512_add_32u(&mut self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32> {
        let add_kernel = self
            .0
            .get_func("instruction_set_op", "vector_512_add_32u")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(VECTOR_SIZE as u32);
        let out = self.0.alloc_zeros::<u32>(VECTOR_SIZE).unwrap(); // unsafe optimization exists with just alloc

        unsafe { add_kernel.launch(cfg, (in1, in2, &out)) }.unwrap();

        out
    }

    // TODO:: Optimize using __vminu4 (might not be possible)?
    fn vector_512_min_32u(&mut self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32> {
        let min_kernel = self
            .0
            .get_func("instruction_set_op", "vector_512_min_32u")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(VECTOR_SIZE as u32);
        let out = self.0.alloc_zeros::<u32>(VECTOR_SIZE).unwrap();

        unsafe { min_kernel.launch(cfg, (in1, in2, &out)) }.unwrap();

        out
    }

    fn vector_512_sub_32u(&mut self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32> {
        let sub_kernel = self
            .0
            .get_func("instruction_set_op", "vector_512_sub_32u")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(VECTOR_SIZE as u32);
        let out = self.0.alloc_zeros::<u32>(VECTOR_SIZE).unwrap();

        unsafe { sub_kernel.launch(cfg, (in1, in2, &out)) }.unwrap();

        out
    }

    fn vector_512_set_32u(&mut self, val: &CudaSlice<u32>) -> CudaSlice<u32> {
        let sub_kernel = self
            .0
            .get_func("instruction_set_op", "vector_512_set_32u")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(VECTOR_SIZE as u32);
        let out = self.0.alloc_zeros::<u32>(VECTOR_SIZE).unwrap();

        unsafe { sub_kernel.launch(cfg, (val, &out)) }.unwrap();

        out
    }

    // If we can make this work, it will clean up code dramatically
    // This is an attempt at getting an m512p constant pointer from the kernel
    // fn get_m512p(&mut self) -> CudaSlice<u32> {
    //     let m512p_kernel = self.device.get_func("constants", "get_m512p").unwrap();

    //     let cfg: LaunchConfig = LaunchConfig::for_num_elems(1);
    //     let out = self.device.alloc_zeros::<u32>(1).unwrap();

    //     unsafe { m512p_kernel.launch(cfg, (&out,)) }.unwrap();

    //     out
    // }

    // Gets Constant, if exists
    // fn get_constant(&mut self, str: &str) -> Result<Arc<CudaSlice<u32>>, Error> {
    //     if let Some(value) = self.global.constants.iter().find(|c| c.0.eq(str)) {
    //         Ok(value.clone().1)
    //     } else {
    //         Err(Error::FindConstantError(str.to_string()))
    //     }
    // }
}

trait Load {
    fn load_vector_512_operations(&mut self);
}

impl Load for Device {
    fn load_vector_512_operations(&mut self) {
        let vector_512_operations = compile_ptx("
            extern \"C\" __global__ void vector_512_add_32u(const unsigned int *in1, const unsigned int *in2, unsigned int *out) {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int VECTOR_SIZE = 16; 
                if (i < VECTOR_SIZE) {
                    out[i] = in1[i] + in2[i];
                }
            }

            extern \"C\" __global__ void vector_512_min_32u(const unsigned int *in1, const unsigned int *in2, unsigned int *out) {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int VECTOR_SIZE = 16; 
                if (i < VECTOR_SIZE) {
                    out[i] = in1[i] < in2[i] ? in1[i] : in2[i];
                }
            }

            extern \"C\" __global__ void vector_512_sub_32u(const unsigned int *in1, const unsigned int *in2, unsigned int *out) {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int VECTOR_SIZE = 16; 
                if (i < VECTOR_SIZE) {
                    out[i] = __vsub4(in1[i], in2[i]);
                }
            }

            extern \"C\" __global__ void vector_512_set_32u(const unsigned int val, unsigned int *out) {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int VECTOR_SIZE = 16; 
                if (i < VECTOR_SIZE) {
                    out[i] = val;
                }
            }


        ").unwrap();

        self.0
            .load_ptx(
                vector_512_operations,
                "instruction_set_op",
                &[
                    "vector_512_add_32u",
                    "vector_512_min_32u",
                    "vector_512_sub_32u",
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

    use super::Device;
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

        let mut device = Device::new_loaded();

        let in1 = device
            .0
            .htod_copy(values.iter().map(|m31| m31.0).collect_vec())
            .unwrap();
        let in2 = device
            .0
            .htod_copy(values.iter().map(|m31| m31.0).collect_vec())
            .unwrap();

        // Test vector_512_add_32u
        let out = device.vector_512_add_32u(&in1, &in2);
        let out_host: Vec<u32> = device.0.dtoh_sync_copy(&out).unwrap();
        assert!(out_host == values.iter().map(|v| v.0 * 2).collect_vec());

        // Test vector_512_min_32u
        let out = device.vector_512_min_32u(&in1, &out);
        let out_host: Vec<u32> = device.0.dtoh_sync_copy(&out).unwrap();
        let out1: Vec<u32> = device.0.dtoh_sync_copy(&in1).unwrap();
        assert!(out_host == out1);

        // Test vector_512_sub_32u
        let out = device.vector_512_sub_32u(&in1, &in2);
        let out_host: Vec<u32> = device.0.dtoh_sync_copy(&out).unwrap();
        assert!(out_host == values.iter().map(|v| v.0 - v.0).collect_vec());
    }
}
