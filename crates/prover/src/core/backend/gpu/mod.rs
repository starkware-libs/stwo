pub mod m31;
pub mod testing;

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

// use self::m31::_512u;
#[allow(dead_code)]
pub struct Device {
    device: Arc<CudaDevice>,
}
#[allow(dead_code)]
impl Device {
    // Todo:: Load all instructions
    fn new() -> Self {
        let device = CudaDevice::new(0).unwrap();
        Self { device }
    }

    fn load_vector_512_add_32u(&mut self) {
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


        ").unwrap();

        self.device
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

    fn vector_512_add_32u(&mut self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32> {
        let add_kernel = self
            .device
            .get_func("instruction_set_op", "vector_512_add_32u")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(16);
        let out = self.device.alloc_zeros::<u32>(16).unwrap(); // unsafe optimization exists with just alloc

        unsafe { add_kernel.launch(cfg, (in1, in2, &out)) }.unwrap();

        out
    }

    // TODO:: Optimize using __vminu4 (is different function)?
    fn vector_512_min_32u(&mut self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32> {
        let min_kernel = self
            .device
            .get_func("instruction_set_op", "vector_512_min_32u")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(16);
        let out = self.device.alloc_zeros::<u32>(16).unwrap();

        unsafe { min_kernel.launch(cfg, (in1, in2, &out)) }.unwrap();

        out
    }

    fn vector_512_sub_32u(&mut self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32> {
        let sub_kernel = self
            .device
            .get_func("instruction_set_op", "vector_512_sub_32u")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(16);
        let out = self.device.alloc_zeros::<u32>(16).unwrap();

        unsafe { sub_kernel.launch(cfg, (in1, in2, &out)) }.unwrap();

        out
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

        let mut device = Device::new();
        device.load_vector_512_add_32u();

        let in1 = device
            .device
            .htod_copy(values.iter().map(|m31| m31.0).collect_vec())
            .unwrap();
        let in2 = device
            .device
            .htod_copy(values.iter().map(|m31| m31.0).collect_vec())
            .unwrap();

        // Test vector_512_add_32u
        let out = device.vector_512_add_32u(&in1, &in2);
        let out_host: Vec<u32> = device.device.dtoh_sync_copy(&out).unwrap();
        assert!(out_host == values.iter().map(|v| v.0 * 2).collect_vec());

        // Test vector_512_min_32u
        let out = device.vector_512_min_32u(&in1, &out);
        let out_host: Vec<u32> = device.device.dtoh_sync_copy(&out).unwrap();
        let out1: Vec<u32> = device.device.dtoh_sync_copy(&in1).unwrap();
        assert!(out_host == out1);

        // Test vector_512_sub_32u
        let out = device.vector_512_sub_32u(&in1, &in2);
        let out_host: Vec<u32> = device.device.dtoh_sync_copy(&out).unwrap();
        assert!(out_host == values.iter().map(|v| v.0 * 0).collect_vec());
    }
}
