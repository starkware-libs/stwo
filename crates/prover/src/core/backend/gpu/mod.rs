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

// TODO:: cleanup unwraps with error handling?
// (We can replace lazy statics with unsafe global references)
static DEVICE: Lazy<Arc<CudaDevice>> = Lazy::new(|| CudaDevice::new(0).unwrap().load());
static M512P: Lazy<CudaSlice<u32>> =
    Lazy::new(|| DEVICE.htod_copy([P; VECTOR_SIZE].to_vec()).unwrap());

trait InstructionSet {
    fn load(self) -> Self;
    fn vector_512_add_u32(&self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32>;
    fn vector_512_min_u32(&self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32>;
    fn vector_512_sub_u32(&self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32>;
    fn vector_512_set_u32(&self, val: &CudaSlice<u32>) -> CudaSlice<u32>;
    fn vector_512_srli_i64(&self, in1: &CudaSlice<u32>, shift: &CudaSlice<u32>) -> CudaSlice<u32>;
    fn vector_512_mul_u32(&self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32>;
    fn vector_512_permute_u32(
        &self,
        in1: &CudaSlice<u32>,
        in2: &CudaSlice<u32>,
        idx: &CudaSlice<u32>,
    ) -> CudaSlice<u32>;
    fn vector_512_clone_slice(&self, in1: &CudaSlice<u32>) -> CudaSlice<u32>;
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Device(Arc<CudaDevice>);

// TODO (Cleanup): Refactor kernel launching into a single function call
// TODO (Research): inline optimizations exist
// TODO (Optimization): unsafe optimization exists with just alloc instead of alloc_zeros
#[allow(dead_code)]
impl InstructionSet for Arc<CudaDevice> {
    // Returns a new Device with functions and constants pre-loaded
    fn load(mut self) -> Self {
        // let mut device = CudaDevice::new(0).unwrap();
        self.load_vector_512_operations();
        self
    }

    fn vector_512_add_u32(&self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32> {
        let add_kernel = self
            .get_func("instruction_set", "vector_512_add_u32")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(VECTOR_SIZE as u32);
        let out = self.alloc_zeros::<u32>(VECTOR_SIZE).unwrap();

        unsafe { add_kernel.launch(cfg, (in1, in2, &out)) }.unwrap();

        out
    }

    fn vector_512_min_u32(&self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32> {
        let min_kernel = self
            .get_func("instruction_set", "vector_512_min_u32")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(VECTOR_SIZE as u32);
        let out = self.alloc_zeros::<u32>(VECTOR_SIZE).unwrap();

        unsafe { min_kernel.launch(cfg, (in1, in2, &out)) }.unwrap();

        out
    }

    fn vector_512_sub_u32(&self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32> {
        let sub_kernel = self
            .get_func("instruction_set", "vector_512_sub_u32")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(VECTOR_SIZE as u32);
        let out = self.alloc_zeros::<u32>(VECTOR_SIZE).unwrap();

        unsafe { sub_kernel.launch(cfg, (in1, in2, &out)) }.unwrap();

        out
    }

    fn vector_512_set_u32(&self, val: &CudaSlice<u32>) -> CudaSlice<u32> {
        let sub_kernel = self
            .get_func("instruction_set", "vector_512_set_u32")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(VECTOR_SIZE as u32);
        let out = self.alloc_zeros::<u32>(VECTOR_SIZE).unwrap();

        unsafe { sub_kernel.launch(cfg, (val, &out)) }.unwrap();

        out
    }

    fn vector_512_clone_slice(&self, val: &CudaSlice<u32>) -> CudaSlice<u32> {
        let clone_kernel = self
            .get_func("instruction_set", "vector_512_clone_slice")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(VECTOR_SIZE as u32);
        let out = self.alloc_zeros::<u32>(VECTOR_SIZE).unwrap();

        unsafe { clone_kernel.launch(cfg, (val, &out)) }.unwrap();

        out
    }
    fn vector_512_srli_i64(&self, in1: &CudaSlice<u32>, shift: &CudaSlice<u32>) -> CudaSlice<u32> {
        let shift_right_kernel = self
            .get_func("instruction_set", "vector_512_srli_u64")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems((VECTOR_SIZE / 2) as u32);
        let out = self.alloc_zeros::<u32>(VECTOR_SIZE).unwrap();

        unsafe { shift_right_kernel.launch(cfg, (in1, shift, &out)) }.unwrap();

        out
    }

    // CUDA implementation of AVX512 _mm512_mul_epu32
    fn vector_512_mul_u32(&self, in1: &CudaSlice<u32>, in2: &CudaSlice<u32>) -> CudaSlice<u32> {
        let mul_kernel = self
            .get_func("instruction_set", "vector_512_mul_u32")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems((VECTOR_SIZE / 2) as u32);
        let out = self.alloc_zeros::<u32>(VECTOR_SIZE).unwrap();

        unsafe { mul_kernel.launch(cfg, (in1, in2, &out)) }.unwrap();

        out
    }

    fn vector_512_permute_u32(
        &self,
        in1: &CudaSlice<u32>,
        in2: &CudaSlice<u32>,
        idx: &CudaSlice<u32>,
    ) -> CudaSlice<u32> {
        let permute_kernel = self
            .get_func("instruction_set", "vector_512_permute_u32")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(VECTOR_SIZE as u32);
        let out = self.alloc_zeros::<u32>(VECTOR_SIZE).unwrap();

        unsafe { permute_kernel.launch(cfg, (in1, in2, idx, &out)) }.unwrap();

        out
    }
}

/// Implements loading capability for the Device
/// * 'load_vector_512_operations' - basic operations for 512 bit vector
trait Load {
    fn load_vector_512_operations(&mut self);
}

/// 512 bit vector operations
impl Load for Arc<CudaDevice> {
    // Intrinsic operations don't work because they don't overflow (ie. __viadd4 or __vminu4)
    // TODO:: Look at performance trade off of explicit addition versus intrinsic built addition
    fn load_vector_512_operations(&mut self) {
        let vector_512_operations = compile_ptx("
            extern \"C\" __global__ void vector_512_add_u32( unsigned int *in1,  unsigned int *in2, unsigned int *out) {
                unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int VECTOR_SIZE = 16; 
                if (tid < VECTOR_SIZE) {
                    out[tid] = in1[tid] + in2[tid];
                    // out[tid] = __vadd4(in1[tid], in2[tid]);
                }
            }

            extern \"C\" __global__ void vector_512_min_u32(const unsigned int *in1, const unsigned int *in2, unsigned int *out) {
                unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int VECTOR_SIZE = 16; 
                if (tid  < VECTOR_SIZE) {
                    out[tid] = min(in1[tid], in2[tid]);
                    // out[tid] = __vminu4(in1[tid], in2[tid]);
                }
            }

            extern \"C\" __global__ void vector_512_sub_u32(const unsigned int *in1, const unsigned int *in2, unsigned int *out) {
                unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int VECTOR_SIZE = 16; 
                if (tid < VECTOR_SIZE) {
                    // out[tid] = __vsub4(in1[tid], in2[tid]);
                    out[tid] = in1[tid] -in2[tid]; 
                }
            }

            extern \"C\" __global__ void vector_512_set_u32(const unsigned int *val, unsigned int *out) {
                unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int VECTOR_SIZE = 16; 
                if (tid < VECTOR_SIZE) {
                    out[tid] = *val;
                }
            }

            extern \"C\" __global__ void vector_512_clone_slice(const unsigned int *in1, unsigned int *out) {
                unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int VECTOR_SIZE = 16; 
                if (tid < VECTOR_SIZE) {
                    out[tid] = in1[tid];
                }
            }

            extern \"C\" __global__ void vector_512_srli_u64(const unsigned long long int *in, unsigned long long int *out, const unsigned int *shift_count) {
                int tid = threadIdx.x + blockIdx.x * blockDim.x;
                if (tid < 8) {
                    out[tid] = in[tid] >> *shift_count;
                }
            }
            
            extern \"C\" __global__ void vector_512_mul_u32(const unsigned long long int *in1, const unsigned long long int *in2, unsigned long long int *out) {
                int tid = threadIdx.x + blockIdx.x * blockDim.x;
                if (tid < 8) {
                    out[tid] = in1[tid] * in2[tid];
                }
            }

            extern \"C\" __global__ void vector_512_permute_u32(const unsigned int *in1, const unsigned int *in2, const unsigned int *idx, unsigned int *out) {
                int tid = threadIdx.x; // enforce single warp
                const unsigned int VECTOR_SIZE = 16; 
                if (tid < VECTOR_SIZE) {
                    unsigned int a_val = in1[tid];
                    unsigned int b_val = in2[tid];
                    unsigned int index = idx[tid];

                    if (index & 0x80000000) { // ie. index < 0
                        out[tid] = __shfl_sync(0xFFFFFFFF, b_val, index & 0x0F);
                    }
                    else {
                        out[tid] = __shfl_sync(0xFFFFFFFF, a_val, index);
                    }
                }
            }

        ").unwrap();

        self.load_ptx(
            vector_512_operations,
            "instruction_set",
            &[
                "vector_512_add_u32",
                "vector_512_min_u32",
                "vector_512_sub_u32",
                "vector_512_set_u32",
                "vector_512_clone_slice",
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

        // Create initial device memory variables
        let in1 = DEVICE
            .htod_copy(values.iter().map(|m31| m31.0).collect_vec())
            .unwrap();
        let in2 = DEVICE
            .htod_copy(values.iter().map(|m31| m31.0).collect_vec())
            .unwrap();
        let in3 = DEVICE.htod_copy(vec![P]).unwrap();

        // Test vector_512_add_u32
        let out = DEVICE.vector_512_add_u32(&in1, &in2);
        let out_host: Vec<u32> = DEVICE.dtoh_sync_copy(&out).unwrap();
        assert!(out_host == values.iter().map(|v| v.0 * 2).collect_vec());

        // Test vector_512_min_u32
        let out = DEVICE.vector_512_min_u32(&in1, &out);
        let out_host: Vec<u32> = DEVICE.dtoh_sync_copy(&out).unwrap();
        let out1: Vec<u32> = DEVICE.dtoh_sync_copy(&in1).unwrap();
        assert!(out_host == out1);

        // Test vector_512_sub_u32
        let out = DEVICE.vector_512_sub_u32(&in1, &in2);
        let out_host: Vec<u32> = DEVICE.dtoh_sync_copy(&out).unwrap();
        assert!(out_host == values.iter().map(|v| v.0 - v.0).collect_vec());

        // Test vector_512_set_u32
        let out = DEVICE.vector_512_set_u32(&in3);
        let out_host: Vec<u32> = DEVICE.dtoh_sync_copy(&out).unwrap();
        assert!(out_host == [P; VECTOR_SIZE]);

        // Test vector_512_clone_slice
        let out = DEVICE.vector_512_clone_slice(&out);
        let out_host: Vec<u32> = DEVICE.dtoh_sync_copy(&out).unwrap();
        assert!(out_host == [P; VECTOR_SIZE]);
    }
}
