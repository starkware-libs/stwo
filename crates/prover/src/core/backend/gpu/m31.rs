// CUDA implementation of packed m31
#[allow(unused_imports)]
use std::fmt::Display;
#[allow(unused_imports)]
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// use std::sync::Arc;
#[allow(unused_imports)]
use cudarc::driver::{CudaDevice, CudaSlice};
use cudarc::driver::{LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
#[allow(unused_imports)]
use num_traits::{One, Zero};
use once_cell::sync::Lazy;

#[allow(unused_imports)]
use super::Device;
use super::{DEVICE, MODULUS};
use crate::core::backend::gpu::InstructionSet;
#[allow(unused_imports)]
use crate::core::fields::m31::pow2147483645;
#[allow(unused_imports)]
use crate::core::fields::m31::{M31, P};
#[allow(unused_imports)]
use crate::core::fields::FieldExpOps;
pub const K_BLOCK_SIZE: usize = 16;
pub const PACKED_BASE_FIELD_SIZE: usize = 1 << 4;

type PackedBaseField = PackedM31;
type U32_16 = CudaSlice<u32>;

static EVENS_INTERLEAVE_EVENS: Lazy<CudaSlice<u32>> = Lazy::new(|| {
    DEVICE
        .htod_copy(vec![
            0b00000, 0b10000, 0b00010, 0b10010, 0b00100, 0b10100, 0b00110, 0b10110, 0b01000,
            0b11000, 0b01010, 0b11010, 0b01100, 0b11100, 0b01110, 0b11110,
        ])
        .unwrap()
});
static ODDS_INTERLEAVE_ODDS: Lazy<CudaSlice<u32>> = Lazy::new(|| {
    DEVICE
        .htod_copy(vec![
            0b00001, 0b10001, 0b00011, 0b10011, 0b00101, 0b10101, 0b00111, 0b10111, 0b01001,
            0b11001, 0b01011, 0b11011, 0b01101, 0b11101, 0b01111, 0b11111,
        ])
        .unwrap()
});

static SHIFT_COUNT_32: Lazy<CudaSlice<u32>> = Lazy::new(|| DEVICE.htod_copy(vec![32]).unwrap());
static SHIFT_COUNT_1: Lazy<CudaSlice<u32>> = Lazy::new(|| DEVICE.htod_copy(vec![1]).unwrap());

pub trait LoadPackedBaseField {
    fn mul(&self);
    fn reduce(&self);
    fn add(&self);
    fn neg(&self);
    fn sub(&self);
    fn load(&self);
}

impl LoadPackedBaseField for Device {
    fn reduce(&self) {
        let reduce_packed_base_field = compile_ptx(
            "
            extern \"C\" __global__ void reduce(unsigned int *f, const unsigned int *m) {
                unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int VECTOR_SIZE = 16; 
                if (tid  < VECTOR_SIZE) {
                    f[tid] = min(f[tid], f[tid] - m[tid]);
                }
            }
        ",
        )
        .unwrap();

        self.load_ptx(
            reduce_packed_base_field,
            "PackedBaseFieldReduce",
            &["reduce"],
        )
        .unwrap();
    }

    fn add(&self) {
        // Add word by word. Each word is in the range [0, 2P].

        // Apply min(c, c-P) to each word.
        // When c in [P,2P], then c-P in [0,P] which is always less than [P,2P].
        // When c in [0,P-1], then c-P in [2^32-P,2^32-1] which is always greater than
        let add_packed_base_field = compile_ptx("
            extern \"C\" __global__ void add(unsigned int *lhs, const unsigned int *rhs, const unsigned int *m) {
                unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int VECTOR_SIZE = 16; 
                if (tid  < VECTOR_SIZE) {
                    lhs[tid] += rhs[tid]; 
                    lhs[tid] = min(lhs[tid], lhs[tid] - m[tid]);
                }
            }
        ").unwrap();

        self.load_ptx(add_packed_base_field, "PackedBaseFieldAdd", &["add"])
            .unwrap();
    }

    fn neg(&self) {
        let neg_packed_base_field = compile_ptx(
            "
            extern \"C\" __global__ void neg(unsigned int *f, const unsigned int *m) {
                unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int VECTOR_SIZE = 16; 
                if (tid  < VECTOR_SIZE) {
                    f[tid] = f[tid] - m[tid];
                }
            }
        ",
        )
        .unwrap();

        self.load_ptx(neg_packed_base_field, "PackedBaseFieldNeg", &["neg"])
            .unwrap();
    }

    fn sub(&self) {
        // Subtract word by word. Each word is in the range [-P, P].

        // Apply min(c, c+P) to each word.
        // When c in [0,P], then c+P in [P,2P] which is always greater than [0,P].
        // When c in [2^32-P,2^32-1], then c+P in [0,P-1] which is always less than
        // [2^32-P,2^32-1].
        let sub_packed_base_field = compile_ptx(
            "
            extern \"C\" __global__ void sub(unsigned int *lhs, const unsigned int *rhs, const unsigned int *m) {
                unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int VECTOR_SIZE = 16; 
                if (tid  < VECTOR_SIZE) {
                    lhs[tid] = lhs[tid] - rhs[tid];
                    lhs[tid] = min(lhs[tid], lhs[tid] + m[tid]);
                }
            }
        ",
        )
        .unwrap();

        self.load_ptx(sub_packed_base_field, "PackedBaseFieldSub", &["sub"])
            .unwrap();
    }

    fn mul(&self) {
        let mul_packed_base_field = compile_ptx(
            "
            extern \"C\" __global__ void mul(unsigned int *lhs, unsigned int *rhs) {
                unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
                const unsigned int VECTOR_SIZE = 16; 
                if (tid  < VECTOR_SIZE) {
                    unsigned long long int a_e[8]; 

                }
            }
        ",
        )
        .unwrap();

        self.load_ptx(mul_packed_base_field, "PackedBaseFieldMul", &["mul"])
            .unwrap();
    }

    fn load(&self) {
        self.reduce();
        self.add();
        self.neg();
        self.sub();
        self.mul();
    }
}

// CUDA implementation
/// Stores 16 M31 elements in a custom 512-bit vector.
/// Each M31 element is unreduced in the range [0, P].
#[derive(Debug)]
#[repr(C)]
pub struct PackedM31(U32_16);

impl PackedBaseField {
    /// Constructs a new instance with all vector elements set to `value`.
    pub fn broadcast(M31(v): M31) -> Self {
        Self(DEVICE.htod_copy(vec![v; PACKED_BASE_FIELD_SIZE]).unwrap())
    }

    pub fn from_array(v: [M31; PACKED_BASE_FIELD_SIZE]) -> PackedBaseField {
        Self(DEVICE.htod_copy(v.map(|M31(v)| v).to_vec()).unwrap()) // Can we condense this?
    }

    pub fn to_array(self) -> [M31; PACKED_BASE_FIELD_SIZE] {
        let host = TryInto::<[u32; K_BLOCK_SIZE]>::try_into(
            DEVICE.dtoh_sync_copy(&self.reduce().0).unwrap(),
        )
        .unwrap();
        host.map(M31)
    }

    /// Reduces each word in the 512-bit register to the range `[0, P)`.
    pub fn reduce(self) -> PackedBaseField {
        let reduce_kernel = self
            .0
            .device()
            .get_func("PackedBaseFieldReduce", "reduce")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);

        unsafe { reduce_kernel.launch(cfg, (&self.0, &*MODULUS)) }.unwrap();

        self
    }

    // TODO:: Implement or Optimize as needed
    // /// Interleaves two vectors.
    // pub fn interleave(self, other: Self) -> (Self, Self) {
    //     let (a, b) = self.0.interleave(other.0);
    //     (Self(a), Self(b))
    // }

    // TODO:: Implement or Optimize as needed
    // /// Deinterleaves two vectors.
    // pub fn deinterleave(self, other: Self) -> (Self, Self) {
    //     let (a, b) = self.0.deinterleave(other.0);
    //     (Self(a), Self(b))
    // }

    /// Sums all the elements in the vector.
    pub fn pointwise_sum(self) -> M31 {
        self.to_array().into_iter().sum()
    }

    // TODO:: Implement or Optimize as needed
    // /// Doubles each element in the vector.
    // pub fn double(self) -> Self {
    //     // TODO: Make more optimal.
    //     self + self
    // }

    /// # Safety
    ///
    /// Vector elements must be in the range `[0, P]`.
    pub unsafe fn from_cuda_slice_unchecked(v: CudaSlice<u32>) -> Self {
        Self(v)
    }

    // TODO:: Implement or Optimize as needed
    // /// # Safety
    // ///
    // /// Behavior is undefined if the pointer does not have the same alignment as
    // /// [`PackedM31`]. The loaded `u32` values must be in the range `[0, P]`.
    // pub unsafe fn load(mem_addr: *const u32) -> Self {
    //     Self(ptr::read(mem_addr as *const u32x16))
    // }

    // TODO:: Implement or Optimize as needed
    // /// # Safety
    // ///
    // /// Behavior is undefined if the pointer does not have the same alignment as
    // /// [`PackedM31`]. The loaded `u32` values must be in the range `[0, P]`.
    // pub unsafe fn load(mem_addr: *const u32) -> Self {
    //     Self(ptr::read(mem_addr as *const u32x16))
    // }

    // TODO:: Implement or Optimize as needed
    // /// # Safety
    // ///
    // /// Behavior is undefined if the pointer does not have the same alignment as
    // /// [`PackedM31`].
    // pub unsafe fn store(self, dst: *mut u32) {
    //     ptr::write(dst as *mut u32x16, self.0)
    // }
}

// impl Display for PackedBaseField {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let v = self.clone().to_array();
//         for elem in v.iter() {
//             write!(f, "{} ", elem)?;
//         }
//         Ok(())
//     }
// }

// Clone is a device to device copy
impl Clone for PackedBaseField {
    fn clone(&self) -> Self {
        let mut out = unsafe { self.0.device().alloc::<u32>(PACKED_BASE_FIELD_SIZE) }.unwrap();
        self.0.device().dtod_copy(&self.0, &mut out).unwrap();
        Self(out)
    }
}

impl Add for PackedBaseField {
    type Output = Self;

    // Do we need to consume self? (if not we can change back add implementation to allocate new
    // vector)
    /// Adds two packed M31 elements, and reduces the result to the range `[0,P]`.
    /// Each value is assumed to be in unreduced form, [0, P] including P.
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let add_kernel = self
            .0
            .device()
            .get_func("PackedBaseFieldAdd", "add")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);

        unsafe { add_kernel.launch(cfg, (&self.0, &rhs.0, &*MODULUS)) }.unwrap();

        self
    }
}

// Adds in place
impl AddAssign for PackedBaseField {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        let add_kernel = self
            .0
            .device()
            .get_func("PackedBaseFieldAdd", "add")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);

        unsafe { add_kernel.launch(cfg, (&self.0, &rhs.0, &*MODULUS)) }.unwrap();
    }
}

impl Mul for PackedBaseField {
    type Output = Self;

    /// Computes the product of two packed M31 elements
    /// Each value is assumed to be in unreduced form, [0, P] including P.
    /// Returned values are in unreduced form, [0, P] including P.
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        // Set up a word s.t. the lower half of each 64-bit word has the odd 32-bit words of
        // the first operand.
        let val0_o = DEVICE.vector_512_srli_i64(&self.0, &SHIFT_COUNT_32);
        // Set up a word s.t. the lower half of each 64-bit word has the even 32-bit words of
        // the first operand.
        let val0_e = self.0;

        // Double the second operand.
        let val1 = DEVICE.vector_512_add_u32(&rhs.0, &rhs.0);
        let val1_o = DEVICE.vector_512_srli_i64(&val1, &SHIFT_COUNT_32);
        let val1_e = val1;

        // To compute prod = val0 * val1 start by multiplying
        // val0_e/o by val1_e/o.
        let prod_e_dbl = DEVICE.vector_512_mul_u32(&val0_e, &val1_e);
        let prod_o_dbl = DEVICE.vector_512_mul_u32(&val0_o, &val1_o);

        // The result of a multiplication holds val1*twiddle_dbl in as 64-bits.
        // Each 64b-bit word looks like this:
        //               1    31       31    1
        // prod_e_dbl - |0|prod_e_h|prod_e_l|0|
        // prod_o_dbl - |0|prod_o_h|prod_o_l|0|

        // Interleave the even words of prod_e_dbl with the even words of prod_o_dbl:
        let prod_ls =
            DEVICE.vector_512_permute_u32(&prod_e_dbl, &prod_o_dbl, &EVENS_INTERLEAVE_EVENS); // prod_ls -    |prod_o_l|0|prod_e_l|0|

        // Divide by 2:
        let prod_ls = Self(DEVICE.vector_512_srli_i64(&prod_ls, &SHIFT_COUNT_1));
        // prod_ls -    |0|prod_o_l|0|prod_e_l|

        // Interleave the odd words of prod_e_dbl with the odd words of prod_o_dbl:
        let prod_hs =
            Self(DEVICE.vector_512_permute_u32(&prod_e_dbl, &prod_o_dbl, &ODDS_INTERLEAVE_ODDS));
        // prod_hs -    |0|prod_o_h|0|prod_e_h|

        Self::add(prod_ls, prod_hs)
    }
}

// impl MulAssign for PackedBaseField {
//     #[inline(always)]
//     fn mul_assign(&mut self, rhs: Self) {
//         *self = *self * rhs;
//     }
// }

impl Neg for PackedBaseField {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        let neg_kernel = self
            .0
            .device()
            .get_func("PackedBaseFieldNeg", "neg")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);

        unsafe { neg_kernel.launch(cfg, (&self.0, &*MODULUS)) }.unwrap();

        self
    }
}

/// Subtracts two packed M31 elements, and reduces the result to the range `[0,P]`.
/// Each value is assumed to be in unreduced form, [0, P] including P.
impl Sub for PackedBaseField {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let sub_kernel = self
            .0
            .device()
            .get_func("PackedBaseFieldSub", "sub")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);

        unsafe { sub_kernel.launch(cfg, (&self.0, &rhs.0, &*MODULUS)) }.unwrap();

        self
    }
}

// Subtract in place
impl SubAssign for PackedBaseField {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        let sub_kernel = self
            .0
            .device()
            .get_func("PackedBaseFieldSub", "sub")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);

        unsafe { sub_kernel.launch(cfg, (&self.0, &rhs.0, &*MODULUS)) }.unwrap();
    }
}

impl Zero for PackedBaseField {
    fn zero() -> Self {
        Self(DEVICE.alloc_zeros::<u32>(PACKED_BASE_FIELD_SIZE).unwrap())
    }

    // TODO:: Optimize? It currently does a htod copy
    fn is_zero(&self) -> bool {
        self.clone().to_array().iter().all(M31::is_zero)
    }
}

impl One for PackedBaseField {
    fn one() -> Self {
        Self(DEVICE.htod_copy([1; K_BLOCK_SIZE].to_vec()).unwrap())
    }
}

// impl FieldExpOps for PackedBaseField {
//     fn inverse(&self) -> Self {
//         assert!(!self.is_zero(), "0 has no inverse");
//         pow2147483645(*self)
//     }
// }

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use itertools::Itertools;

    use super::PackedBaseField;
    #[allow(unused_imports)]
    use crate::core::backend::gpu::{DEVICE, MODULUS};
    #[allow(unused_imports)]
    use crate::core::fields::m31::{M31, P};
    #[allow(unused_imports)]
    use crate::core::fields::{Field, FieldExpOps};
    /// Tests field operations where field elements are in reduced form.
    #[test]
    fn test_gpu_basic_ops() {
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

        let avx_values1 = PackedBaseField::from_array(values);
        let avx_values2 = avx_values1.clone();
        let avx_values3 = avx_values1.clone();
        let avx_values4 = avx_values1.clone();

        assert_eq!(
            (avx_values1 + avx_values2)
                .to_array()
                .into_iter()
                .collect_vec(),
            values.iter().map(|x| x.double()).collect_vec()
        );

        assert_eq!(
            (avx_values4.clone() * avx_values4.clone())
                .to_array()
                .into_iter()
                .collect_vec(),
            values.iter().map(|x| x.square()).collect_vec()
        );

        assert_eq!(
            (-avx_values3).to_array().into_iter().collect_vec(),
            values.iter().map(|x| -*x).collect_vec()
        );
    }
}
