// // CUDA implementation of packed m31
// use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};
// use cudarc::nvrtc::compile_ptx;
// use num_traits::{One, Zero};

// use super::{Device, DEVICE, MODULUS};
// // use crate::core::fields::m31::pow2147483645;
// use crate::core::fields::m31::M31;
// // use crate::core::fields::FieldExpOps;
// pub const K_BLOCK_SIZE: usize = 16;
// pub const PACKED_BASE_FIELD_SIZE: usize = 1 << 4;

// type PackedBaseField = PackedM31;
// type U32_16 = CudaSlice<u32>;
// pub trait LoadPackedBaseField {
//     fn mul_packed(&self);
//     fn mul(&self);
//     fn reduce(&self);
//     fn add(&self);
//     fn neg(&self);
//     fn sub(&self);
//     fn load(&self);
// }

// impl LoadPackedBaseField for Device {
//     fn reduce(&self) {
//         let reduce_packed_base_field = compile_ptx(
//             "
//             extern \"C\" __global__ void reduce(unsigned int *f, const unsigned int *m) {
//                 unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
//                 const unsigned int VECTOR_SIZE = 16;
//                 if (tid  < VECTOR_SIZE) {
//                     f[tid] = min(f[tid], f[tid] - m[tid]);
//                 }
//             }
//         ",
//         )
//         .unwrap();

//         self.load_ptx(
//             reduce_packed_base_field,
//             "PackedBaseFieldReduce",
//             &["reduce"],
//         )
//         .unwrap();
//     }

//     fn add(&self) {
//         // Add word by word. Each word is in the range [0, 2P].

//         // Apply min(c, c-P) to each word.
//         // When c in [P,2P], then c-P in [0,P] which is always less than [P,2P].
//         // When c in [0,P-1], then c-P in [2^32-P,2^32-1] which is always greater than
//         let add_packed_base_field = compile_ptx("
//             extern \"C\" __global__ void add(unsigned int *lhs, const unsigned int *rhs, const
// unsigned int *m) {                 unsigned int tid = threadIdx.x;
//                 const unsigned int VECTOR_SIZE = 16;
//                 if (tid  < VECTOR_SIZE) {
//                     lhs[tid] += rhs[tid];
//                     lhs[tid] = min(lhs[tid], lhs[tid] - m[tid]);
//                 }
//             }
//         ").unwrap();

//         self.load_ptx(add_packed_base_field, "PackedBaseFieldAdd", &["add"])
//             .unwrap();
//     }

//     fn neg(&self) {
//         let neg_packed_base_field = compile_ptx(
//             "
//             extern \"C\" __global__ void neg(unsigned int *f, const unsigned int *m) {
//                 unsigned int tid = threadIdx.x;
//                 const unsigned int VECTOR_SIZE = 16;
//                 if (tid  < VECTOR_SIZE) {
//                     f[tid] = m[tid] - f[tid];
//                 }
//             }
//         ",
//         )
//         .unwrap();

//         self.load_ptx(neg_packed_base_field, "PackedBaseFieldNeg", &["neg"])
//             .unwrap();
//     }

//     fn sub(&self) {
//         // Subtract word by word. Each word is in the range [-P, P].

//         // Apply min(c, c+P) to each word.
//         // When c in [0,P], then c+P in [P,2P] which is always greater than [0,P].
//         // When c in [2^32-P,2^32-1], then c+P in [0,P-1] which is always less than
//         // [2^32-P,2^32-1].
//         let sub_packed_base_field = compile_ptx(
//             "
//             extern \"C\" __global__ void sub(unsigned int *lhs, const unsigned int *rhs, const
// unsigned int *m) {                 unsigned int tid = threadIdx.x;
//                 const unsigned int VECTOR_SIZE = 16;
//                 if (tid  < VECTOR_SIZE) {
//                     lhs[tid] = lhs[tid] - rhs[tid];
//                     lhs[tid] = min(lhs[tid], lhs[tid] + m[tid]);
//                 }
//             }
//         ",
//         )
//         .unwrap();

//         self.load_ptx(sub_packed_base_field, "PackedBaseFieldSub", &["sub"])
//             .unwrap();
//     }

//     // TODO:: Optimize to save memory and fewer copies
//     fn mul(&self) {
//         let mul_packed_base_field = compile_ptx(
//             "
//             extern \"C\" __global__ void mul(unsigned int *a, unsigned int *b_dbl, unsigned int
// *out, const unsigned int *m) {                 unsigned int tid = threadIdx.x;
//                 const unsigned int VECTOR_SIZE = 16;
//                 __shared__ unsigned long long int a_e[8];
//                 __shared__ unsigned long long int a_o[8];
//                 __shared__ unsigned long long int b_dbl_e[8];
//                 __shared__ unsigned long long int b_dbl_o[8];
//                 __shared__ unsigned long long int prod_e_dbl[8];
//                 __shared__ unsigned long long int prod_o_dbl[8];
//                 __shared__ unsigned int prod_lows[16];
//                 __shared__ unsigned int prod_highs[16];

//                 if (tid < 16) {
//                     // Double b value
//                     b_dbl[tid] <<= 1;

//                     // if Odd
//                     if (tid % 2) {
//                         // Set up a word s.t. the lower half of each 64-bit word has the odd
// 32-bit words of                         // the first operand.
//                         a_o[tid/2] = static_cast<unsigned long long int>(a[tid]);

//                         b_dbl_o[tid/2] = static_cast<unsigned long long int>(b_dbl[tid]);
//                     }
//                     else {
//                         // Set up a word s.t. the lower half of each 64-bit word has the even
// 32-bit words of                         // the first operand. uint64_t
//                         a_e[tid/2] = static_cast<unsigned long long int>(a[tid]);
//                         b_dbl_e[tid/2] = static_cast<unsigned long long int>(b_dbl[tid]);
//                     }

//                     // To compute prod = a * b start by multiplying
//                     // a_e/o by b_dbl_e/o.
//                     if (tid < VECTOR_SIZE/2) {
//                         prod_e_dbl[tid] = a_e[tid] * b_dbl_e[tid];
//                         prod_o_dbl[tid] = a_o[tid] * b_dbl_o[tid];
//                     }

//                     // if Odd
//                     if (tid % 2) {
//                         // Interleave the even words of prod_e_dbl with the even words of
// prod_o_dbl:                         prod_lows[tid] = static_cast<unsigned int>(prod_o_dbl[tid /
// 2] & 0xFFFFFFFF);                         // Divide by 2
//                         prod_lows[tid] >>= 1;
//                         // Interleave the odd words of prod_e_dbl with the odd words of
// prod_o_dbl:                         prod_highs[tid] = static_cast<unsigned int>(prod_o_dbl[tid /
// 2] >> 32);                     }
//                     else {
//                         prod_lows[tid] = static_cast<unsigned int>(prod_e_dbl[tid / 2] &
// 0xFFFFFFFF);                         prod_lows[tid] >>= 1;
//                         prod_highs[tid] = static_cast<unsigned int>(prod_e_dbl[tid / 2] >> 32);

//                     }

//                     // add
//                     out[tid] = prod_lows[tid] + prod_highs[tid];
//                     out[tid] = min(out[tid], out[tid] - m[tid]);
//                 }
//             }
//         ",
//         )
//         .unwrap();

//         self.load_ptx(mul_packed_base_field, "PackedBaseFieldMul", &["mul"])
//             .unwrap();
//     }

//     fn mul_packed(&self) {
//         let ptx_src_mul_m31 = include_str!("m31.cu");
//         let ptx_mul_m31 = compile_ptx(ptx_src_mul_m31).unwrap();
//         self.load_ptx(ptx_mul_m31, "PackedBaseFieldMul_m31", &["mul_packed"])
//             .unwrap();
//     }

//     fn load(&self) {
//         let ptx_src_mul_m31 = include_str!("m31.cu");
//         let ptx_mul_m31 = compile_ptx(ptx_src_mul_m31).unwrap();
//         self.load_ptx(ptx_mul_m31, "PackedBaseFieldMul_m31", &["mul_packed"])
//             .unwrap();

//         self.reduce();
//         self.add();
//         self.neg();
//         self.sub();
//         self.mul();
//     }
// }

// /// Stores 16 M31 elements
// /// Each M31 element is unreduced in the range [0, P].
// #[derive(Debug)]
// #[repr(C)]
// pub struct PackedM31(U32_16);

// impl PackedBaseField {
//     /// Constructs a new instance with all vector elements set to `value`.
//     pub fn broadcast(M31(v): M31) -> Self {
//         Self(DEVICE.htod_copy(vec![v; PACKED_BASE_FIELD_SIZE]).unwrap())
//     }

//     pub fn from_array(v: [M31; PACKED_BASE_FIELD_SIZE]) -> PackedBaseField {
//         Self(DEVICE.htod_copy(v.map(|M31(v)| v).to_vec()).unwrap())
//     }

//     pub fn to_array(self) -> [M31; PACKED_BASE_FIELD_SIZE] {
//         let host = TryInto::<[u32; K_BLOCK_SIZE]>::try_into(
//             DEVICE.dtoh_sync_copy(&self.reduce().0).unwrap(),
//         )
//         .unwrap();
//         host.map(M31)
//     }

//     /// Reduces each word in the 512-bit register to the range `[0, P)`.
//     pub fn reduce(self) -> PackedBaseField {
//         let reduce_kernel = self
//             .0
//             .device()
//             .get_func("PackedBaseFieldReduce", "reduce")
//             .unwrap();
//         let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);

//         unsafe { reduce_kernel.launch(cfg, (&self.0, &*MODULUS)) }.unwrap();

//         self
//     }

//     // TODO:: Implement or Optimize as needed
//     // /// Interleaves two vectors.
//     // pub fn interleave(self, other: Self) -> (Self, Self) {
//     //     let (a, b) = self.0.interleave(other.0);
//     //     (Self(a), Self(b))
//     // }

//     // TODO:: Implement or Optimize as needed
//     // /// Deinterleaves two vectors.
//     // pub fn deinterleave(self, other: Self) -> (Self, Self) {
//     //     let (a, b) = self.0.deinterleave(other.0);
//     //     (Self(a), Self(b))
//     // }

//     /// Sums all the elements in the vector.
//     pub fn pointwise_sum(self) -> M31 {
//         self.to_array().into_iter().sum()
//     }

//     // TODO:: Implement or Optimize as needed
//     // /// Doubles each element in the vector.
//     // pub fn double(self) -> Self {
//     //     // TODO: Make more optimal.
//     //     self + self
//     // }

//     /// # Safety
//     ///
//     /// Vector elements must be in the range `[0, P]`.
//     pub unsafe fn from_cuda_slice_unchecked(v: CudaSlice<u32>) -> Self {
//         Self(v)
//     }

//     // TODO:: Implement or Optimize as needed
//     // /// # Safety
//     // ///
//     // /// Behavior is undefined if the pointer does not have the same alignment as
//     // /// [`PackedM31`]. The loaded `u32` values must be in the range `[0, P]`.
//     // pub unsafe fn load(mem_addr: *const u32) -> Self {
//     //     Self(ptr::read(mem_addr as *const u32x16))
//     // }

//     // TODO:: Implement or Optimize as needed
//     // /// # Safety
//     // ///
//     // /// Behavior is undefined if the pointer does not have the same alignment as
//     // /// [`PackedM31`]. The loaded `u32` values must be in the range `[0, P]`.
//     // pub unsafe fn load(mem_addr: *const u32) -> Self {
//     //     Self(ptr::read(mem_addr as *const u32x16))
//     // }

//     // TODO:: Implement or Optimize as needed
//     // /// # Safety
//     // ///
//     // /// Behavior is undefined if the pointer does not have the same alignment as
//     // /// [`PackedM31`].
//     // pub unsafe fn store(self, dst: *mut u32) {
//     //     ptr::write(dst as *mut u32x16, self.0)
//     // }

//     #[allow(dead_code)]
//     fn mul_m31(self, rhs: Self) -> Self {
//         let mul_kernel = self
//             .0
//             .device()
//             .get_func("PackedBaseFieldMul_m31", "mul_m31")
//             .unwrap();
//         let out = DEVICE.alloc_zeros::<u32>(PACKED_BASE_FIELD_SIZE).unwrap();

//         let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);
//         unsafe { mul_kernel.launch(cfg, (&self.0, &rhs.0, &out, &*MODULUS)) }.unwrap();

//         unsafe { PackedBaseField::from_cuda_slice_unchecked(out) }
//     }
// }

// // Clone is a device to device copy
// impl Clone for PackedBaseField {
//     fn clone(&self) -> Self {
//         let mut out = unsafe { self.0.device().alloc::<u32>(PACKED_BASE_FIELD_SIZE) }.unwrap();
//         self.0.device().dtod_copy(&self.0, &mut out).unwrap();
//         Self(out)
//     }
// }

// impl Add for PackedBaseField {
//     type Output = Self;

//     // Do we need to consume self? (if not we can change back add implementation to allocate new
//     // vector)
//     /// Adds two packed M31 elements, and reduces the result to the range `[0,P]`.
//     /// Each value is assumed to be in unreduced form, [0, P] including P.
//     #[inline(always)]
//     fn add(self, rhs: Self) -> Self::Output {
//         let add_kernel = self
//             .0
//             .device()
//             .get_func("PackedBaseFieldAdd", "add")
//             .unwrap();
//         let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);

//         unsafe { add_kernel.launch(cfg, (&self.0, &rhs.0, &*MODULUS)) }.unwrap();

//         self
//     }
// }

// // Adds in place
// impl AddAssign for PackedBaseField {
//     #[inline(always)]
//     fn add_assign(&mut self, rhs: Self) {
//         let add_kernel = self
//             .0
//             .device()
//             .get_func("PackedBaseFieldAdd", "add")
//             .unwrap();
//         let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);

//         unsafe { add_kernel.launch(cfg, (&self.0, &rhs.0, &*MODULUS)) }.unwrap();
//     }
// }

// impl Mul for PackedBaseField {
//     type Output = Self;

//     /// Computes the product of two packed M31 elements
//     /// Each value is assumed to be in unreduced form, [0, P] including P.
//     /// Returned values are in unreduced form, [0, P] including P.
//     #[inline(always)]
//     fn mul(self, rhs: Self) -> Self::Output {
//         let mul_kernel = self
//             .0
//             .device()
//             .get_func("PackedBaseFieldMul", "mul")
//             .unwrap();
//         let out = DEVICE.alloc_zeros::<u32>(PACKED_BASE_FIELD_SIZE).unwrap();

//         let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);
//         unsafe { mul_kernel.launch(cfg, (&self.0, &rhs.0, &out, &*MODULUS)) }.unwrap();

//         unsafe { PackedBaseField::from_cuda_slice_unchecked(out) }
//     }
// }

// impl MulAssign for PackedBaseField {
//     #[inline(always)]
//     fn mul_assign(&mut self, rhs: Self) {
//         let mul_kernel = self
//             .0
//             .device()
//             .get_func("PackedBaseFieldMul", "mul")
//             .unwrap();

//         let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);
//         unsafe { mul_kernel.launch(cfg, (&self.0, &rhs.0, &self.0, &*MODULUS)) }.unwrap();
//     }
// }

// impl Neg for PackedBaseField {
//     type Output = Self;

//     #[inline(always)]
//     fn neg(self) -> Self::Output {
//         let neg_kernel = self
//             .0
//             .device()
//             .get_func("PackedBaseFieldNeg", "neg")
//             .unwrap();
//         let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);

//         unsafe { neg_kernel.launch(cfg, (&self.0, &*MODULUS)) }.unwrap();

//         self
//     }
// }

// /// Subtracts two packed M31 elements, and reduces the result to the range `[0,P]`.
// /// Each value is assumed to be in unreduced form, [0, P] including P.
// impl Sub for PackedBaseField {
//     type Output = Self;

//     #[inline(always)]
//     fn sub(self, rhs: Self) -> Self::Output {
//         let sub_kernel = self
//             .0
//             .device()
//             .get_func("PackedBaseFieldSub", "sub")
//             .unwrap();
//         let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);

//         unsafe { sub_kernel.launch(cfg, (&self.0, &rhs.0, &*MODULUS)) }.unwrap();

//         self
//     }
// }

// // Subtract in place
// impl SubAssign for PackedBaseField {
//     #[inline(always)]
//     fn sub_assign(&mut self, rhs: Self) {
//         let sub_kernel = self
//             .0
//             .device()
//             .get_func("PackedBaseFieldSub", "sub")
//             .unwrap();
//         let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);

//         unsafe { sub_kernel.launch(cfg, (&self.0, &rhs.0, &*MODULUS)) }.unwrap();
//     }
// }

// impl Zero for PackedBaseField {
//     fn zero() -> Self {
//         Self(DEVICE.alloc_zeros::<u32>(PACKED_BASE_FIELD_SIZE).unwrap())
//     }

//     // TODO:: Optimize? It currently does a htod copy
//     fn is_zero(&self) -> bool {
//         self.clone().to_array().iter().all(M31::is_zero)
//     }
// }

// impl One for PackedBaseField {
//     fn one() -> Self {
//         Self(DEVICE.htod_copy([1; K_BLOCK_SIZE].to_vec()).unwrap())
//     }
// }

// // TODO:: Implement
// // impl FieldExpOps for PackedBaseField {
// //     fn inverse(&self) -> Self {
// //         assert!(!self.is_zero(), "0 has no inverse");
// //         pow2147483645(*self)
// //     }
// // }

// #[cfg(test)]
// mod tests {
//     use std::array;

//     use rand::rngs::SmallRng;
//     use rand::{Rng, SeedableRng};

//     use super::PackedM31;
//     #[test]
//     fn addition_works() {
//         let mut rng = SmallRng::seed_from_u64(0);
//         let lhs = rng.gen();
//         let rhs = rng.gen();
//         let packed_lhs = PackedM31::from_array(lhs);
//         let packed_rhs = PackedM31::from_array(rhs);

//         let res = packed_lhs + packed_rhs;

//         assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] + rhs[i]));
//     }

//     #[test]
//     fn subtraction_works() {
//         let mut rng = SmallRng::seed_from_u64(0);
//         let lhs = rng.gen();
//         let rhs = rng.gen();
//         let packed_lhs = PackedM31::from_array(lhs);
//         let packed_rhs = PackedM31::from_array(rhs);

//         let res = packed_lhs - packed_rhs;

//         assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] - rhs[i]));
//     }

//     #[test]
//     fn multiplication_works() {
//         let mut rng = SmallRng::seed_from_u64(0);
//         let lhs = rng.gen();
//         let rhs = rng.gen();
//         let packed_lhs = PackedM31::from_array(lhs);
//         let packed_rhs = PackedM31::from_array(rhs);

//         let res = packed_lhs * packed_rhs;

//         assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] * rhs[i]));
//     }

//     #[test]
//     fn negation_works() {
//         let mut rng = SmallRng::seed_from_u64(0);
//         let values = rng.gen();
//         let packed_values = PackedM31::from_array(values);

//         let res = -packed_values;

//         assert_eq!(res.to_array(), array::from_fn(|i| -values[i]));
//     }

//     #[test]
//     fn m31_multiplication_works() {
//         let mut rng = SmallRng::seed_from_u64(0);
//         let lhs = rng.gen();
//         let rhs = rng.gen();
//         let packed_lhs = PackedM31::from_array(lhs);
//         let packed_rhs = PackedM31::from_array(rhs);

//         let res = packed_lhs.mul_m31(packed_rhs);

//         assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] * rhs[i]));
//     }
// }
