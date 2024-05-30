// CUDA implementation of packed m31
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use num_traits::{One, Zero};

use super::{Device, DEVICE};
// use crate::core::fields::m31::pow2147483645;
use crate::core::fields::m31::M31;
// use crate::core::fields::FieldExpOps;
pub const K_BLOCK_SIZE: usize = 16;
pub const PACKED_BASE_FIELD_SIZE: usize = 1 << 4;
pub const M31_SIZE: usize = 1;
type Packed16BaseField = Packed16M31;
type M31Cuda = CudaSlice<u32>;

pub trait LoadBaseField {
    fn load(&self);
}

impl LoadBaseField for Device {
    fn load(&self) {
        let ptx_src_mul_m31 = include_str!("m31.cu");
        let ptx_mul_m31 = compile_ptx(ptx_src_mul_m31).unwrap();
        self.load_ptx(
            ptx_mul_m31,
            "base_field_functions",
            &[
                "mul",
                "reduce",
                "add",
                "sub",
                "neg",
                "mul_m31",
                "reduce_m31",
                "add_m31",
                "sub_m31",
                "neg_m31",
            ],
        )
        .unwrap();
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct BaseField(M31Cuda);

impl BaseField {
    /// Constructs a new instance with element set to `value`.
    pub fn broadcast(M31(v): M31) -> Self {
        Self(DEVICE.htod_copy(vec![v]).unwrap())
    }

    pub fn from_host(v: M31) -> Self {
        Self(DEVICE.htod_copy(vec![v.0]).unwrap())
    }

    pub fn to_device(self) -> M31 {
        let mut host = DEVICE.dtoh_sync_copy(&self.reduce().0).unwrap();
        M31(host.pop().unwrap())
    }

    /// Reduces each word in the 512-bit register to the range `[0, P)`.
    pub fn reduce(mut self) -> Self {
        let reduce_kernel = self
            .0
            .device()
            .get_func("base_field_functions", "reduce_m31")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(M31_SIZE as u32);

        unsafe { reduce_kernel.launch(cfg, (&mut self.0,)) }.unwrap();

        self
    }
}

impl Add for BaseField {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let kernel = self
            .0
            .device()
            .get_func("base_field_functions", "add_m31")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(M31_SIZE as u32);
        let mut out = DEVICE.alloc_zeros::<u32>(M31_SIZE).unwrap();

        unsafe { kernel.launch(cfg, (&self.0, &rhs.0, &mut out)) }.unwrap();
        Self(out)
    }
}

impl Neg for BaseField {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let kernel = self
            .0
            .device()
            .get_func("base_field_functions", "neg_m31")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(M31_SIZE as u32);

        unsafe { kernel.launch(cfg, (&self.0,)) }.unwrap();

        self
    }
}

impl Sub for BaseField {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let kernel = self
            .0
            .device()
            .get_func("base_field_functions", "sub_m31")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(M31_SIZE as u32);
        let mut out = DEVICE.alloc_zeros::<u32>(M31_SIZE).unwrap();

        unsafe { kernel.launch(cfg, (&self.0, &rhs.0, &mut out)) }.unwrap();
        Self(out)
    }
}

impl Mul for BaseField {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let kernel = self
            .0
            .device()
            .get_func("base_field_functions", "mul_m31")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(M31_SIZE as u32);
        let out = DEVICE.alloc_zeros::<u32>(M31_SIZE).unwrap();

        unsafe { kernel.launch(cfg, (&self.0, &rhs.0, &out)) }.unwrap();
        Self(out)
    }
}
impl MulAssign for BaseField {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        let kernel = self
            .0
            .device()
            .get_func("base_field_functions", "mul_m31")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(M31_SIZE as u32);

        unsafe { kernel.launch(cfg, (&self.0, &rhs.0, &self.0)) }.unwrap();
    }
}

// impl FieldExpOps for BaseField {
//     fn inverse(&self) -> Self {
//         assert!(!self.is_zero(), "0 has no inverse");
//         todo!()
//     }
// }

impl One for BaseField {
    fn one() -> Self {
        Self(DEVICE.htod_copy([1; M31_SIZE].to_vec()).unwrap())
    }
}

impl Zero for BaseField {
    fn zero() -> Self {
        Self(DEVICE.alloc_zeros::<u32>(M31_SIZE).unwrap())
    }

    fn is_zero(&self) -> bool {
        self.clone().to_device().is_zero()
    }
}

// Clone is a device to device copy
impl Clone for BaseField {
    fn clone(&self) -> Self {
        let mut out = unsafe { self.0.device().alloc::<u32>(M31_SIZE) }.unwrap();
        self.0.device().dtod_copy(&self.0, &mut out).unwrap();

        Self(out)
    }
}

//-------------------------------------------------------------------------------------

#[derive(Debug)]
#[repr(C)]
pub struct PackedBaseField(M31Cuda);

impl PackedBaseField {
    /// Constructs a new instance with all vector elements set to `value`.
    pub fn broadcast(M31(v): M31) -> Self {
        Self(DEVICE.htod_copy(vec![v; PACKED_BASE_FIELD_SIZE]).unwrap())
    }

    pub fn from_array<const N: usize>(v: [M31; N]) -> PackedBaseField {
        Self(DEVICE.htod_copy(v.map(|M31(v)| v).to_vec()).unwrap())
    }

    pub fn to_array(self) -> [M31; PACKED_BASE_FIELD_SIZE] {
        let host = TryInto::<[u32; K_BLOCK_SIZE]>::try_into(
            DEVICE.dtoh_sync_copy(&self.reduce().0).unwrap(),
        )
        .unwrap();
        host.map(M31)
    }

    /// Reduces each word in the 512-bit register to the range `[0, P)`.
    pub fn reduce(mut self) -> PackedBaseField {
        let reduce_kernel = self
            .0
            .device()
            .get_func("base_field_functions", "reduce")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);

        unsafe { reduce_kernel.launch(cfg, (&mut self.0, PACKED_BASE_FIELD_SIZE)) }.unwrap();

        self
    }
}

/// Stores 16 M31 elements
/// Each M31 element is unreduced in the range [0, P].
#[derive(Debug)]
#[repr(C)]
pub struct Packed16M31(CudaSlice<u32>);

impl Packed16BaseField {
    /// Constructs a new instance with all vector elements set to `value`.
    pub fn broadcast(M31(v): M31) -> Self {
        Self(DEVICE.htod_copy(vec![v; PACKED_BASE_FIELD_SIZE]).unwrap())
    }

    pub fn from_array(v: [M31; PACKED_BASE_FIELD_SIZE]) -> Packed16BaseField {
        Self(DEVICE.htod_copy(v.map(|M31(v)| v).to_vec()).unwrap())
    }

    pub fn to_array(self) -> [M31; PACKED_BASE_FIELD_SIZE] {
        let host = TryInto::<[u32; K_BLOCK_SIZE]>::try_into(
            DEVICE.dtoh_sync_copy(&self.reduce().0).unwrap(),
        )
        .unwrap();
        host.map(M31)
    }

    /// Reduces each word in the 512-bit register to the range `[0, P)`.
    pub fn reduce(mut self) -> Packed16BaseField {
        let reduce_kernel = self
            .0
            .device()
            .get_func("base_field_functions", "reduce")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);

        unsafe { reduce_kernel.launch(cfg, (&mut self.0, PACKED_BASE_FIELD_SIZE)) }.unwrap();

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

    #[allow(dead_code)]
    fn mul_m31(self, rhs: Self) -> Self {
        let mul_kernel = self
            .0
            .device()
            .get_func("base_field_functions", "mul")
            .unwrap();
        let out = DEVICE.alloc_zeros::<u32>(PACKED_BASE_FIELD_SIZE).unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);
        unsafe { mul_kernel.launch(cfg, (&self.0, &rhs.0, &out, PACKED_BASE_FIELD_SIZE)) }.unwrap();

        Self(out)
    }
}

// Clone is a device to device copy
impl Clone for Packed16BaseField {
    fn clone(&self) -> Self {
        let mut out = unsafe { self.0.device().alloc::<u32>(PACKED_BASE_FIELD_SIZE) }.unwrap();
        self.0.device().dtod_copy(&self.0, &mut out).unwrap();

        Self(out)
    }
}

impl Add for Packed16BaseField {
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
            .get_func("base_field_functions", "add")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);
        let mut out = DEVICE.alloc_zeros::<u32>(PACKED_BASE_FIELD_SIZE).unwrap();

        unsafe { add_kernel.launch(cfg, (&self.0, &rhs.0, &mut out, PACKED_BASE_FIELD_SIZE)) }
            .unwrap();
        Self(out)
    }
}

// Adds in place
impl AddAssign for Packed16BaseField {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        let add_kernel = self
            .0
            .device()
            .get_func("base_field_functions", "add")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);

        unsafe { add_kernel.launch(cfg, (&self.0, &rhs.0, &self.0, PACKED_BASE_FIELD_SIZE)) }
            .unwrap();
    }
}

impl Mul for Packed16BaseField {
    type Output = Self;

    /// Computes the product of two packed M31 elements
    /// Each value is assumed to be in unreduced form, [0, P] including P.
    /// Returned values are in unreduced form, [0, P] including P.
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let mul_kernel = self
            .0
            .device()
            .get_func("base_field_functions", "mul")
            .unwrap();
        let out = DEVICE.alloc_zeros::<u32>(PACKED_BASE_FIELD_SIZE).unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);
        unsafe { mul_kernel.launch(cfg, (&self.0, &rhs.0, &out, PACKED_BASE_FIELD_SIZE)) }.unwrap();

        Self(out)
    }
}

impl MulAssign for Packed16BaseField {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        let mul_kernel = self
            .0
            .device()
            .get_func("base_field_functions", "mul")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);
        unsafe { mul_kernel.launch(cfg, (&self.0, &rhs.0, &self.0, PACKED_BASE_FIELD_SIZE)) }
            .unwrap();
    }
}

impl Neg for Packed16BaseField {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        let neg_kernel = self
            .0
            .device()
            .get_func("base_field_functions", "neg")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);

        unsafe { neg_kernel.launch(cfg, (&self.0, PACKED_BASE_FIELD_SIZE)) }.unwrap();

        self
    }
}

/// Subtracts two packed M31 elements, and reduces the result to the range `[0,P]`.
/// Each value is assumed to be in unreduced form, [0, P] including P.
impl Sub for Packed16BaseField {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let sub_kernel = self
            .0
            .device()
            .get_func("base_field_functions", "sub")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);

        unsafe { sub_kernel.launch(cfg, (&self.0, &rhs.0, &self.0, PACKED_BASE_FIELD_SIZE)) }
            .unwrap();

        self
    }
}

// Subtract in place
impl SubAssign for Packed16BaseField {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        let sub_kernel = self
            .0
            .device()
            .get_func("base_field_functions", "sub")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(PACKED_BASE_FIELD_SIZE as u32);

        unsafe { sub_kernel.launch(cfg, (&self.0, &rhs.0, &self.0, PACKED_BASE_FIELD_SIZE)) }
            .unwrap();
    }
}

impl Zero for Packed16BaseField {
    fn zero() -> Self {
        Self(DEVICE.alloc_zeros::<u32>(PACKED_BASE_FIELD_SIZE).unwrap())
    }

    // TODO:: Optimize? It currently does a htod copy
    fn is_zero(&self) -> bool {
        self.clone().to_array().iter().all(M31::is_zero)
    }
}

impl One for Packed16BaseField {
    fn one() -> Self {
        Self(DEVICE.htod_copy([1; K_BLOCK_SIZE].to_vec()).unwrap())
    }
}

// TODO:: Implement
// impl FieldExpOps for Packed16BaseField {
//     fn inverse(&self) -> Self {
//         assert!(!self.is_zero(), "0 has no inverse");
//         pow2147483645(*self)
//     }
// }

#[cfg(test)]
mod tests {
    use std::array;

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::Packed16M31;
    use crate::core::backend::gpu::m31::BaseField;

    #[test]
    fn test_addition_m31() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs: crate::core::fields::m31::M31 = rng.gen();
        let rhs: crate::core::fields::m31::M31 = rng.gen();
        let packed_lhs = BaseField::from_host(lhs);
        let packed_rhs = BaseField::from_host(rhs);
        let res = packed_lhs + packed_rhs;

        assert_eq!(res.to_device(), lhs + rhs);
    }

    #[test]
    fn test_subtraction_m31() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs: crate::core::fields::m31::M31 = rng.gen();
        let rhs: crate::core::fields::m31::M31 = rng.gen();
        let packed_lhs = BaseField::from_host(lhs);
        let packed_rhs = BaseField::from_host(rhs);
        let res = packed_lhs - packed_rhs;

        assert_eq!(res.to_device(), lhs - rhs);
    }

    #[test]
    fn test_multiplication_m31() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs: crate::core::fields::m31::M31 = rng.gen();
        let rhs: crate::core::fields::m31::M31 = rng.gen();
        let packed_lhs = BaseField::from_host(lhs);
        let packed_rhs = BaseField::from_host(rhs);
        let res = packed_lhs * packed_rhs;

        assert_eq!(res.to_device(), lhs * rhs);
    }

    #[test]
    fn test_negation_m31() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs: crate::core::fields::m31::M31 = rng.gen();
        let packed_lhs = BaseField::from_host(lhs);
        let res = -packed_lhs;

        assert_eq!(res.to_device(), -lhs);
    }

    #[test]
    fn test_addition_16() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();
        let packed_lhs = Packed16M31::from_array(lhs);
        let packed_rhs = Packed16M31::from_array(rhs);

        let res = packed_lhs + packed_rhs;

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] + rhs[i]));
    }

    #[test]
    fn test_subtraction_16() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();
        let packed_lhs = Packed16M31::from_array(lhs);
        let packed_rhs = Packed16M31::from_array(rhs);

        let res = packed_lhs - packed_rhs;

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] - rhs[i]));
    }

    #[test]
    fn test_multiplication_16() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs = rng.gen();
        let rhs = rng.gen();
        let packed_lhs = Packed16M31::from_array(lhs);
        let packed_rhs = Packed16M31::from_array(rhs);

        let res = packed_lhs * packed_rhs;

        assert_eq!(res.to_array(), array::from_fn(|i| lhs[i] * rhs[i]));
    }

    #[test]
    fn test_negation_16() {
        let mut rng = SmallRng::seed_from_u64(0);
        let values = rng.gen();
        let packed_values = Packed16M31::from_array(values);

        let res = -packed_values;

        assert_eq!(res.to_array(), array::from_fn(|i| -values[i]));
    }
}
