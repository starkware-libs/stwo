// CUDA implementation of packed m31
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use cudarc::driver::{CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use itertools::Itertools;
use num_traits::{One, Zero};

use super::{Device, DEVICE};
// use crate::core::fields::m31::pow2147483645;
use crate::core::fields::m31::M31;
// use crate::core::fields::FieldExpOps;
pub const K_BLOCK_SIZE: usize = 16;
pub const PACKED_BASE_FIELD_SIZE: usize = 1 << 4;
pub const M31_SIZE: usize = 1;
type GpuM31 = CudaSlice<u32>;

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
                "test_add_m31",
            ],
        )
        .unwrap();
    }
}

pub struct TestBaseField(pub M31);

impl TestBaseField {
    /// Constructs a new instance with element set to `value`.
    pub fn broadcast(M31(v): M31) -> Self {
        Self(M31(v))
    }

    pub fn from_host(v: M31) -> Self {
        Self(v)
    }

    // pub fn to_device(self) -> M31 {
    //     let mut host = DEVICE.htod_copy(vec![self.0 .0]).unwrap();
    //     M31(host.pop().unwrap())
    // }

    /// Reduces each word in the 512-bit register to the range `[0, P)`.
    pub fn reduce(self) -> Self {
        let reduce_kernel = DEVICE
            .get_func("base_field_functions", "reduce_m31")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(M31_SIZE as u32);
        let out = DEVICE.htod_copy(vec![self.0 .0]).unwrap();
        unsafe { reduce_kernel.launch(cfg, (&out,)) }.unwrap();

        self
    }

    pub fn add_assign_ref(&mut self, rhs: &mut Self) {
        let kernel = DEVICE
            .get_func("base_field_functions", "test_add_m31")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(M31_SIZE as u32);
        let out = DEVICE.alloc_zeros::<u32>(M31_SIZE).unwrap();
        unsafe { kernel.launch(cfg, (self.0 .0, rhs.0 .0, &out)) }.unwrap();
        let mut out = DEVICE.dtoh_sync_copy(&out).unwrap();
        self.0 = M31(out.pop().unwrap());
    }
}

// ----------------------------------------------------------
#[derive(Debug)]
#[repr(C)]
pub struct BaseField(GpuM31);

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

    pub fn add_assign_ref_mut(&mut self, rhs: &mut Self) {
        let kernel = self
            .0
            .device()
            .get_func("base_field_functions", "add_m31")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(M31_SIZE as u32);

        unsafe { kernel.launch(cfg, (&self.0, &rhs.0, &self.0)) }.unwrap();
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

impl AddAssign for BaseField {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        let kernel = self
            .0
            .device()
            .get_func("base_field_functions", "add_m31")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(M31_SIZE as u32);

        unsafe { kernel.launch(cfg, (&self.0, &rhs.0, &self.0)) }.unwrap();
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

impl SubAssign for BaseField {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        let kernel = self
            .0
            .device()
            .get_func("base_field_functions", "sub_m31")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(M31_SIZE as u32);

        unsafe { kernel.launch(cfg, (&self.0, &rhs.0, &self.0)) }.unwrap();
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

//------------------------------------------------------------------------------------- TODO:: Finish

#[derive(Debug)]
#[repr(C)]
pub struct PackedBaseField(GpuM31);

impl PackedBaseField {
    /// Constructs a new instance with all vector elements set to `value`.
    /// Size must be less thna 1024 for now
    pub fn broadcast(M31(v): M31, size: Option<usize>) -> Self {
        let size = size.unwrap_or(1024);
        Self(DEVICE.htod_copy(vec![v; size]).unwrap())
    }

    pub fn from_fixed_array<const N: usize>(v: [M31; N]) -> PackedBaseField {
        Self(DEVICE.htod_copy(v.map(|M31(v)| v).to_vec()).unwrap())
    }

    pub fn from_array(v: Vec<M31>) -> PackedBaseField {
        Self(
            DEVICE
                .htod_copy(v.into_iter().map(|M31(v)| v).collect())
                .unwrap(),
        )
    }

    pub fn to_fixed_array<const N: usize>(self) -> [M31; N] {
        let host = TryInto::<[u32; N]>::try_into(DEVICE.dtoh_sync_copy(&self.reduce().0).unwrap())
            .unwrap();
        host.map(M31)
    }

    pub fn to_array(self) -> Vec<M31> {
        DEVICE
            .dtoh_sync_copy(&self.reduce().0)
            .unwrap()
            .iter()
            .map(|&v| M31(v))
            .collect()
    }
    /// Reduces each word in the 512-bit register to the range `[0, P)`.
    pub fn reduce(self) -> PackedBaseField {
        let reduce_kernel = self
            .0
            .device()
            .get_func("base_field_functions", "reduce")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);
        unsafe { reduce_kernel.launch(cfg, (&self.0, self.0.len())) }.unwrap();

        self
    }
    pub fn to_vec(self) -> Vec<M31> {
        let host = DEVICE.dtoh_sync_copy(&self.reduce().0).unwrap();
        host.iter().map(|&x| M31(x)).collect_vec()
    }

    /// # Safety
    ///
    /// Vector elements must be in the range `[0, P]`.
    pub unsafe fn from_cuda_slice_unchecked(v: CudaSlice<u32>) -> Self {
        Self(v)
    }

    /// Adding by reference since CudaSlice cannot implement Copy and Add operator from standard
    /// crates consume the object.
    pub fn add_assign_ref(&self, rhs: &Self) {
        let kernel = self
            .0
            .device()
            .get_func("base_field_functions", "add")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);
        unsafe { kernel.launch(cfg, (&self.0, &rhs.0, &self.0, self.0.len())) }.unwrap();
    }

    pub fn mul_assign_ref(&self, rhs: &Self) {
        let kernel = self
            .0
            .device()
            .get_func("base_field_functions", "mul")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);
        unsafe { kernel.launch(cfg, (&self.0, &rhs.0, &self.0, self.0.len())) }.unwrap();
    }

    pub fn sub_assign_ref(&self, rhs: &Self) {
        let kernel = self
            .0
            .device()
            .get_func("base_field_functions", "sub")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);
        unsafe { kernel.launch(cfg, (&self.0, &rhs.0, &self.0, self.0.len())) }.unwrap();
    }
}

// Clone is a device to device copy
impl Clone for PackedBaseField {
    fn clone(&self) -> Self {
        let mut out = unsafe { self.0.device().alloc::<u32>(self.0.len()) }.unwrap();
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
            .get_func("base_field_functions", "add")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);
        let mut out = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();

        unsafe { add_kernel.launch(cfg, (&self.0, &rhs.0, &mut out, self.0.len())) }.unwrap();
        Self(out)
    }
}

// Adds in place
impl AddAssign for PackedBaseField {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        let add_kernel = self
            .0
            .device()
            .get_func("base_field_functions", "add")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);

        unsafe { add_kernel.launch(cfg, (&self.0, &rhs.0, &self.0, self.0.len())) }.unwrap();
    }
}

impl Mul for PackedBaseField {
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

        let out = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);
        unsafe { mul_kernel.launch(cfg, (&self.0, &rhs.0, &out, self.0.len())) }.unwrap();

        Self(out)
    }
}

impl MulAssign for PackedBaseField {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        let mul_kernel = self
            .0
            .device()
            .get_func("base_field_functions", "mul")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);
        unsafe { mul_kernel.launch(cfg, (&self.0, &rhs.0, &self.0, self.0.len())) }.unwrap();
    }
}

impl Neg for PackedBaseField {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        let neg_kernel = self
            .0
            .device()
            .get_func("base_field_functions", "neg")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);

        unsafe { neg_kernel.launch(cfg, (&self.0, self.0.len())) }.unwrap();

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
            .get_func("base_field_functions", "sub")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);

        unsafe { sub_kernel.launch(cfg, (&self.0, &rhs.0, &self.0, self.0.len())) }.unwrap();

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
            .get_func("base_field_functions", "sub")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);

        unsafe { sub_kernel.launch(cfg, (&self.0, &rhs.0, &self.0, self.0.len())) }.unwrap();
    }
}

impl Zero for PackedBaseField {
    fn zero() -> Self {
        Self(DEVICE.alloc_zeros::<u32>(1024).unwrap())
    }

    // TODO:: Optimize? It currently does a htod copy
    fn is_zero(&self) -> bool {
        self.clone().to_vec().iter().all(M31::is_zero)
    }
}

impl One for PackedBaseField {
    fn one() -> Self {
        Self(DEVICE.htod_copy([1; 1024].to_vec()).unwrap())
    }
}

// TODO:: Implement
// impl FieldExpOps for PackedBaseField {
//     fn inverse(&self) -> Self {
//         assert!(!self.is_zero(), "0 has no inverse");
//         pow2147483645(*self)
//     }
// }

#[cfg(test)]
mod tests {

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::PackedBaseField;
    use crate::core::backend::gpu::m31::{BaseField, TestBaseField};
    use crate::core::fields::m31::M31;

    #[test]
    fn test_host_addition_m31() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs: crate::core::fields::m31::M31 = rng.gen();
        let rhs: crate::core::fields::m31::M31 = rng.gen();
        let mut packed_lhs = TestBaseField::from_host(lhs);
        let mut packed_rhs = TestBaseField::from_host(rhs);
        packed_lhs.add_assign_ref(&mut packed_rhs);

        assert_eq!(packed_lhs.0, lhs + rhs);
    }

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
    fn test_addition_m31_ref_mut() {
        let mut rng = SmallRng::seed_from_u64(0);
        let lhs: crate::core::fields::m31::M31 = rng.gen();
        let rhs: crate::core::fields::m31::M31 = rng.gen();
        let mut packed_lhs = BaseField::from_host(lhs);
        let mut packed_rhs = BaseField::from_host(rhs);
        packed_lhs.add_assign_ref_mut(&mut packed_rhs);

        assert_eq!(packed_lhs.to_device(), lhs + rhs);
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
    fn test_addition() {
        let (lhs, rhs) = setup(100000);
        let mut packed_lhs = PackedBaseField::from_array(lhs.clone());
        let packed_rhs = PackedBaseField::from_array(rhs.clone());

        packed_lhs += packed_rhs;

        assert_eq!(
            packed_lhs.to_array(),
            lhs.iter()
                .zip(rhs.iter())
                .map(|(&l, &r)| l + r)
                .collect::<Vec<M31>>()
        );
    }

    #[test]
    fn test_subtraction() {
        let (lhs, rhs) = setup(100000);
        let mut packed_lhs = PackedBaseField::from_array(lhs.clone());
        let packed_rhs = PackedBaseField::from_array(rhs.clone());

        packed_lhs -= packed_rhs;

        assert_eq!(
            packed_lhs.to_array(),
            lhs.iter()
                .zip(rhs.iter())
                .map(|(&l, &r)| l - r)
                .collect::<Vec<M31>>()
        );
    }

    #[test]
    fn test_multiplication() {
        let (lhs, rhs) = setup(100000);
        let mut packed_lhs = PackedBaseField::from_array(lhs.clone());
        let packed_rhs = PackedBaseField::from_array(rhs.clone());

        packed_lhs *= packed_rhs;

        assert_eq!(
            packed_lhs.to_array(),
            lhs.iter()
                .zip(rhs.iter())
                .map(|(&l, &r)| l * r)
                .collect::<Vec<M31>>()
        );
    }

    #[test]
    fn test_negation() {
        let (lhs, _) = setup(100000);
        let packed_values = PackedBaseField::from_array(lhs.clone());

        let res = -packed_values;

        assert_eq!(
            res.to_array(),
            lhs.iter().map(|&l| -M31(l.0)).collect::<Vec<M31>>()
        )
    }

    #[test]
    fn test_addition_ref_mut() {
        let (lhs, rhs) = setup(100000);

        let packed_lhs = PackedBaseField::from_array(lhs.clone());
        let packed_rhs = PackedBaseField::from_array(rhs.clone());

        packed_lhs.add_assign_ref(&packed_rhs);

        assert_eq!(
            packed_lhs.to_array(),
            lhs.iter()
                .zip(rhs.iter())
                .map(|(&l, &r)| l + r)
                .collect::<Vec<M31>>()
        );
    }

    fn setup(size: usize) -> (Vec<M31>, Vec<M31>) {
        let mut rng: SmallRng = SmallRng::seed_from_u64(0);
        std::iter::repeat_with(|| (rng.gen::<M31>(), rng.gen::<M31>()))
            .take(size)
            .unzip()
    }
}
