use std::env;
// CUDA implementation of arbitrary size packed m31
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::path::PathBuf;

use cudarc::driver::{CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use itertools::Itertools;
use num_traits::{One, Zero};

use super::{Device, DEVICE};
#[allow(unused_imports)]
use crate::core::fields::m31::{pow2147483645, M31};
#[allow(unused_imports)]
use crate::core::fields::FieldExpOps;

type GpuM31 = CudaSlice<u32>;
type PackedM31 = PackedBaseField;

pub trait LoadBaseField {
    fn load(&self);
}

impl LoadBaseField for Device {
    fn load(&self) {
        let ptx_dir = PathBuf::from(env::var("OUT_DIR").unwrap() + "/m31.ptx");

        let ptx = Ptx::from_file(ptx_dir);
        self.load_ptx(
            ptx,
            "base_field_functions",
            &["mul", "reduce", "add", "sub", "neg"],
        )
        .unwrap();
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct PackedBaseField(pub GpuM31);

impl PackedM31 {
    /// Constructs a new instance with all vector elements set to `value`.
    pub fn broadcast(M31(v): M31, size: Option<usize>) -> Self {
        let size = size.unwrap_or(1024);
        Self(DEVICE.htod_copy(vec![v; size]).unwrap())
    }

    pub fn from_fixed_array<const N: usize>(v: [M31; N]) -> PackedM31 {
        Self(DEVICE.htod_copy(v.map(|M31(v)| v).to_vec()).unwrap())
    }

    pub fn from_array(v: Vec<M31>) -> PackedM31 {
        Self(
            DEVICE
                .htod_copy(v.into_iter().map(|M31(v)| v).collect())
                .unwrap(),
        )
    }

    pub fn to_fixed_array<const N: usize>(self) -> [M31; N] {
        TryInto::<[u32; N]>::try_into(DEVICE.dtoh_sync_copy(&self.reduce().0).unwrap())
            .unwrap()
            .map(M31)
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
    pub fn reduce(self) -> PackedM31 {
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
        DEVICE.synchronize().unwrap();
    }

    pub fn mul_assign_ref(&self, rhs: &Self) {
        let kernel = self
            .0
            .device()
            .get_func("base_field_functions", "mul")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);
        unsafe { kernel.launch(cfg, (&self.0, &rhs.0, &self.0, self.0.len())) }.unwrap();
        DEVICE.synchronize().unwrap();
    }

    pub fn sub_assign_ref(&self, rhs: &Self) {
        let kernel = self
            .0
            .device()
            .get_func("base_field_functions", "sub")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);
        unsafe { kernel.launch(cfg, (&self.0, &rhs.0, &self.0, self.0.len())) }.unwrap();
        DEVICE.synchronize().unwrap();
    }
}

// Clone is a device to device copy
impl Clone for PackedM31 {
    fn clone(&self) -> Self {
        let mut out = unsafe { self.0.device().alloc::<u32>(self.0.len()) }.unwrap();
        self.0.device().dtod_copy(&self.0, &mut out).unwrap();

        Self(out)
    }
}

impl Add for PackedM31 {
    type Output = Self;

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
        DEVICE.synchronize().unwrap();

        Self(out)
    }
}

// Adds in place
impl AddAssign for PackedM31 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        let add_kernel = self
            .0
            .device()
            .get_func("base_field_functions", "add")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);

        unsafe { add_kernel.launch(cfg, (&self.0, &rhs.0, &self.0, self.0.len())) }.unwrap();
        DEVICE.synchronize().unwrap();
    }
}

impl Mul for PackedM31 {
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
        DEVICE.synchronize().unwrap();

        Self(out)
    }
}

impl MulAssign for PackedM31 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        let mul_kernel = self
            .0
            .device()
            .get_func("base_field_functions", "mul")
            .unwrap();

        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);
        unsafe { mul_kernel.launch(cfg, (&self.0, &rhs.0, &self.0, self.0.len())) }.unwrap();
        DEVICE.synchronize().unwrap();
    }
}

impl Neg for PackedM31 {
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
        DEVICE.synchronize().unwrap();

        self
    }
}

/// Subtracts two packed M31 elements, and reduces the result to the range `[0,P]`.
/// Each value is assumed to be in unreduced form, [0, P] including P.
impl Sub for PackedM31 {
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
        DEVICE.synchronize().unwrap();

        self
    }
}

// Subtract in place
impl SubAssign for PackedM31 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        let sub_kernel = self
            .0
            .device()
            .get_func("base_field_functions", "sub")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);

        unsafe { sub_kernel.launch(cfg, (&self.0, &rhs.0, &self.0, self.0.len())) }.unwrap();
        DEVICE.synchronize().unwrap();
    }
}

// Returns a single zero
impl Zero for PackedM31 {
    fn zero() -> Self {
        Self(DEVICE.alloc_zeros::<u32>(1).unwrap())
    }

    // TODO:: Optimize? It currently does a htod copy
    fn is_zero(&self) -> bool {
        self.clone().to_vec().iter().all(M31::is_zero)
    }
}

impl One for PackedM31 {
    fn one() -> Self {
        Self(DEVICE.htod_copy([1; 1].to_vec()).unwrap())
    }
}

// // // impl<T: DeviceRepr> Copy for CudaSlice<T> {}
// impl<'a> Copy for TestPackedM31<'a> {}
// pub struct TestPackedM31<'a> {
//     field: std::sync::Arc<&'a CudaSlice<u32>>,
// }
// // TODO:: Implement
// impl FieldExpOps for TestPackedM31 {
//     fn inverse(&self) -> Self {
//         assert!(!self.is_zero(), "0 has no inverse");
//         pow2147483645(*self)
//     }
// }

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::PackedM31;
    use crate::core::fields::m31::M31;

    const SIZE: usize = 1 << 4;

    fn setup(size: usize) -> (Vec<M31>, Vec<M31>) {
        let mut rng: SmallRng = SmallRng::seed_from_u64(0);
        (0..size)
            .into_iter()
            .map(|_| (rng.gen::<M31>(), rng.gen::<M31>()))
            .unzip()
    }

    #[test]
    fn test_addition() {
        let (lhs, rhs) = setup(SIZE);
        let mut packed_lhs = PackedM31::from_array(lhs.clone());
        let packed_rhs = PackedM31::from_array(rhs.clone());

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
        let (lhs, rhs) = setup(SIZE);
        let mut packed_lhs = PackedM31::from_array(lhs.clone());
        let packed_rhs = PackedM31::from_array(rhs.clone());

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
        let (lhs, rhs) = setup(SIZE);
        let mut packed_lhs = PackedM31::from_array(lhs.clone());
        let packed_rhs = PackedM31::from_array(rhs.clone());

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
        let (lhs, _) = setup(SIZE);
        let packed_values = PackedM31::from_array(lhs.clone());

        let res = -packed_values;

        assert_eq!(
            res.to_array(),
            lhs.iter().map(|&l| -M31(l.0)).collect::<Vec<M31>>()
        )
    }

    #[test]
    fn test_addition_ref() {
        let (lhs, rhs) = setup(SIZE);

        let packed_lhs = PackedM31::from_array(lhs.clone());
        let packed_rhs = PackedM31::from_array(rhs.clone());

        packed_lhs.add_assign_ref(&packed_rhs);

        assert_eq!(
            packed_lhs.to_array(),
            lhs.iter()
                .zip(rhs.iter())
                .map(|(&l, &r)| l + r)
                .collect::<Vec<M31>>()
        );
    }
}
