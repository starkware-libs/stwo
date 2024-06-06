// CUDA implementation of arbitrary size packed m31
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use cudarc::driver::{CudaSlice, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use itertools::Itertools;
use num_traits::{One, Zero};

use super::{Device, DEVICE};
use crate::core::backend::gpu::m31::PackedBaseField as PackedM31;
use crate::core::fields::cm31::CM31;
#[allow(unused_imports)]
use crate::core::fields::m31::{pow2147483645, M31};
use crate::core::fields::qm31::QM31;
#[allow(unused_imports)]
use crate::core::fields::FieldExpOps;

pub trait LoadSecureBaseField {
    fn load(&self);
}

impl LoadSecureBaseField for Device {
    fn load(&self) {
        let ptx_src_mul_m31 = include_str!("qm31.cu");
        let ptx_mul_m31 = compile_ptx(ptx_src_mul_m31).unwrap();
        self.load_ptx(
            ptx_mul_m31,
            "base_field_functions",
            &["mul", "reduce", "add", "sub", "neg"],
        )
        .unwrap();
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct PackedQM31(pub [PackedM31; 4]);

impl PackedQM31 {
    /// Constructs a new instance with all vector elements set to `value`.
    pub fn broadcast(M31(v): M31, size: Option<usize>) -> Self {
        let size = size.unwrap_or(1024);
        Self([
            PackedM31(DEVICE.htod_copy(vec![v; size]).unwrap()),
            PackedM31(DEVICE.htod_copy(vec![v; size]).unwrap()),
            PackedM31(DEVICE.htod_copy(vec![v; size]).unwrap()),
            PackedM31(DEVICE.htod_copy(vec![v; size]).unwrap()),
        ])
    }

    // pub fn from_fixed_array<const N: usize>(v: [QM31; N]) -> PackedQM31 {
    //     Self(DEVICE.htod_copy(v.map(|M31(v)| v).to_vec()).unwrap())
    // }

    pub fn from_array(v: Vec<QM31>) -> PackedQM31 {
        let cm31_a: Vec<CM31> = v.iter().map(|v| v.0).collect();
        let cm31_b: Vec<CM31> = v.iter().map(|v| v.1).collect();

        let a: Vec<M31> = cm31_a.iter().map(|v| v.0).collect();
        let b: Vec<M31> = cm31_a.iter().map(|v| v.1).collect();
        let c: Vec<M31> = cm31_b.iter().map(|v| v.0).collect();
        let d: Vec<M31> = cm31_b.iter().map(|v| v.1).collect();

        let a = PackedM31::from_array(a);
        let b = PackedM31::from_array(b);
        let c = PackedM31::from_array(c);
        let d = PackedM31::from_array(d);

        Self([a, b, c, d])
    }

    // pub fn to_fixed_array<const N: usize>(self) -> [QM31; N] {
    //     let a = TryInto::<[u32; N]>::try_into(DEVICE.dtoh_sync_copy(&self.0[0].0).unwrap())
    //         .unwrap()
    //         .map(M31);
    //     let b = TryInto::<[u32; N]>::try_into(DEVICE.dtoh_sync_copy(&self.0[1].0).unwrap())
    //         .unwrap()
    //         .map(M31);
    //     let c = TryInto::<[u32; N]>::try_into(DEVICE.dtoh_sync_copy(&self.0[2].0).unwrap())
    //         .unwrap()
    //         .map(M31);
    //     let d = TryInto::<[u32; N]>::try_into(DEVICE.dtoh_sync_copy(&self.0[3].0).unwrap())
    //         .unwrap()
    //         .map(M31);
    //     PackedQM31([a, b, c, d])
    // }

    pub fn to_array(self) -> Vec<QM31> {
        let a = DEVICE
            .dtoh_sync_copy(&self.0[0].0)
            .unwrap()
            .into_iter()
            .map(M31)
            .collect_vec();
        let b = DEVICE
            .dtoh_sync_copy(&self.0[1].0)
            .unwrap()
            .into_iter()
            .map(M31)
            .collect_vec();
        let c = DEVICE
            .dtoh_sync_copy(&self.0[2].0)
            .unwrap()
            .into_iter()
            .map(M31)
            .collect_vec();
        let d = DEVICE
            .dtoh_sync_copy(&self.0[3].0)
            .unwrap()
            .into_iter()
            .map(M31)
            .collect_vec();

        let qm31_a = QM31(CM31(a[0], b[0]), CM31(c[0], d[0]));

        a.into_iter()
            .zip(b.into_iter())
            .zip(c.into_iter())
            .zip(d.into_iter())
            .map(|(((a, b), c), d)| QM31(CM31(a, b), CM31(c, d)))
            .collect_vec()
    }
}

// Clone is a device to device copy
impl Clone for PackedQM31 {
    fn clone(&self) -> Self {
        let device = self.0[0].0.device();
        let mut a = unsafe { device.alloc::<u32>(self.0.len()) }.unwrap();
        let mut b = unsafe { device.alloc::<u32>(self.0.len()) }.unwrap();
        let mut c = unsafe { device.alloc::<u32>(self.0.len()) }.unwrap();
        let mut d = unsafe { device.alloc::<u32>(self.0.len()) }.unwrap();

        device.dtod_copy(&self.0[0].0, &mut a).unwrap();
        device.dtod_copy(&self.0[1].0, &mut b).unwrap();
        device.dtod_copy(&self.0[2].0, &mut c).unwrap();
        device.dtod_copy(&self.0[3].0, &mut d).unwrap();

        Self([PackedM31(a), PackedM31(b), PackedM31(c), PackedM31(d)])
    }
}

impl Add for PackedQM31 {
    type Output = Self;

    /// Adds two packed M31 elements, and reduces the result to the range `[0,P]`.
    /// Each value is assumed to be in unreduced form, [0, P] including P.
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let add_kernel = self.0[0]
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
impl AddAssign for PackedQM31 {
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

impl Mul for PackedQM31 {
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

impl MulAssign for PackedQM31 {
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

impl Neg for PackedQM31 {
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
impl Sub for PackedQM31 {
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
impl SubAssign for PackedQM31 {
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
impl Zero for PackedQM31 {
    fn zero() -> Self {
        Self(DEVICE.alloc_zeros::<u32>(1).unwrap())
    }

    // TODO:: Optimize? It currently does a htod copy
    fn is_zero(&self) -> bool {
        self.clone().to_vec().iter().all(M31::is_zero)
    }
}

impl One for PackedQM31 {
    fn one() -> Self {
        Self(DEVICE.htod_copy([1; 1].to_vec()).unwrap())
    }
}

// // TODO:: Implement
// impl FieldExpOps for TestPackedQM31 {
//     fn inverse(&self) -> Self {
//         assert!(!self.is_zero(), "0 has no inverse");
//         pow2147483645(*self)
//     }
// }

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::PackedQM31;
    use crate::core::fields::m31::M31;

    const SIZE: usize = 1 << 24;

    fn setup(size: usize) -> (Vec<M31>, Vec<M31>) {
        let mut rng: SmallRng = SmallRng::seed_from_u64(0);
        std::iter::repeat_with(|| (rng.gen::<M31>(), rng.gen::<M31>()))
            .take(size)
            .unzip()
    }

    #[test]
    fn test_addition() {
        let (lhs, rhs) = setup(SIZE);
        let mut packed_lhs = PackedQM31::from_array(lhs.clone());
        let packed_rhs = PackedQM31::from_array(rhs.clone());

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
        let mut packed_lhs = PackedQM31::from_array(lhs.clone());
        let packed_rhs = PackedQM31::from_array(rhs.clone());

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
        let mut packed_lhs = PackedQM31::from_array(lhs.clone());
        let packed_rhs = PackedQM31::from_array(rhs.clone());

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
        let packed_values = PackedQM31::from_array(lhs.clone());

        let res = -packed_values;

        assert_eq!(
            res.to_array(),
            lhs.iter().map(|&l| -M31(l.0)).collect::<Vec<M31>>()
        )
    }

    #[test]
    fn test_addition_ref() {
        let (lhs, rhs) = setup(SIZE);

        let packed_lhs = PackedQM31::from_array(lhs.clone());
        let packed_rhs = PackedQM31::from_array(rhs.clone());

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
