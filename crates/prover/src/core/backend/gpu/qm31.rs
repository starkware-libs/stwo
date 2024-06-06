// CUDA implementation of arbitrary size packed m31
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use cudarc::driver::{DeviceSlice, LaunchAsync, LaunchConfig};
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
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0[0].0.len() as u32);
        let a = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let b = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let c = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let d = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();

        let stream_b = DEVICE.fork_default_stream().unwrap();
        let stream_c = DEVICE.fork_default_stream().unwrap();
        let stream_d = DEVICE.fork_default_stream().unwrap();

        unsafe {
            add_kernel
                .clone()
                .launch(cfg, (&self.0[0].0, &rhs.0[0].0, &a, self.0.len()))
        }
        .unwrap();
        unsafe {
            add_kernel.clone().launch_on_stream(
                &stream_b,
                cfg,
                (&self.0[1].0, &rhs.0[1].0, &b, self.0.len()),
            )
        }
        .unwrap();
        unsafe {
            add_kernel.clone().launch_on_stream(
                &stream_c,
                cfg,
                (&self.0[2].0, &rhs.0[2].0, &c, self.0.len()),
            )
        }
        .unwrap();
        unsafe {
            add_kernel.launch_on_stream(
                &stream_d,
                cfg,
                (&self.0[3].0, &rhs.0[3].0, &d, self.0.len()),
            )
        }
        .unwrap();

        DEVICE.synchronize().unwrap();

        Self([PackedM31(a), PackedM31(b), PackedM31(c), PackedM31(d)])
    }
}

// Adds in place
impl AddAssign for PackedQM31 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        let add_kernel = self.0[0]
            .0
            .device()
            .get_func("base_field_functions", "add")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0[0].0.len() as u32);
        let a = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let b = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let c = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let d = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();

        let stream_b = DEVICE.fork_default_stream().unwrap();
        let stream_c = DEVICE.fork_default_stream().unwrap();
        let stream_d = DEVICE.fork_default_stream().unwrap();

        unsafe {
            add_kernel
                .clone()
                .launch(cfg, (&self.0[0].0, &rhs.0[0].0, &a, self.0.len()))
        }
        .unwrap();
        unsafe {
            add_kernel.clone().launch_on_stream(
                &stream_b,
                cfg,
                (&self.0[1].0, &rhs.0[1].0, &b, self.0.len()),
            )
        }
        .unwrap();
        unsafe {
            add_kernel.clone().launch_on_stream(
                &stream_c,
                cfg,
                (&self.0[2].0, &rhs.0[2].0, &c, self.0.len()),
            )
        }
        .unwrap();
        unsafe {
            add_kernel.launch_on_stream(
                &stream_d,
                cfg,
                (&self.0[3].0, &rhs.0[3].0, &d, self.0.len()),
            )
        }
        .unwrap();

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
        let mul_kernel = self.0[0]
            .0
            .device()
            .get_func("base_field_functions", "mul")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0[0].0.len() as u32);
        let a = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let b = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let c = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let d = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();

        let stream_b = DEVICE.fork_default_stream().unwrap();
        let stream_c = DEVICE.fork_default_stream().unwrap();
        let stream_d = DEVICE.fork_default_stream().unwrap();

        unsafe {
            mul_kernel
                .clone()
                .launch(cfg, (&self.0[0].0, &rhs.0[0].0, &a, self.0.len()))
        }
        .unwrap();
        unsafe {
            mul_kernel.clone().launch_on_stream(
                &stream_b,
                cfg,
                (&self.0[1].0, &rhs.0[1].0, &b, self.0.len()),
            )
        }
        .unwrap();
        unsafe {
            mul_kernel.clone().launch_on_stream(
                &stream_c,
                cfg,
                (&self.0[2].0, &rhs.0[2].0, &c, self.0.len()),
            )
        }
        .unwrap();
        unsafe {
            mul_kernel.launch_on_stream(
                &stream_d,
                cfg,
                (&self.0[3].0, &rhs.0[3].0, &d, self.0.len()),
            )
        }
        .unwrap();

        DEVICE.synchronize().unwrap();

        Self([PackedM31(a), PackedM31(b), PackedM31(c), PackedM31(d)])
    }
}

impl MulAssign for PackedQM31 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        let mul_kernel = self.0[0]
            .0
            .device()
            .get_func("base_field_functions", "mul")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0[0].0.len() as u32);
        let a = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let b = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let c = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let d = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();

        let stream_b = DEVICE.fork_default_stream().unwrap();
        let stream_c = DEVICE.fork_default_stream().unwrap();
        let stream_d = DEVICE.fork_default_stream().unwrap();

        unsafe {
            mul_kernel
                .clone()
                .launch(cfg, (&self.0[0].0, &rhs.0[0].0, &a, self.0.len()))
        }
        .unwrap();
        unsafe {
            mul_kernel.clone().launch_on_stream(
                &stream_b,
                cfg,
                (&self.0[1].0, &rhs.0[1].0, &b, self.0.len()),
            )
        }
        .unwrap();
        unsafe {
            mul_kernel.clone().launch_on_stream(
                &stream_c,
                cfg,
                (&self.0[2].0, &rhs.0[2].0, &c, self.0.len()),
            )
        }
        .unwrap();
        unsafe {
            mul_kernel.launch_on_stream(
                &stream_d,
                cfg,
                (&self.0[3].0, &rhs.0[3].0, &d, self.0.len()),
            )
        }
        .unwrap();

        DEVICE.synchronize().unwrap();
    }
}

impl Neg for PackedQM31 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        let neg_kernel = self.0[0]
            .0
            .device()
            .get_func("base_field_functions", "neg")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0[0].0.len() as u32);

        let stream_b = DEVICE.fork_default_stream().unwrap();
        let stream_c = DEVICE.fork_default_stream().unwrap();
        let stream_d = DEVICE.fork_default_stream().unwrap();

        unsafe { neg_kernel.clone().launch(cfg, (&self.0[0].0, self.0.len())) }.unwrap();
        unsafe {
            neg_kernel
                .clone()
                .launch_on_stream(&stream_b, cfg, (&self.0[1].0, self.0.len()))
        }
        .unwrap();
        unsafe {
            neg_kernel
                .clone()
                .launch_on_stream(&stream_c, cfg, (&self.0[2].0, self.0.len()))
        }
        .unwrap();
        unsafe { neg_kernel.launch_on_stream(&stream_d, cfg, (&self.0[3].0, self.0.len())) }
            .unwrap();

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
        let sub_kernel = self.0[0]
            .0
            .device()
            .get_func("base_field_functions", "sub")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0[0].0.len() as u32);

        let a = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let b = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let c = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let d = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();

        let stream_b = DEVICE.fork_default_stream().unwrap();
        let stream_c = DEVICE.fork_default_stream().unwrap();
        let stream_d = DEVICE.fork_default_stream().unwrap();

        unsafe {
            sub_kernel
                .clone()
                .launch(cfg, (&self.0[0].0, &rhs.0[0].0, &a, self.0.len()))
        }
        .unwrap();
        unsafe {
            sub_kernel.clone().launch_on_stream(
                &stream_b,
                cfg,
                (&self.0[1].0, &rhs.0[1].0, &b, self.0.len()),
            )
        }
        .unwrap();
        unsafe {
            sub_kernel.clone().launch_on_stream(
                &stream_c,
                cfg,
                (&self.0[2].0, &rhs.0[2].0, &c, self.0.len()),
            )
        }
        .unwrap();
        unsafe {
            sub_kernel.launch_on_stream(
                &stream_d,
                cfg,
                (&self.0[3].0, &rhs.0[3].0, &d, self.0.len()),
            )
        }
        .unwrap();

        DEVICE.synchronize().unwrap();

        Self([PackedM31(a), PackedM31(b), PackedM31(c), PackedM31(d)])
    }
}

// Subtract in place
impl SubAssign for PackedQM31 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        let sub_kernel = self.0[0]
            .0
            .device()
            .get_func("base_field_functions", "sub")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0[0].0.len() as u32);
        let a = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let b = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let c = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();
        let d = DEVICE.alloc_zeros::<u32>(self.0.len()).unwrap();

        let stream_b = DEVICE.fork_default_stream().unwrap();
        let stream_c = DEVICE.fork_default_stream().unwrap();
        let stream_d = DEVICE.fork_default_stream().unwrap();

        unsafe {
            sub_kernel
                .clone()
                .launch(cfg, (&self.0[0].0, &rhs.0[0].0, &a, self.0.len()))
        }
        .unwrap();
        unsafe {
            sub_kernel.clone().launch_on_stream(
                &stream_b,
                cfg,
                (&self.0[1].0, &rhs.0[1].0, &b, self.0.len()),
            )
        }
        .unwrap();
        unsafe {
            sub_kernel.clone().launch_on_stream(
                &stream_c,
                cfg,
                (&self.0[2].0, &rhs.0[2].0, &c, self.0.len()),
            )
        }
        .unwrap();
        unsafe {
            sub_kernel.launch_on_stream(
                &stream_d,
                cfg,
                (&self.0[3].0, &rhs.0[3].0, &d, self.0.len()),
            )
        }
        .unwrap();

        DEVICE.synchronize().unwrap();
    }
}

// Returns a single zero
impl Zero for PackedQM31 {
    fn zero() -> Self {
        let a = PackedM31::zero();
        let b = PackedM31::zero();
        let c = PackedM31::zero();
        let d = PackedM31::zero();
        PackedQM31([a, b, c, d])
    }

    fn is_zero(&self) -> bool {
        let a = self.0[0].clone().to_vec().iter().all(M31::is_zero);
        let b = self.0[0].clone().to_vec().iter().all(M31::is_zero);
        let c = self.0[0].clone().to_vec().iter().all(M31::is_zero);
        let d = self.0[0].clone().to_vec().iter().all(M31::is_zero);
        a && b && c && d
    }
}

impl One for PackedQM31 {
    fn one() -> Self {
        let a = PackedM31::one();
        let b = PackedM31::one();
        let c = PackedM31::one();
        let d = PackedM31::one();
        PackedQM31([a, b, c, d])
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

    fn setup(size: usize) -> (Vec<M31>, Vec<M31>, Vec<M31>, Vec<M31>) {
        let mut rng: SmallRng = SmallRng::seed_from_u64(0);
        let (a, b) = std::iter::repeat_with(|| (rng.gen::<M31>(), rng.gen::<M31>()))
            .take(size)
            .unzip();
        let (c, d) = std::iter::repeat_with(|| (rng.gen::<M31>(), rng.gen::<M31>()))
            .take(size)
            .unzip();
        (a, b, c, d)
    }

    #[test]
    fn test_addition() {
        let (a_lhs, b_lhs, c_lhs, d_lhs) = setup(SIZE);
        let (a_rhs, b_rhs, c_rhs, d_rhs) = setup(SIZE);

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
