#[allow(unused_imports)]
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use cudarc::driver::{CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
#[allow(unused_imports)]
use itertools::Itertools;
#[allow(unused_imports)]
use num_traits::{One, Zero};

use super::{Device, DEVICE};
#[allow(unused_imports)]
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
        let curr_dir = std::env::current_dir()
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        let opts = CompileOptions {
            include_paths: vec![curr_dir + "/src/core/backend/gpu"],
            ..Default::default()
        };
        let ptx_mul_m31 = compile_ptx_with_opts(ptx_src_mul_m31, opts).unwrap();
        self.load_ptx(ptx_mul_m31, "secure_field_functions", &["mul"])
            .unwrap();
    }
}

// Flattened PackedQM31
#[derive(Debug)]
#[repr(C)]
pub struct PackedQM31(pub CudaSlice<M31>);

#[allow(dead_code)]
impl PackedQM31 {
    /// Constructs a new instance with all vector elements set to `value`.
    pub fn broadcast(v: M31, size: Option<usize>) -> Self {
        let size = size.unwrap_or(1024);
        Self(DEVICE.htod_copy(vec![v; size * 4]).unwrap())
    }

    pub fn from_array(v: Vec<QM31>) -> PackedQM31 {
        let flat: Vec<M31> = v
            .into_iter()
            .flat_map(|qm31| vec![qm31.0 .0, qm31.0 .1, qm31.1 .0, qm31.1 .1])
            .collect();
        let slice = DEVICE.htod_copy(flat).unwrap();
        Self(slice)
    }

    pub fn to_array(self) -> Vec<QM31> {
        let flat = DEVICE.dtoh_sync_copy(&self.0).unwrap();

        flat.chunks_exact(4)
            .map(|m31x4| {
                let [a, b, c, d] = m31x4.try_into().unwrap();
                QM31(CM31(a, b), CM31(c, d))
            })
            .collect()
    }
}

// Clone is a device to device copy
impl Clone for PackedQM31 {
    fn clone(&self) -> Self {
        let mut out = unsafe { self.0.device().alloc::<M31>(self.0.len()) }.unwrap();
        self.0.device().dtod_copy(&self.0, &mut out).unwrap();

        Self(out)
    }
}

impl Add for PackedQM31 {
    type Output = Self;

    /// Adds two packed M31 elements, and reduces the result to the range `[0,P]`.
    /// Each value is assumed to be in unreduced form, [0, P] including P.
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let kernel = self
            .0
            .device()
            .get_func("base_field_functions", "add")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);
        let out = unsafe { self.0.device().alloc::<M31>(self.0.len()) }.unwrap();

        unsafe {
            kernel
                .clone()
                .launch(cfg, (&self.0, &rhs.0, &out, self.0.len()))
        }
        .unwrap();

        DEVICE.synchronize().unwrap();

        Self(out)
    }
}

// Adds in place
impl AddAssign for PackedQM31 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        let kernel = self
            .0
            .device()
            .get_func("base_field_functions", "add")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);

        unsafe {
            kernel
                .clone()
                .launch(cfg, (&self.0, &rhs.0, &self.0, self.0.len()))
        }
        .unwrap();

        DEVICE.synchronize().unwrap();
    }
}

// impl Mul for PackedQM31 {
//     type Output = Self;

//     /// Computes the product of two packed M31 elements
//     /// Each value is assumed to be in unreduced form, [0, P] including P.
//     /// Returned values are in unreduced form, [0, P] including P.
//     #[inline(always)]
//     fn mul(self, rhs: Self) -> Self::Output {
//         let mul_kernel = self.0[0]
//             .0
//             .device()
//             .get_func("base_field_functions", "mul")
//             .unwrap();
//         let len = self.0[0].0.len();
//         let cfg: LaunchConfig = LaunchConfig::for_num_elems(len as u32);
//         let a = DEVICE.alloc_zeros::<u32>(len).unwrap();
//         let b = DEVICE.alloc_zeros::<u32>(len).unwrap();
//         let c = DEVICE.alloc_zeros::<u32>(len).unwrap();
//         let d = DEVICE.alloc_zeros::<u32>(len).unwrap();

//         let stream_b = DEVICE.fork_default_stream().unwrap();
//         let stream_c = DEVICE.fork_default_stream().unwrap();
//         let stream_d = DEVICE.fork_default_stream().unwrap();

//         unsafe {
//             mul_kernel
//                 .clone()
//                 .launch(cfg, (&self.0[0].0, &rhs.0[0].0, &a, len))
//         }
//         .unwrap();
//         unsafe {
//             mul_kernel.clone().launch_on_stream(
//                 &stream_b,
//                 cfg,
//                 (&self.0[1].0, &rhs.0[1].0, &b, len),
//             )
//         }
//         .unwrap();
//         unsafe {
//             mul_kernel.clone().launch_on_stream(
//                 &stream_c,
//                 cfg,
//                 (&self.0[2].0, &rhs.0[2].0, &c, len),
//             )
//         }
//         .unwrap();
//         unsafe {
//             mul_kernel.launch_on_stream(&stream_d, cfg, (&self.0[3].0, &rhs.0[3].0, &d, len))
//         }
//         .unwrap();

//         DEVICE.synchronize().unwrap();

//         Self([PackedM31(a), PackedM31(b), PackedM31(c), PackedM31(d)])
//     }
// }

// impl MulAssign for PackedQM31 {
//     #[inline(always)]
//     fn mul_assign(&mut self, rhs: Self) {
//

//         let mul_kernel = self.0[0]
//             .0
//             .device()
//             .get_func("base_field_functions", "mul")
//             .unwrap();
//         let len = self.0[0].0.len();
//         let cfg: LaunchConfig = LaunchConfig::for_num_elems(len as u32);

//         let stream_b = DEVICE.fork_default_stream().unwrap();
//         let stream_c = DEVICE.fork_default_stream().unwrap();
//         let stream_d = DEVICE.fork_default_stream().unwrap();

//         unsafe {
//             mul_kernel
//                 .clone()
//                 .launch(cfg, (&self.0[0].0, &rhs.0[0].0, &self.0[0].0, len))
//         }
//         .unwrap();
//         unsafe {
//             mul_kernel.clone().launch_on_stream(
//                 &stream_b,
//                 cfg,
//                 (&self.0[1].0, &rhs.0[1].0, &self.0[1].0, len),
//             )
//         }
//         .unwrap();
//         unsafe {
//             mul_kernel.clone().launch_on_stream(
//                 &stream_c,
//                 cfg,
//                 (&self.0[2].0, &rhs.0[2].0, &self.0[2].0, len),
//             )
//         }
//         .unwrap();
//         unsafe {
//             mul_kernel.launch_on_stream(
//                 &stream_d,
//                 cfg,
//                 (&self.0[3].0, &rhs.0[3].0, &self.0[3].0, len),
//             )
//         }
//         .unwrap();

//         DEVICE.synchronize().unwrap();
//     }
// }

impl Neg for PackedQM31 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        let kernel = self
            .0
            .device()
            .get_func("base_field_functions", "neg")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);

        unsafe { kernel.clone().launch(cfg, (&self.0, self.0.len())) }.unwrap();

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
        let kernel = self
            .0
            .device()
            .get_func("base_field_functions", "sub")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);
        let out = unsafe { self.0.device().alloc::<M31>(self.0.len()) }.unwrap();

        unsafe {
            kernel
                .clone()
                .launch(cfg, (&self.0, &rhs.0, &out, self.0.len()))
        }
        .unwrap();

        DEVICE.synchronize().unwrap();

        Self(out)
    }
}

// Subtract in place
impl SubAssign for PackedQM31 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        let kernel = self
            .0
            .device()
            .get_func("base_field_functions", "sub")
            .unwrap();
        let cfg: LaunchConfig = LaunchConfig::for_num_elems(self.0.len() as u32);

        unsafe {
            kernel
                .clone()
                .launch(cfg, (&self.0, &rhs.0, &self.0, self.0.len()))
        }
        .unwrap();

        DEVICE.synchronize().unwrap();
    }
}

// // Returns a single zero
// impl Zero for PackedQM31 {
//     fn zero() -> Self {
//         let a = PackedM31::zero();
//         let b = PackedM31::zero();
//         let c = PackedM31::zero();
//         let d = PackedM31::zero();
//         PackedQM31([a, b, c, d])
//     }

//     fn is_zero(&self) -> bool {
//         let a = self.0[0].clone().to_vec().iter().all(M31::is_zero);
//         let b = self.0[0].clone().to_vec().iter().all(M31::is_zero);
//         let c = self.0[0].clone().to_vec().iter().all(M31::is_zero);
//         let d = self.0[0].clone().to_vec().iter().all(M31::is_zero);
//         a && b && c && d
//     }
// }

// impl One for PackedQM31 {
//     fn one() -> Self {
//         let a = PackedM31::one();
//         let b = PackedM31::one();
//         let c = PackedM31::one();
//         let d = PackedM31::one();
//         PackedQM31([a, b, c, d])
//     }
// }

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
    use crate::core::fields::qm31::QM31;

    const SIZE: usize = 1 << 4;

    fn setup(size: usize) -> (Vec<QM31>, Vec<QM31>) {
        let mut rng: SmallRng = SmallRng::seed_from_u64(0);
        (0..size)
            .into_iter()
            .map(|_| (rng.gen::<QM31>(), rng.gen::<QM31>()))
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
                .collect::<Vec<QM31>>()
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
                .collect::<Vec<QM31>>()
        );
    }

    // #[test]
    // fn test_multiplication() {
    //     let (lhs, rhs) = setup(4);
    //     let mut packed_lhs = PackedQM31::from_array(lhs.clone());
    //     let packed_rhs = PackedQM31::from_array(rhs.clone());

    //     println!("{:?}", lhs);
    //     println!("{:?}", rhs);
    //     println!("{:?}", packed_lhs.clone().to_array());
    //     println!("{:?}", packed_rhs.clone().to_array());
    //     packed_lhs = packed_lhs * packed_rhs;

    //     assert_eq!(
    //         packed_lhs.to_array(),
    //         lhs.iter()
    //             .zip(rhs.iter())
    //             .map(|(&l, &r)| l * r)
    //             .collect::<Vec<QM31>>()
    //     );
    // }

    #[test]
    fn test_negation() {
        let (lhs, _) = setup(SIZE);
        let packed_lhs = PackedQM31::from_array(lhs.clone());

        let packed_lhs = -packed_lhs;

        assert_eq!(
            packed_lhs.to_array(),
            lhs.iter().map(|&l| -l).collect::<Vec<QM31>>()
        );
    }

    // #[test]
    // fn test_addition_ref() {
    //     let (lhs, rhs) = setup(SIZE);

    //     let packed_lhs = PackedQM31::from_array(lhs.clone());
    //     let packed_rhs = PackedQM31::from_array(rhs.clone());

    //     packed_lhs.add_assign_ref(&packed_rhs);

    //     assert_eq!(
    //         packed_lhs.to_array(),
    //         lhs.iter()
    //             .zip(rhs.iter())
    //             .map(|(&l, &r)| l + r)
    //             .collect::<Vec<M31>>()
    //     );
    // }
}
