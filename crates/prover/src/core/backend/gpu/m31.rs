// CUDA implementation of packed m31
#[allow(unused_imports)]
use std::fmt::Display;
#[allow(unused_imports)]
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[allow(unused_imports)]
use num_traits::{One, Zero};

#[allow(unused_imports)]
use crate::core::fields::m31::{M31, P};
// use crate::core::fields::FieldExpOps;

pub const K_BLOCK_SIZE: usize = 16;
// pub const M512P: __m512i = unsafe { core::mem::transmute([P; K_BLOCK_SIZE]) };

// custom 512 bit vector type (TODO:: performance test on structure ie. )
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct _512u {
    pub inner: [u32; 16],
}

// CUDA implementation of avx512
/// Stores 16 M31 elements in a custom 512-bit vector.
/// Each M31 element is unreduced in the range [0, P].
#[derive(Copy, Clone, Debug)]
pub struct PackedBaseField(pub _512u);

impl PackedBaseField {
    // pub fn broadcast(value: M31) -> Self {
    //     Self(unsafe { _mm512_set1_epi32(value.0 as u32) })
    // }

    pub fn from_array(v: [M31; K_BLOCK_SIZE]) -> PackedBaseField {
        unsafe { Self(std::mem::transmute(v)) }
    }

    pub fn from_512_unchecked(x: _512u) -> Self {
        Self(x)
    }

    // pub fn to_array(self) -> [M31; K_BLOCK_SIZE] {
    //     // unsafe { std::mem::transmute(self.reduce()) }
    // }

    /// Reduces each word in the 512-bit register to the range `[0, P)`, excluding P.
    pub fn reduce(self) -> PackedBaseField {
        Self(unsafe { _mm512_min_epu32(self.0, _mm512_sub_epi32(self.0, M512P)) })
    }

    // /// Interleaves self with other.
    // /// Returns the result as two packed M31 elements.
    // pub fn interleave_with(self, other: Self) -> (Self, Self) {
    //     (
    //         Self(unsafe { _mm512_permutex2var_epi32(self.0, LHALF_INTERLEAVE_LHALF, other.0) }),
    //         Self(unsafe { _mm512_permutex2var_epi32(self.0, HHALF_INTERLEAVE_HHALF, other.0) }),
    //     )
    // }

    // /// Deinterleaves self with other.
    // /// Done by concatenating the even words of self with the even words of other, and the odd
    // words /// The inverse of [Self::interleave_with].
    // /// Returns the result as two packed M31 elements.
    // pub fn deinterleave_with(self, other: Self) -> (Self, Self) {
    //     (
    //         Self(unsafe { _mm512_permutex2var_epi32(self.0, EVENS_CONCAT_EVENS, other.0) }),
    //         Self(unsafe { _mm512_permutex2var_epi32(self.0, ODDS_CONCAT_ODDS, other.0) }),
    //     )
    // }

    // /// # Safety
    // ///
    // /// This function is unsafe because it performs a load from a raw pointer. The pointer must
    // be /// valid and aligned to 64 bytes.
    // pub unsafe fn load(ptr: *const i32) -> Self {
    //     Self(_mm512_load_epi32(ptr))
    // }

    // /// # Safety
    // ///
    // /// This function is unsafe because it performs a load from a raw pointer. The pointer must
    // be /// valid and aligned to 64 bytes.
    // pub unsafe fn store(self, ptr: *mut i32) {
    //     _mm512_store_epi32(ptr, self.0);
    // }

    // /// Sums all the elements in the packed M31 element.
    // pub fn pointwise_sum(self) -> M31 {
    //     self.to_array().into_iter().sum()
    // }
}

// impl Display for PackedBaseField {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let v = self.to_array();
//         for elem in v.iter() {
//             write!(f, "{} ", elem)?;
//         }
//         Ok(())
//     }
// }

// impl Add for PackedBaseField {
//     type Output = Self;

//     /// Adds two packed M31 elements, and reduces the result to the range `[0,P]`.
//     /// Each value is assumed to be in unreduced form, [0, P] including P.
//     #[inline(always)]
//     fn add(self, rhs: Self) -> Self::Output {
//         Self(unsafe {
//             // Add word by word. Each word is in the range [0, 2P].
//             let c = _mm512_add_epi32(self.0, rhs.0);
//             // Apply min(c, c-P) to each word.
//             // When c in [P,2P], then c-P in [0,P] which is always less than [P,2P].
//             // When c in [0,P-1], then c-P in [2^32-P,2^32-1] which is always greater than
// [0,P-1].             _mm512_min_epu32(c, _mm512_sub_epi32(c, M512P))
//         })
//     }
// }
#[cfg(test)]
mod tests {
    // use itertools::Itertools;
    #[allow(unused_imports)]
    use super::PackedBaseField;
    #[allow(unused_imports)]
    use crate::core::fields::m31::{M31, P};
    // use crate::core::fields::{Field, FieldExpOps};

    /// Tests field operations where field elements are in reduced form.
    #[test]
    fn test_gpu_basic_ops() {
        // let values = [
        //     0,
        //     1,
        //     2,
        //     10,
        //     (P - 1) / 2,
        //     (P + 1) / 2,
        //     P - 2,
        //     P - 1,
        //     0,
        //     1,
        //     2,
        //     10,
        //     (P - 1) / 2,
        //     (P + 1) / 2,
        //     P - 2,
        //     P - 1,
        // ]
        // .map(M31::from_u32_unchecked);
        // let avx_values = PackedBaseField::from_array(values);

        // assert_eq!(
        //     (avx_values + avx_values)
        //         .to_array()
        //         .into_iter()
        //         .collect_vec(),
        //     values.iter().map(|x| x.double()).collect_vec()
        // );
        // assert_eq!(
        //     (avx_values * avx_values)
        //         .to_array()
        //         .into_iter()
        //         .collect_vec(),
        //     values.iter().map(|x| x.square()).collect_vec()
        // );
        // assert_eq!(
        //     (-avx_values).to_array().into_iter().collect_vec(),
        //     values.iter().map(|x| -*x).collect_vec()
        // );
    }
}
