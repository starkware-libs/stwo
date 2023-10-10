use super::m31::{K_BITS, M31, P};
use core::arch::x86_64::*;

pub const K_BLOCK_SIZE: usize = 8;

#[derive(Copy, Clone)]
pub struct Consts {
    p: __m512i,
    one: __m512i,
}

impl Consts {
    pub fn new() -> Self {
        Self {
            p: unsafe { _mm512_set1_epi64(P as i64) },
            one: unsafe { _mm512_set1_epi64(1) },
        }
    }
}

impl Default for Consts {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct M31AVX512(__m512i);

impl M31AVX512 {
    pub fn one() -> Self {
        Self(unsafe { _mm512_set1_epi64(1) })
    }

    #[inline(always)]
    pub fn add(self, cn: Consts, rhs: Self) -> Self {
        unsafe {
            let sum = _mm512_add_epi64(self.0, rhs.0);
            let shifted_sum = _mm512_srli_epi64(sum, K_BITS);
            Self(_mm512_and_si512(_mm512_add_epi64(sum, shifted_sum), cn.p))
        }
    }

    #[inline(always)]
    fn reduce(cn: Consts, a: __m512i) -> Self {
        unsafe {
            let a_plus_one: __m512i = _mm512_add_epi64(a, cn.one);
            let z: __m512i = _mm512_srli_epi64(
                _mm512_add_epi64(_mm512_srli_epi64(a, K_BITS), a_plus_one),
                K_BITS,
            );
            let result: __m512i = _mm512_add_epi64(a, z);
            Self(_mm512_and_epi64(result, cn.p))
        }
    }

    #[inline(always)]
    pub fn mul(self, cn: Consts, rhs: M31AVX512) -> Self {
        unsafe { Self::reduce(cn, _mm512_mul_epu32(self.0, rhs.0)) }
    }

    pub fn from_vec_unchecked(v: &Vec<M31>) -> M31AVX512 {
        unsafe {
            Self(_mm512_cvtepu32_epi64(_mm256_loadu_si256(
                v.as_ptr() as *const __m256i
            )))
        }
    }

    pub fn to_vec(self) -> Vec<M31> {
        unsafe {
            let mut v = Vec::with_capacity(K_BLOCK_SIZE);
            _mm256_storeu_si256(
                v.as_mut_ptr() as *mut __m256i,
                _mm512_cvtepi64_epi32(self.0),
            );
            v.set_len(K_BLOCK_SIZE);
            v
        }
    }
}

#[test]
fn test_avx512_mul() {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let cn = Consts::new();

    let values = (0..K_BLOCK_SIZE)
        .map(|_x| M31::from_u32_unchecked(rng.gen::<u32>() % P))
        .collect::<Vec<M31>>();
    let avx_values = M31AVX512::from_vec_unchecked(&values);

    let double_avx_values = avx_values.add(cn, avx_values);
    let square_avx_values = avx_values.mul(cn, avx_values);
    assert_eq!(
        double_avx_values.to_vec(),
        values.iter().map(|x| x.double()).collect::<Vec<_>>()
    );
    assert_eq!(
        square_avx_values.to_vec(),
        values.iter().map(|x| x.square()).collect::<Vec<_>>()
    );
}
