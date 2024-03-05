#![feature(stdsimd)]

use std::arch::x86_64::__m512i;

use criterion::Criterion;
use num_traits::{One, Zero};
use stwo::core::backend::avx512::PackedBaseField;
use stwo::core::fields::m31::BaseField;

fn _pow(x: PackedBaseField, exp: u128) -> PackedBaseField {
    let mut res = PackedBaseField::from_array([BaseField::one(); 16]);
    let mut base = x;
    let mut exp = exp;
    while exp > 0 {
        if exp & 1 == 1 {
            res *= base;
        }
        base = base * base;
        exp >>= 1;
    }
    res
}

fn pow5(x: PackedBaseField) -> PackedBaseField {
    let mut b = x * x;
    b = b * b;
    x * b
}

fn mul4(x: [PackedBaseField; 4], y: [PackedBaseField; 4]) -> [PackedBaseField; 4] {
    std::array::from_fn(|i| x[i] * y[i])
}
fn root5(mut x: [PackedBaseField; 4]) -> [PackedBaseField; 4] {
    let mut b = x; // 1
    b = mul4(b, b); // 10
    x = mul4(b, x); // 11
    b = x;
    for _ in 0..4 {
        b = mul4(b, b); // 110000
    }
    x = mul4(b, x);
    b = x;
    for _ in 0..8 {
        b = mul4(b, b); // 110000
    }
    x = mul4(b, x);
    b = x;
    for _ in 0..16 {
        b = mul4(b, b); // 110000
    }
    mul4(b, x)
}

#[cfg(target_arch = "x86_64")]
pub fn rescue_pow(c: &mut criterion::Criterion) {
    let mut ss = [PackedBaseField::from_array([BaseField::zero(); 16]); 4];
    let a = PackedBaseField::from_array([BaseField::one(); 16]);
    // let exp = ((P as u128) * 2 - 1) / 5;

    c.bench_function("rescue pow 5", |b| {
        b.iter(|| {
            for _ in 0..20 {
                for s in ss.iter_mut() {
                    *s = pow5(*s) + a;
                }
            }
        })
    });

    c.bench_function("rescue pow root5", |b| {
        b.iter(|| {
            for _ in 0..20 {
                ss = root5(ss);
                for s in ss.iter_mut() {
                    *s = *s + a;
                }
            }
        })
    });
}

#[cfg(target_arch = "x86_64")]
pub fn rescue_lu(c: &mut criterion::Criterion) {
    use std::arch::x86_64::{_mm512_and_epi32, _mm512_i32gather_epi32, _mm512_srli_epi32};

    use itertools::Itertools;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut ss: [PackedBaseField; 4] = std::array::from_fn(|j| {
        PackedBaseField::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked((i * 1789 + j) as u32)
        }))
    });
    let a = PackedBaseField::from_array(std::array::from_fn(|i| {
        BaseField::from_u32_unchecked((i * 325 + 7) as u32)
    }));

    let rng = &mut StdRng::seed_from_u64(0);
    let tbl0 = (0..(1 << 22))
        .map(|_| BaseField::from_u32_unchecked(rng.gen_range(1..(1 << 30))))
        .collect_vec();
    let tbl1 = (0..(1 << 22))
        .map(|_| BaseField::from_u32_unchecked(rng.gen_range(1..(1 << 30))))
        .collect_vec();
    const MASK22: __m512i = unsafe { core::mem::transmute([((1 << 22) - 1) as u32; 16]) };

    c.bench_function("rescue lu", |b| {
        b.iter(|| unsafe {
            for _ in 0..20 {
                for s in ss.iter_mut() {
                    let offset0 = _mm512_srli_epi32::<9>(s.0);
                    let k = PackedBaseField::from_m512_unchecked(_mm512_i32gather_epi32::<4>(
                        offset0,
                        tbl0.as_ptr() as *const u8,
                    ));
                    let sk = *s * k;
                    let offset1 = _mm512_and_epi32(sk.0, MASK22);
                    let t = PackedBaseField::from_m512_unchecked(_mm512_i32gather_epi32::<4>(
                        offset1,
                        tbl1.as_ptr() as *const u8,
                    ));
                    // TODO(spapini): Mult by a different load for tbl0;
                    *s = k * t + a;
                }
            }
        })
    });

    let tbl0 = (0..(1 << 19))
        .map(|_| BaseField::from_u32_unchecked(rng.gen_range(1..(1 << 30))))
        .collect_vec();
    let tbl1 = (0..(1 << 19))
        .map(|_| BaseField::from_u32_unchecked(rng.gen_range(1..(1 << 30))))
        .collect_vec();
    let tbl2 = (0..(1 << 18))
        .map(|_| BaseField::from_u32_unchecked(rng.gen_range(1..(1 << 30))))
        .collect_vec();
    const MASK18: __m512i = unsafe { core::mem::transmute([((1 << 18) - 1) as u32; 16]) };

    c.bench_function("rescue lu 18", |b| {
        b.iter(|| unsafe {
            for _ in 0..20 {
                for s in ss.iter_mut() {
                    let offset = _mm512_srli_epi32::<13>(s.0);
                    let k0 = PackedBaseField::from_m512_unchecked(_mm512_i32gather_epi32::<8>(
                        offset,
                        tbl0.as_ptr() as *const u8,
                    ));
                    let k1 = PackedBaseField::from_m512_unchecked(_mm512_i32gather_epi32::<8>(
                        offset,
                        (tbl0.as_ptr().add(1)) as *const u8,
                    ));
                    *s *= k0;
                    let mut res = k1;

                    let offset = _mm512_srli_epi32::<13>(s.0);
                    let k0 = PackedBaseField::from_m512_unchecked(_mm512_i32gather_epi32::<8>(
                        offset,
                        tbl1.as_ptr() as *const u8,
                    ));
                    let k1 = PackedBaseField::from_m512_unchecked(_mm512_i32gather_epi32::<8>(
                        offset,
                        (tbl1.as_ptr().add(1)) as *const u8,
                    ));
                    *s *= k0;
                    res *= k1;

                    let offset = _mm512_and_epi32(s.0, MASK18);
                    let t = PackedBaseField::from_m512_unchecked(_mm512_i32gather_epi32::<4>(
                        offset,
                        tbl2.as_ptr() as *const u8,
                    ));
                    *s = res * t;
                }
            }
        })
    });

    let tbl0 = (0..(1 << 17))
        .map(|_| BaseField::from_u32_unchecked(rng.gen_range(1..(1 << 30))))
        .collect_vec();
    let tbl1 = (0..(1 << 17))
        .map(|_| BaseField::from_u32_unchecked(rng.gen_range(1..(1 << 30))))
        .collect_vec();
    let tbl2 = (0..(1 << 17))
        .map(|_| BaseField::from_u32_unchecked(rng.gen_range(1..(1 << 30))))
        .collect_vec();
    let tbl3 = (0..(1 << 16))
        .map(|_| BaseField::from_u32_unchecked(rng.gen_range(1..(1 << 30))))
        .collect_vec();
    const MASK16: __m512i = unsafe { core::mem::transmute([((1 << 16) - 1) as u32; 16]) };

    c.bench_function("rescue lu 16", |b| {
        b.iter(|| unsafe {
            for _ in 0..20 {
                for s in ss.iter_mut() {
                    let offset = _mm512_srli_epi32::<15>(s.0);
                    let k0 = PackedBaseField::from_m512_unchecked(_mm512_i32gather_epi32::<8>(
                        offset,
                        tbl0.as_ptr() as *const u8,
                    ));
                    let k1 = PackedBaseField::from_m512_unchecked(_mm512_i32gather_epi32::<8>(
                        offset,
                        (tbl0.as_ptr().add(1)) as *const u8,
                    ));
                    *s *= k0;
                    let mut res = k1;

                    let offset = _mm512_srli_epi32::<15>(s.0);
                    let k0 = PackedBaseField::from_m512_unchecked(_mm512_i32gather_epi32::<8>(
                        offset,
                        tbl1.as_ptr() as *const u8,
                    ));
                    let k1 = PackedBaseField::from_m512_unchecked(_mm512_i32gather_epi32::<8>(
                        offset,
                        (tbl1.as_ptr().add(1)) as *const u8,
                    ));
                    *s *= k0;
                    res *= k1;

                    let offset = _mm512_srli_epi32::<15>(s.0);
                    let k0 = PackedBaseField::from_m512_unchecked(_mm512_i32gather_epi32::<8>(
                        offset,
                        tbl2.as_ptr() as *const u8,
                    ));
                    let k1 = PackedBaseField::from_m512_unchecked(_mm512_i32gather_epi32::<8>(
                        offset,
                        (tbl2.as_ptr().add(1)) as *const u8,
                    ));
                    *s *= k0;
                    res *= k1;

                    let offset = _mm512_and_epi32(s.0, MASK16);
                    let t = PackedBaseField::from_m512_unchecked(_mm512_i32gather_epi32::<4>(
                        offset,
                        tbl3.as_ptr() as *const u8,
                    ));
                    *s = res * t;
                }
            }
        })
    });

    c.bench_function("rescue inv lu 16", |b| {
        b.iter(|| unsafe {
            for _ in 0..20 {
                for s in ss.iter_mut() {
                    let offset = _mm512_srli_epi32::<15>(s.0);
                    let k0 = PackedBaseField::from_m512_unchecked(_mm512_i32gather_epi32::<4>(
                        offset,
                        tbl0.as_ptr() as *const u8,
                    ));
                    *s *= k0;
                    let mut res = k0;

                    let offset = _mm512_srli_epi32::<15>(s.0);
                    let k0 = PackedBaseField::from_m512_unchecked(_mm512_i32gather_epi32::<4>(
                        offset,
                        tbl1.as_ptr() as *const u8,
                    ));
                    *s *= k0;
                    res *= k0;

                    let offset = _mm512_srli_epi32::<15>(s.0);
                    let k0 = PackedBaseField::from_m512_unchecked(_mm512_i32gather_epi32::<4>(
                        offset,
                        tbl2.as_ptr() as *const u8,
                    ));
                    *s *= k0;
                    res *= k0;

                    let offset = _mm512_and_epi32(s.0, MASK16);
                    let t = PackedBaseField::from_m512_unchecked(_mm512_i32gather_epi32::<4>(
                        offset,
                        tbl3.as_ptr() as *const u8,
                    ));
                    *s = res * t;
                }
            }
        })
    });
}

#[cfg(target_arch = "x86_64")]
criterion::criterion_group!(
    name=hash_bench;
    config = Criterion::default().sample_size(10);
    targets=rescue_pow,rescue_lu);
criterion::criterion_main!(hash_bench);
