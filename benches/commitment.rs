#![feature(sync_unsafe_cell)]
#![feature(portable_simd)]
#![feature(stdsimd)]
#![feature(iter_array_chunks)]

use criterion::{criterion_group, criterion_main, Criterion};
use prover_research::core::{fft::FFTree, poly::line::LineDomain};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::{
    arch::x86_64::{
        __m512i, _mm512_add_epi64, _mm512_and_epi32, _mm512_load_epi32, _mm512_mul_epi32,
        _mm512_or_epi32, _mm512_set1_epi32, _mm512_slli_epi64, _mm512_srli_epi64,
        _mm512_store_epi32, _mm512_sub_epi64,
    },
    cell::SyncUnsafeCell,
};

const N_VEC_BITS: usize = 4;
// const N_VEC: usize = 1 << N_VEC_BITS;
const N_THR_BITS: usize = 3;
const N_THR: usize = 1 << N_THR_BITS;
const N_TOTAL_BITS: usize = 28;

#[derive(Copy, Clone)]
struct SomeWords([__m512i; 8]);

#[inline(never)]
unsafe fn fft3<const I_OFF: usize>(_tree: &FFTree, values: *mut i32, i_thr: usize) {
    let a_shift = N_TOTAL_BITS - I_OFF - 3;
    let twiddle = _mm512_set1_epi32(1);
    let mask = _mm512_set1_epi32(i32::MAX);
    let mask1 = _mm512_set1_epi32(-1);
    // 2**18 times.
    for i_h in 0..(1 << I_OFF) {
        let index0 = (i_h << (N_TOTAL_BITS - I_OFF)) | (i_thr << N_VEC_BITS);
        for i_l in 0..(1 << (N_TOTAL_BITS - I_OFF - 3 - N_THR_BITS - N_VEC_BITS)) {
            let index = index0 | (i_l << (N_THR_BITS + N_VEC_BITS));

            // 8 * 4 = 32 cc
            let val0 = _mm512_load_epi32(values.add(index | (0 << a_shift)).cast_const());
            let val1 = _mm512_load_epi32(values.add(index | (1 << a_shift)).cast_const());
            let val2 = _mm512_load_epi32(values.add(index | (2 << a_shift)).cast_const());
            let val3 = _mm512_load_epi32(values.add(index | (3 << a_shift)).cast_const());
            let val4 = _mm512_load_epi32(values.add(index | (4 << a_shift)).cast_const());
            let val5 = _mm512_load_epi32(values.add(index | (5 << a_shift)).cast_const());
            let val6 = _mm512_load_epi32(values.add(index | (6 << a_shift)).cast_const());
            let val7 = _mm512_load_epi32(values.add(index | (7 << a_shift)).cast_const());
            let (val0l, val0h) = split(mask1, val0);
            let (val1l, val1h) = split(mask1, val1);
            let (val2l, val2h) = split(mask1, val2);
            let (val3l, val3h) = split(mask1, val3);
            let (val4l, val4h) = split(mask1, val4);
            let (val5l, val5h) = split(mask1, val5);
            let (val6l, val6h) = split(mask1, val6);
            let (val7l, val7h) = split(mask1, val7);

            // 9*24 = 216 cc
            let a = SomeWords([val0l, val0h, val1l, val1h, val2l, val2h, val3l, val3h]);
            let b = SomeWords([val4l, val4h, val5l, val5h, val6l, val6h, val7l, val7h]);
            let (a, b) = butterfly(mask, a, b, twiddle);
            let SomeWords([val0l, val0h, val1l, val1h, val2l, val2h, val3l, val3h]) = a;
            let SomeWords([val4l, val4h, val5l, val5h, val6l, val6h, val7l, val7h]) = b;

            let a = SomeWords([val0l, val0h, val1l, val1h, val4l, val4h, val5l, val5h]);
            let b = SomeWords([val2l, val2h, val3l, val3h, val6l, val6h, val7l, val7h]);
            let (a, b) = butterfly(mask, a, b, twiddle);
            let SomeWords([val0l, val0h, val1l, val1h, val4l, val4h, val5l, val5h]) = a;
            let SomeWords([val2l, val2h, val3l, val3h, val6l, val6h, val7l, val7h]) = b;

            let a = SomeWords([val0l, val0h, val2l, val2h, val4l, val4h, val6l, val6h]);
            let b = SomeWords([val1l, val1h, val3l, val3h, val5l, val5h, val7l, val7h]);
            let (a, b) = butterfly(mask, a, b, twiddle);
            let SomeWords([val0l, val0h, val2l, val2h, val4l, val4h, val6l, val6h]) = a;
            let SomeWords([val1l, val1h, val3l, val3h, val5l, val5h, val7l, val7h]) = b;

            // 8*4 = 32cc
            let val0 = combine(val0l, val0h);
            let val1 = combine(val1l, val1h);
            let val2 = combine(val2l, val2h);
            let val3 = combine(val3l, val3h);
            let val4 = combine(val4l, val4h);
            let val5 = combine(val5l, val5h);
            let val6 = combine(val6l, val6h);
            let val7 = combine(val7l, val7h);
            _mm512_store_epi32(values.add(index | (0 << a_shift)), val0);
            _mm512_store_epi32(values.add(index | (1 << a_shift)), val1);
            _mm512_store_epi32(values.add(index | (2 << a_shift)), val2);
            _mm512_store_epi32(values.add(index | (3 << a_shift)), val3);
            _mm512_store_epi32(values.add(index | (4 << a_shift)), val4);
            _mm512_store_epi32(values.add(index | (5 << a_shift)), val5);
            _mm512_store_epi32(values.add(index | (6 << a_shift)), val6);
            _mm512_store_epi32(values.add(index | (7 << a_shift)), val7);
        }
    }
}

// 1.5 cc
unsafe fn split(mask1: __m512i, val: __m512i) -> (__m512i, __m512i) {
    let h = _mm512_srli_epi64(val, 32);
    let l = _mm512_and_epi32(val, mask1);
    (l, h)
}

// 1.5 cc
unsafe fn combine(l: __m512i, h: __m512i) -> __m512i {
    _mm512_or_epi32(_mm512_slli_epi64(h, 32), l)
}

// 2 cc
unsafe fn reduce(mask: __m512i, x: SomeWords) -> SomeWords {
    let x_h = words_srli::<31>(x);
    let x_l = words_and(
        x,
        SomeWords([mask, mask, mask, mask, mask, mask, mask, mask]),
    );
    words_add(x_h, x_l)
}

// 9 cc
unsafe fn butterfly(
    mask: __m512i, // mask is 2^31-1.
    val0: SomeWords,
    val1: SomeWords,
    twiddle: __m512i,
) -> (SomeWords, SomeWords) {
    /*
        Compute val0+twiddle*val1, val0-twiddle*val1.
        Answer is reduced modulo 2^31-1 in an efficient way.
    */
    // m = val1*twiddle
    let m = reduce(
        mask,
        words_mul(
            val1,
            SomeWords([
                twiddle, twiddle, twiddle, twiddle, twiddle, twiddle, twiddle, twiddle,
            ]),
        ),
    );

    // m is in [0, 2^31-1].
    let r0 = reduce(mask, words_add(val0, m));
    // TODO; this is not correct.
    let r1 = reduce(mask, words_sub(val0, m));

    (r0, r1)
}

unsafe fn words_mul(a: SomeWords, b: SomeWords) -> SomeWords {
    let SomeWords([a0, a1, a2, a3, a4, a5, a6, a7]) = a;
    let SomeWords([b0, b1, b2, b3, b4, b5, b6, b7]) = b;
    SomeWords([
        _mm512_mul_epi32(a0, b0),
        _mm512_mul_epi32(a1, b1),
        _mm512_mul_epi32(a2, b2),
        _mm512_mul_epi32(a3, b3),
        _mm512_mul_epi32(a4, b4),
        _mm512_mul_epi32(a5, b5),
        _mm512_mul_epi32(a6, b6),
        _mm512_mul_epi32(a7, b7),
    ])
}

unsafe fn words_add(a: SomeWords, b: SomeWords) -> SomeWords {
    let SomeWords([a0, a1, a2, a3, a4, a5, a6, a7]) = a;
    let SomeWords([b0, b1, b2, b3, b4, b5, b6, b7]) = b;
    SomeWords([
        _mm512_add_epi64(a0, b0),
        _mm512_add_epi64(a1, b1),
        _mm512_add_epi64(a2, b2),
        _mm512_add_epi64(a3, b3),
        _mm512_add_epi64(a4, b4),
        _mm512_add_epi64(a5, b5),
        _mm512_add_epi64(a6, b6),
        _mm512_add_epi64(a7, b7),
    ])
}

unsafe fn words_sub(a: SomeWords, b: SomeWords) -> SomeWords {
    let SomeWords([a0, a1, a2, a3, a4, a5, a6, a7]) = a;
    let SomeWords([b0, b1, b2, b3, b4, b5, b6, b7]) = b;
    SomeWords([
        _mm512_sub_epi64(a0, b0),
        _mm512_sub_epi64(a1, b1),
        _mm512_sub_epi64(a2, b2),
        _mm512_sub_epi64(a3, b3),
        _mm512_sub_epi64(a4, b4),
        _mm512_sub_epi64(a5, b5),
        _mm512_sub_epi64(a6, b6),
        _mm512_sub_epi64(a7, b7),
    ])
}

unsafe fn words_srli<const IMM8: u32>(a: SomeWords) -> SomeWords {
    SomeWords([
        _mm512_srli_epi64(a.0[0], IMM8),
        _mm512_srli_epi64(a.0[1], IMM8),
        _mm512_srli_epi64(a.0[2], IMM8),
        _mm512_srli_epi64(a.0[3], IMM8),
        _mm512_srli_epi64(a.0[4], IMM8),
        _mm512_srli_epi64(a.0[5], IMM8),
        _mm512_srli_epi64(a.0[6], IMM8),
        _mm512_srli_epi64(a.0[7], IMM8),
    ])
}

unsafe fn words_and(a: SomeWords, b: SomeWords) -> SomeWords {
    SomeWords([
        _mm512_and_epi32(a.0[0], b.0[0]),
        _mm512_and_epi32(a.0[1], b.0[1]),
        _mm512_and_epi32(a.0[2], b.0[2]),
        _mm512_and_epi32(a.0[3], b.0[3]),
        _mm512_and_epi32(a.0[4], b.0[4]),
        _mm512_and_epi32(a.0[5], b.0[5]),
        _mm512_and_epi32(a.0[6], b.0[6]),
        _mm512_and_epi32(a.0[7], b.0[7]),
    ])
}

#[repr(C, align(64))]
struct Alignedi32s([i32; 16]);

fn criterion_benchmark(c: &mut Criterion) {
    let n_bits = N_TOTAL_BITS;
    let domain = LineDomain::canonic(n_bits);
    let tree = FFTree::preprocess(domain);
    let values = (0..(domain.len()) as i32)
        .array_chunks::<16>()
        .map(Alignedi32s)
        .collect::<Vec<_>>();
    // Get an unsafe mutable pointer to values.
    let unsafe_values = SyncUnsafeCell::new(values);
    c.bench_function("fft", |b| {
        b.iter(|| {
            (0..N_THR).into_par_iter().for_each(|i_thr| {
                // convert pointer back to a mutable slice.
                let values_slice = unsafe { unsafe_values.get().as_mut().unwrap() };
                unsafe { fft3::<0>(&tree, values_slice[..].as_mut_ptr() as *mut i32, i_thr) };
            });
            (0..N_THR).into_par_iter().for_each(|i_thr| {
                // convert pointer back to a mutable slice.
                let values_slice = unsafe { unsafe_values.get().as_mut().unwrap() };
                unsafe { fft3::<3>(&tree, values_slice[..].as_mut_ptr() as *mut i32, i_thr) };
            });
            (0..N_THR).into_par_iter().for_each(|i_thr| {
                // convert pointer back to a mutable slice.
                let values_slice = unsafe { unsafe_values.get().as_mut().unwrap() };
                unsafe { fft3::<6>(&tree, values_slice[..].as_mut_ptr() as *mut i32, i_thr) };
            });
            (0..N_THR).into_par_iter().for_each(|i_thr| {
                // convert pointer back to a mutable slice.
                let values_slice = unsafe { unsafe_values.get().as_mut().unwrap() };
                unsafe { fft3::<9>(&tree, values_slice[..].as_mut_ptr() as *mut i32, i_thr) };
            });
        })
    });
    let val = unsafe_values.into_inner();
    assert!(val[0].0[0] != val[1].0[0]);
}

criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets=criterion_benchmark
);
criterion_main!(benches);
