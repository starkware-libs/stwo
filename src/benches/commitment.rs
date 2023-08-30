// use prover_research::core::{fft::FFTree, poly::line::LineDomain};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::{
    arch::x86_64::{
        __m512i, _mm512_add_epi32, _mm512_add_epi64, _mm512_and_epi32, _mm512_load_epi32,
        _mm512_mul_epi32, _mm512_or_epi32, _mm512_or_epi64, _mm512_set1_epi64, _mm512_slli_epi64,
        _mm512_srli_epi64, _mm512_store_epi32, _mm512_sub_epi64,
    },
    cell::SyncUnsafeCell,
};

const N_VEC_BITS: usize = 4;
// const N_VEC: usize = 1 << N_VEC_BITS;
const N_THR_BITS: usize = 3;
const N_THR: usize = 1 << N_THR_BITS;
pub const N_TOTAL_BITS: usize = 28;
const N_C_BITS: usize = N_TOTAL_BITS - N_THR_BITS - N_VEC_BITS - 3;

// Index: |TTTTTTTTTT|CCCCCCCCCCCCCCCCCCCCC|VVVVVVVVV|
//         N_THR_BITS         N_C_BITS      N_VEC_BITS
#[inline(never)]
unsafe fn fft_outer(values: *mut i32, twiddles: &[Vec<[Alignedi32s; 4]>], i_thr: usize) {
    let thr_offset = i_thr << (N_C_BITS + N_VEC_BITS + 3);
    for i_h in 0..(1 << N_C_BITS) {
        fft3::<0>(
            values,
            &twiddles.get_unchecked(0)[..],
            thr_offset | (i_h << (N_VEC_BITS + 3)),
        );
        if i_h & ((1 << 3) - 1) != ((1 << 3) - 1) {
            continue;
        }
        let i_h = i_h & (!((1 << 3) - 1));
        fft3::<3>(
            values,
            &twiddles.get_unchecked(1)[..],
            thr_offset | (i_h << (N_VEC_BITS + 3)),
        );
        if i_h & ((1 << 6) - 1) != ((1 << 6) - 1) {
            continue;
        }
        let i_h = i_h & (!((1 << 6) - 1));
        fft3::<6>(
            values,
            &twiddles.get_unchecked(2)[..],
            thr_offset | (i_h << (N_VEC_BITS + 3)),
        );
        if i_h & ((1 << 9) - 1) != ((1 << 9) - 1) {
            continue;
        }
        let i_h = i_h & (!((1 << 9) - 1));
        fft3::<9>(
            values,
            &twiddles.get_unchecked(3)[..],
            thr_offset | (i_h << (N_VEC_BITS + 3)),
        );
    }
}

// Index: |TTTTTTTTTT|HHHHHH|AAA|LLLLLLLLLL|VVVVVVVVV|
//         N_THR_BITS          3   I_OFF    N_VEC_BITS
unsafe fn fft3<const I_OFF: usize>(
    values: *mut i32,
    twiddles: &[[Alignedi32s; 4]],
    c_offset: usize,
) {
    let twiddle_mask = twiddles.len() - 1;
    let a_shift = N_VEC_BITS + I_OFF;
    let mask = _mm512_set1_epi64((1 << 31) - 1);
    let mask1 = _mm512_set1_epi64((1 << 32) - 1);
    let p4 = _mm512_set1_epi64(((1 << 31) - 1) * 3);
    let one = _mm512_set1_epi64(1);
    for i_l in 0..(1 << I_OFF) {
        let index = c_offset | (i_l << (N_VEC_BITS));
        let twids = twiddles.get_unchecked(i_l & twiddle_mask);

        // load
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

        let (t0l, t0h) = split(mask1, _mm512_load_epi32(twids[0].0.as_ptr()));
        let (t1l, t1h) = split(mask1, _mm512_load_epi32(twids[1].0.as_ptr()));
        let (t2l, t2h) = split(mask1, _mm512_load_epi32(twids[2].0.as_ptr()));
        let (t3l, t3h) = split(mask1, _mm512_load_epi32(twids[3].0.as_ptr()));

        // low
        let t4l = psi(one, t0l);
        let t5l = psi(one, t1l);
        let t6l = psi(one, t4l);
        let (val0l, val4l) = butterfly(p4, mask, val0l, val4l, t6l);
        let (val1l, val5l) = butterfly(p4, mask, val1l, val5l, t6l);
        let (val2l, val6l) = butterfly(p4, mask, val2l, val6l, t6l);
        let (val3l, val7l) = butterfly(p4, mask, val3l, val7l, t6l);

        let (val0l, val2l) = butterfly(p4, mask, val0l, val2l, t4l);
        let (val1l, val3l) = butterfly(p4, mask, val1l, val3l, t5l);
        let (val4l, val6l) = butterfly(p4, mask, val4l, val6l, t4l);
        let (val5l, val7l) = butterfly(p4, mask, val5l, val7l, t5l);

        let (val0l, val1l) = butterfly(p4, mask, val0l, val1l, t0l);
        let (val2l, val3l) = butterfly(p4, mask, val2l, val3l, t1l);
        let (val4l, val5l) = butterfly(p4, mask, val4l, val5l, t2l);
        let (val6l, val7l) = butterfly(p4, mask, val6l, val7l, t3l);

        // high
        let t4h = psi(one, t0h);
        let t5h = psi(one, t1h);
        let t6h = psi(one, t4h);

        let (val0h, val4h) = butterfly(p4, mask, val0h, val4h, t6h);
        let (val1h, val5h) = butterfly(p4, mask, val1h, val5h, t6h);
        let (val2h, val6h) = butterfly(p4, mask, val2h, val6h, t6h);
        let (val3h, val7h) = butterfly(p4, mask, val3h, val7h, t6h);

        let (val0h, val2h) = butterfly(p4, mask, val0h, val2h, t4h);
        let (val1h, val3h) = butterfly(p4, mask, val1h, val3h, t5h);
        let (val4h, val6h) = butterfly(p4, mask, val4h, val6h, t4h);
        let (val5h, val7h) = butterfly(p4, mask, val5h, val7h, t5h);

        let (val0h, val1h) = butterfly(p4, mask, val0h, val1h, t0h);
        let (val2h, val3h) = butterfly(p4, mask, val2h, val3h, t1h);
        let (val4h, val5h) = butterfly(p4, mask, val4h, val5h, t2h);
        let (val6h, val7h) = butterfly(p4, mask, val6h, val7h, t3h);

        // store
        _mm512_store_epi32(values.add(index | (0 << a_shift)), combine(val0l, val0h));
        _mm512_store_epi32(values.add(index | (1 << a_shift)), combine(val1l, val1h));
        _mm512_store_epi32(values.add(index | (2 << a_shift)), combine(val2l, val2h));
        _mm512_store_epi32(values.add(index | (3 << a_shift)), combine(val3l, val3h));
        _mm512_store_epi32(values.add(index | (4 << a_shift)), combine(val4l, val4h));
        _mm512_store_epi32(values.add(index | (5 << a_shift)), combine(val5l, val5h));
        _mm512_store_epi32(values.add(index | (6 << a_shift)), combine(val6l, val6h));
        _mm512_store_epi32(values.add(index | (7 << a_shift)), combine(val7l, val7h));
    }
}

//1/(2(1/x)^2-1)=1/y
// x is nonzero.
unsafe fn psi(one: __m512i, v: __m512i) -> __m512i {
    let v = _mm512_mul_epi32(v, v);
    let q = _mm512_add_epi64(_mm512_add_epi64(_mm512_srli_epi64(v, 31), v), one);
    let q = _mm512_add_epi32(_mm512_srli_epi64(q, 31), v);
    _mm512_sub_epi64(_mm512_add_epi32(q, q), one)
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
/// Computes (x>>31)|(x&MASK)
unsafe fn reduce(mask: __m512i, x: __m512i) -> __m512i {
    let x_h = _mm512_srli_epi64(x, 31);
    let x_l = _mm512_and_epi32(x, mask);
    _mm512_or_epi64(x_h, x_l)
}

// 9 cc
unsafe fn butterfly(
    p3: __m512i,
    mask: __m512i, // mask is 2^31-1.
    val0: __m512i,
    val1: __m512i,
    twiddle: __m512i,
) -> (__m512i, __m512i) {
    /*
        Compute val0+twiddle*val1, val0-twiddle*val1.
        Answer is reduced modulo 2^31-1 in an efficient way.
    */
    // m = val1*twiddle
    // twiddle in [0,2^31-1]. val in [0,2^32-1]
    // m in [0, 2^63-3*2^31+1].
    let m = reduce(mask, _mm512_mul_epi32(val1, twiddle));
    // m in [0, 3*2^31-4]

    // val0+m in [0, 4*2^31-5].
    let r0 = reduce(mask, _mm512_add_epi64(val0, m));
    // r0 in [0, 2^31+2].
    // val0+3p-m in [4,4*2^31+2].
    let r1 = reduce(mask, _mm512_sub_epi64(_mm512_add_epi64(val0, p3), m));
    // r1 in [0, 2^31+3].

    (r0, r1)
}

#[repr(C, align(64))]
pub struct Alignedi32s(pub [i32; 16]);

pub fn run_standalone() {
    let (values, twiddles) = prepare();

    // Get an unsafe mutable pointer to values.
    for _ in 0..20 {
        fft_bench(&values, &twiddles);
    }
    let val = values.into_inner();
    assert!(val[0].0[0] != val[1].0[0]);
    println!("val[0] = {:?}", val[0].0[0]);
}

pub fn prepare() -> (SyncUnsafeCell<Vec<Alignedi32s>>, Vec<Vec<[Alignedi32s; 4]>>) {
    let values = (0..(1 << N_TOTAL_BITS))
        .array_chunks::<16>()
        .map(Alignedi32s)
        .collect::<Vec<_>>();
    let mut twiddles = vec![];
    for i in 0..4 {
        twiddles.push(
            (0..(4 * (1 << (N_TOTAL_BITS - 3 * i - 1))))
                .array_chunks::<16>()
                .map(Alignedi32s)
                .array_chunks::<4>()
                .collect::<Vec<_>>(),
        );
    }
    let values = SyncUnsafeCell::new(values);
    (values, twiddles)
}

#[inline(never)]
pub fn fft_bench(
    unsafe_values: &SyncUnsafeCell<Vec<Alignedi32s>>,
    twiddles: &[Vec<[Alignedi32s; 4]>],
) {
    (0..N_THR).into_par_iter().for_each(|i_thr| {
        // convert pointer back to a mutable slice.
        let values_slice = unsafe { unsafe_values.get().as_mut().unwrap() };
        unsafe { fft_outer(values_slice[..].as_mut_ptr() as *mut i32, twiddles, i_thr) };
    });
}
