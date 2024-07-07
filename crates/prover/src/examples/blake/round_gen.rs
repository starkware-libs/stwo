use std::simd::{u32x16, Simd};

use bytemuck::cast_slice_mut;
use itertools::Itertools;
use num_traits::{One, Zero};
use tracing::{span, Level};

use super::lookup::LookupElements;
use crate::core::backend::simd::column::{BaseFieldVec, SecureFieldVec};
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES, N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column, ColumnOps};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::{SecureColumn, SECURE_EXTENSION_DEGREE};
use crate::core::fields::{FieldExpOps, FieldOps};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, SecureEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::{
    bit_reverse_index, circle_domain_order_to_coset_order, shifted_secure_combination,
};
use crate::core::ColumnVec;
use crate::examples::blake::round::blake_counter;

pub struct BlakeTraceGenerator {
    log_size: u32,
    trace: Vec<BaseFieldVec>,
    col_index: usize,
    lookup_exprs: Vec<BaseFieldVec>,
    lookup_index: usize,
    vec_row: usize,
}
impl BlakeTraceGenerator {
    pub fn gen_trace(
        log_size: u32,
    ) -> (
        ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        Vec<BaseFieldVec>,
    ) {
        assert!(log_size >= LOG_N_LANES);
        let trace = (0..blake_counter().mask_offsets[0].len())
            .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
            .collect_vec();
        let lookup_exprs = vec![];

        let mut gen = Self {
            log_size,
            trace,
            col_index: 0,
            lookup_exprs,
            lookup_index: 0,
            vec_row: 0,
        };
        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            gen.vec_row = vec_row;
            gen.col_index = 0;
            gen.lookup_index = 0;

            let v: [u32x16; 16] = std::array::from_fn(|state_i| {
                Simd::from_array(std::array::from_fn(|i| (vec_row * 16 + i + state_i) as u32))
            });
            let m: [u32x16; 16] = std::array::from_fn(|state_i| {
                Simd::from_array(std::array::from_fn(|i| {
                    (vec_row * 16 + i + state_i + 100) as u32
                }))
            });

            gen.gen_row(v, m);
        }
        let domain = CanonicCoset::new(log_size).circle_domain();
        (
            gen.trace
                .into_iter()
                .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
                .collect_vec(),
            gen.lookup_exprs,
        )
    }

    fn append_felt(&mut self, val: u32x16) {
        self.trace[self.col_index].data[self.vec_row] =
            unsafe { PackedBaseField::from_simd_unchecked(val) };
        self.col_index += 1;
    }

    fn append_u32(&mut self, val: [u32x16; 2]) {
        self.append_felt(val[0]);
        self.append_felt(val[1]);
    }

    fn gen_row(&mut self, v: [u32x16; 16], m: [u32x16; 16]) {
        let mut v = v.map(|state| [state & u32x16::splat(0xffff), state >> 16]);
        let m = m.map(|state| [state & u32x16::splat(0xffff), state >> 16]);
        v.iter().for_each(|s| {
            self.append_u32(*s);
        });

        self.g(0, v.get_many_mut([0, 4, 8, 12]).unwrap(), m[0], m[1]);
        self.g(1, v.get_many_mut([1, 5, 9, 13]).unwrap(), m[2], m[3]);
        self.g(2, v.get_many_mut([2, 6, 10, 14]).unwrap(), m[4], m[5]);
        self.g(3, v.get_many_mut([3, 7, 11, 15]).unwrap(), m[6], m[7]);
        self.g(4, v.get_many_mut([0, 5, 10, 15]).unwrap(), m[8], m[9]);
        self.g(5, v.get_many_mut([1, 6, 11, 12]).unwrap(), m[10], m[11]);
        self.g(6, v.get_many_mut([2, 7, 8, 13]).unwrap(), m[12], m[13]);
        self.g(7, v.get_many_mut([3, 4, 9, 14]).unwrap(), m[14], m[15]);
    }

    fn g(&mut self, _round: u32, v: [&mut [u32x16; 2]; 4], m0: [u32x16; 2], m1: [u32x16; 2]) {
        let [a, b, c, d] = v;
        self.append_u32(m0);
        self.append_u32(m1);

        *a = self.add3_u32s(*a, *b, m0);
        *d = self.xor_rotr16_u32(*a, *d);
        *c = self.add2_u32s(*c, *d);
        *b = self.xor_rotr_u32(*b, *c, 12);
        *a = self.add3_u32s(*a, *b, m1);
        *d = self.xor_rotr_u32(*a, *d, 8);
        *c = self.add2_u32s(*c, *d);
        *b = self.xor_rotr_u32(*b, *c, 7);
    }

    fn add2_u32s(&mut self, a: [u32x16; 2], b: [u32x16; 2]) -> [u32x16; 2] {
        let sl = a[0] + b[0];
        let carryl = sl >> 16;
        self.append_felt(carryl);

        let sh = a[1] + b[1] + carryl;
        let carryh = sh >> 16;
        self.append_felt(carryh);

        [sl & u32x16::splat(0xffff), sh & u32x16::splat(0xffff)]
    }

    fn add3_u32s(&mut self, a: [u32x16; 2], b: [u32x16; 2], c: [u32x16; 2]) -> [u32x16; 2] {
        let sl = a[0] + b[0] + c[0];
        let carryl = sl >> 16;
        self.append_felt(carryl);

        let sh = a[1] + b[1] + c[1] + carryl;
        let carryh = sh >> 16;
        self.append_felt(carryh);

        [sl & u32x16::splat(0xffff), sh & u32x16::splat(0xffff)]
    }

    fn xor_rotr_u32(&mut self, a: [u32x16; 2], b: [u32x16; 2], r: u32) -> [u32x16; 2] {
        let (all, alh) = self.split(a[0], r);
        let (ahl, ahh) = self.split(a[1], r);
        let (bll, blh) = self.split(b[0], r);
        let (bhl, bhh) = self.split(b[1], r);

        // These also guarantee that all elements are in range.
        let xorll = self.xor(r, all, bll);
        let xorlh = self.xor(16 - r, alh, blh);
        let xorhl = self.xor(r, ahl, bhl);
        let xorhh = self.xor(16 - r, ahh, bhh);

        [(xorhl << (16 - r)) + xorlh, (xorll << (16 - r)) + xorhh]
    }

    fn xor_rotr16_u32(&mut self, a: [u32x16; 2], b: [u32x16; 2]) -> [u32x16; 2] {
        let (all, alh) = self.split(a[0], 8);
        let (ahl, ahh) = self.split(a[1], 8);
        let (bll, blh) = self.split(b[0], 8);
        let (bhl, bhh) = self.split(b[1], 8);

        // These also guarantee that all elements are in range.
        let xorll = self.xor(8, all, bll);
        let xorlh = self.xor(8, alh, blh);
        let xorhl = self.xor(8, ahl, bhl);
        let xorhh = self.xor(8, ahh, bhh);

        [(xorhh << 8) + xorhl, (xorlh << 8) + xorll]
    }

    fn split(&mut self, a: u32x16, r: u32) -> (u32x16, u32x16) {
        let h = a >> r;
        let l = a & u32x16::splat((1 << r) - 1);
        self.append_felt(h);
        (l, h)
    }

    fn xor(&mut self, _w: u32, a: u32x16, b: u32x16) -> u32x16 {
        let c = a ^ b;
        self.append_felt(c);
        if self.lookup_exprs.len() < self.lookup_index + 3 {
            self.lookup_exprs.resize_with(self.lookup_index + 3, || {
                BaseFieldVec::zeros(1 << self.log_size)
            });
        }
        self.lookup_exprs[self.lookup_index].data[self.vec_row] =
            unsafe { PackedBaseField::from_simd_unchecked(a) };
        self.lookup_exprs[self.lookup_index + 1].data[self.vec_row] =
            unsafe { PackedBaseField::from_simd_unchecked(b) };
        self.lookup_exprs[self.lookup_index + 2].data[self.vec_row] =
            unsafe { PackedBaseField::from_simd_unchecked(c) };
        self.lookup_index += 3;
        c
    }
}

fn set_vec(vec: &mut BaseFieldVec, i: usize, val: BaseField) {
    let el = vec.data[i >> LOG_N_LANES];
    let mut arr = el.to_array();
    arr[i & (N_LANES - 1)] = val;
    vec.data[i >> LOG_N_LANES] = PackedBaseField::from_array(arr);
}

fn set_secure_vec(vec: &mut SecureFieldVec, i: usize, val: SecureField) {
    let el = vec.data[i >> LOG_N_LANES];
    let mut arr = el.to_array();
    arr[i & (N_LANES - 1)] = val;
    vec.data[i >> LOG_N_LANES] = PackedSecureField::from_array(arr);
}

fn set_secure_col(vec: &mut SecureColumn<SimdBackend>, i: usize, val: SecureField) {
    let vals = val.to_m31_array();
    #[allow(clippy::needless_range_loop)]
    for j in 0..SECURE_EXTENSION_DEGREE {
        set_vec(&mut vec.columns[j], i, vals[j]);
    }
}

pub fn gen_interaction_trace(
    log_size: u32,
    mut lookup_exprs: Vec<BaseFieldVec>,
    lookup_elements: LookupElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let span = span!(Level::INFO, "Generate interaction trace").entered();
    let LookupElements { z, alpha } = lookup_elements;
    let alpha = PackedSecureField::broadcast(alpha);
    let z = PackedSecureField::broadcast(z);
    assert_eq!(lookup_exprs.len() % 6, 0);
    let mut trace =
        vec![SecureColumn::<SimdBackend>::zeros(1 << log_size); lookup_exprs.len() / 3 / 2];

    let mut temp_denom = SecureFieldVec::zeros(1 << log_size);
    let mut temp_denom_inv = SecureFieldVec::zeros(1 << log_size);

    for (i, [l0, l1]) in lookup_exprs
        .iter()
        .array_chunks::<3>()
        .array_chunks::<2>()
        .enumerate()
    {
        // First row.
        #[allow(clippy::needless_range_loop)]
        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            let p0 = shifted_secure_combination(&l0.map(|l| l.data[vec_row]), alpha, z);
            let p1 = shifted_secure_combination(&l1.map(|l| l.data[vec_row]), alpha, z);
            let mut num = p0 + p1;
            let mut denom = p0 * p1;
            unsafe { trace[i].set_packed(vec_row, num) };
            temp_denom.data[vec_row] = denom;
        }

        FieldExpOps::batch_inverse(&temp_denom.data, &mut temp_denom_inv.data);

        // Multiply
        #[allow(clippy::needless_range_loop)]
        for row in 0..(1 << (log_size - LOG_N_LANES)) {
            unsafe {
                let value = trace[i].packed_at(row) * temp_denom_inv.data[row];
                trace[i].set_packed(row, value)
            };
        }
    }

    // Cumulative sum on the last column.
    let span1 = span!(Level::INFO, "Cumulative").entered();
    // TODO: optimize.
    let mut cur = SecureField::zero();
    let col = trace.last_mut().unwrap();
    #[allow(clippy::needless_range_loop)]
    for i in 0..(1 << log_size) {
        let index = if i & 1 == 0 {
            i / 2
        } else {
            (1 << (log_size - 1)) + ((1 << log_size) - 1 - i) / 2
        };
        let index = bit_reverse_index(index, log_size);
        cur += col.at(index);
        set_secure_col(col, index, cur);
    }

    let claimed_xor_sum = cur;

    let trace = trace
        .into_iter()
        .flat_map(|eval| {
            eval.columns.map(|c| {
                CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(
                    CanonicCoset::new(log_size).circle_domain(),
                    c,
                )
            })
        })
        .collect_vec();
    (trace, claimed_xor_sum)
}

pub fn get_constant_trace(
    log_size: u32,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let mut col = BaseFieldVec::zeros(1 << log_size);
    set_vec(&mut col, 0, BaseField::one());
    vec![CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(
        CanonicCoset::new(log_size).circle_domain(),
        col,
    )]
}
