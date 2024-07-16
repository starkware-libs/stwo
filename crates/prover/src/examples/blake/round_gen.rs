use std::simd::u32x16;
use std::vec;

use itertools::{chain, Itertools};
use num_traits::Zero;
use tracing::{span, Level};

use crate::constraint_framework::logup::{LogupTraceGenerator, LookupElements};
use crate::core::backend::simd::column::BaseFieldVec;
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;
use crate::examples::blake::round::blake_round_info;

pub struct BlakeLookupData {
    xor_lookups: Vec<[BaseFieldVec; 3]>,
    round_lookups: [BaseFieldVec; 16 * 3 * 2],
}

pub struct TraceGenerator {
    log_size: u32,
    trace: Vec<BaseFieldVec>,
    xor_lookups: Vec<[BaseFieldVec; 3]>,
    round_lookups: [BaseFieldVec; 16 * 3 * 2],
}
impl TraceGenerator {
    fn new(log_size: u32) -> Self {
        assert!(log_size >= LOG_N_LANES);
        let trace = (0..blake_round_info().mask_offsets[0].len())
            .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
            .collect_vec();
        Self {
            log_size,
            trace,
            xor_lookups: vec![],
            round_lookups: std::array::from_fn(|_| BaseFieldVec::zeros(1 << log_size)),
        }
    }

    fn gen_row(&mut self, vec_row: usize) -> TraceGeneratorRow<'_> {
        TraceGeneratorRow {
            gen: self,
            col_index: 0,
            vec_row,
            xor_lookups_index: 0,
        }
    }
}

struct TraceGeneratorRow<'a> {
    gen: &'a mut TraceGenerator,
    col_index: usize,
    vec_row: usize,
    xor_lookups_index: usize,
}
impl<'a> TraceGeneratorRow<'a> {
    fn append_felt(&mut self, val: u32x16) {
        self.gen.trace[self.col_index].data[self.vec_row] =
            unsafe { PackedBaseField::from_simd_unchecked(val) };
        self.col_index += 1;
    }

    fn append_u32(&mut self, val: u32x16) {
        self.append_felt(val & u32x16::splat(0xffff));
        self.append_felt(val >> 16);
    }

    fn generate(&mut self, mut v: [u32x16; 16], m: [u32x16; 16]) {
        let input_v = v;
        v.iter().for_each(|s| {
            self.append_u32(*s);
        });
        m.iter().for_each(|s| {
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

        chain![input_v.iter(), v.iter(), m.iter()]
            .flat_map(|s| [s & u32x16::splat(0xffff), s >> 16])
            .enumerate()
            .for_each(|(i, val)| {
                self.gen.round_lookups[i].data[self.vec_row] =
                    unsafe { PackedBaseField::from_simd_unchecked(val) }
            });
    }

    fn g(&mut self, _round: u32, v: [&mut u32x16; 4], m0: u32x16, m1: u32x16) {
        let [a, b, c, d] = v;

        *a = self.add3_u32s(*a, *b, m0);
        *d = self.xor_rotr16_u32(*a, *d);
        *c = self.add2_u32s(*c, *d);
        *b = self.xor_rotr_u32(*b, *c, 12);
        *a = self.add3_u32s(*a, *b, m1);
        *d = self.xor_rotr_u32(*a, *d, 8);
        *c = self.add2_u32s(*c, *d);
        *b = self.xor_rotr_u32(*b, *c, 7);
    }

    fn add2_u32s(&mut self, a: u32x16, b: u32x16) -> u32x16 {
        let s = a + b;
        self.append_u32(s);
        s
    }

    fn add3_u32s(&mut self, a: u32x16, b: u32x16, c: u32x16) -> u32x16 {
        let s = a + b + c;
        self.append_u32(s);
        s
    }

    fn xor_rotr_u32(&mut self, a: u32x16, b: u32x16, r: u32) -> u32x16 {
        let c = a ^ b;
        let cr = (c >> r) | (c << (32 - r));

        let (all, alh) = self.split(a & u32x16::splat(0xffff), r);
        let (ahl, ahh) = self.split(a >> 16, r);
        let (bll, blh) = self.split(b & u32x16::splat(0xffff), r);
        let (bhl, bhh) = self.split(b >> 16, r);

        // These also guarantee that all elements are in range.
        let _xorll = self.xor(r, all, bll);
        let _xorlh = self.xor(16 - r, alh, blh);
        let _xorhl = self.xor(r, ahl, bhl);
        let _xorhh = self.xor(16 - r, ahh, bhh);

        cr
    }

    fn xor_rotr16_u32(&mut self, a: u32x16, b: u32x16) -> u32x16 {
        let c = a ^ b;
        let cr = (c >> 16) | (c << 16);

        let (all, alh) = self.split(a & u32x16::splat(0xffff), 8);
        let (ahl, ahh) = self.split(a >> 16, 8);
        let (bll, blh) = self.split(b & u32x16::splat(0xffff), 8);
        let (bhl, bhh) = self.split(b >> 16, 8);

        // These also guarantee that all elements are in range.
        let _xorll = self.xor(8, all, bll);
        let _xorlh = self.xor(8, alh, blh);
        let _xorhl = self.xor(8, ahl, bhl);
        let _xorhh = self.xor(8, ahh, bhh);

        cr
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
        if self.gen.xor_lookups.len() <= self.xor_lookups_index {
            self.gen.xor_lookups.push(std::array::from_fn(|_| {
                BaseFieldVec::zeros(1 << self.gen.log_size)
            }));
        }
        self.gen.xor_lookups[self.xor_lookups_index][0].data[self.vec_row] =
            unsafe { PackedBaseField::from_simd_unchecked(a) };
        self.gen.xor_lookups[self.xor_lookups_index][1].data[self.vec_row] =
            unsafe { PackedBaseField::from_simd_unchecked(b) };
        self.gen.xor_lookups[self.xor_lookups_index][2].data[self.vec_row] =
            unsafe { PackedBaseField::from_simd_unchecked(c) };
        self.xor_lookups_index += 1;
        c
    }
}

#[derive(Copy, Clone, Default)]
pub struct BlakeRoundInput {
    pub v: [u32x16; 16],
    pub m: [u32x16; 16],
}

pub fn gen_trace(
    log_size: u32,
    inputs: &[BlakeRoundInput],
    xor_mults: &mut [u32],
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    BlakeLookupData,
) {
    let mut generator = TraceGenerator::new(log_size);

    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let mut row_gen = generator.gen_row(vec_row);
        let BlakeRoundInput { v, m } = inputs.get(vec_row).copied().unwrap_or_default();
        row_gen.generate(v, m);
        for [a, b, _c] in &generator.xor_lookups {
            let a = a.data[vec_row].into_simd();
            let b = b.data[vec_row].into_simd();
            let idx = (a << 12) + b;
            // TODO: Bug. Index can collide.
            // (Simd::gather_or_default(xor_mults, idx.cast()) + u32x16::splat(1))
            //     .scatter(xor_mults, idx.cast());
            for i in idx.as_array() {
                xor_mults[*i as usize] += 1;
            }
        }
    }
    let domain = CanonicCoset::new(log_size).circle_domain();
    (
        generator
            .trace
            .into_iter()
            .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
            .collect_vec(),
        BlakeLookupData {
            xor_lookups: generator.xor_lookups,
            round_lookups: generator.round_lookups,
        },
    )
}

pub fn gen_interaction_trace(
    log_size: u32,
    lookup_data: BlakeLookupData,
    xor_lookup_elements: &LookupElements,
    round_lookup_elements: &LookupElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let _span = span!(Level::INFO, "Generate interaction trace").entered();
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    for [l0, l1] in lookup_data.xor_lookups.array_chunks::<2>() {
        let mut col_gen = logup_gen.new_col();

        #[allow(clippy::needless_range_loop)]
        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            let p0: PackedSecureField =
                xor_lookup_elements.combine(&l0.each_ref().map(|l| l.data[vec_row]));
            let p1: PackedSecureField =
                xor_lookup_elements.combine(&l1.each_ref().map(|l| l.data[vec_row]));
            col_gen.write_frac(vec_row, p0 + p1, p0 * p1);
        }

        col_gen.finalize_col();
    }

    let mut col_gen = logup_gen.new_col();
    #[allow(clippy::needless_range_loop)]
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let p = round_lookup_elements.combine(
            &lookup_data
                .round_lookups
                .each_ref()
                .map(|l| l.data[vec_row]),
        );
        col_gen.write_frac(vec_row, -PackedSecureField::zero(), p);
    }
    col_gen.finalize_col();

    logup_gen.finalize()
}
