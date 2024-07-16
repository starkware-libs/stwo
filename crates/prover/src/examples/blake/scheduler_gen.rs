use std::simd::u32x16;

use itertools::{chain, Itertools};
use tracing::{span, Level};

use super::round_gen::BlakeRoundInput;
use super::scheduler::blake_scheduler_info;
use super::{to_felts, N_ROUNDS};
use crate::constraint_framework::logup::{LogupTraceGenerator, LookupElements};
use crate::core::backend::simd::column::BaseFieldVec;
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::{blake2s, SimdBackend};
use crate::core::backend::Column;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

#[derive(Copy, Clone, Default)]
pub struct BlakeInput {
    pub v: [u32x16; 16],
    pub m: [u32x16; 16],
}

pub struct BlakeSchedulerLookupData {
    pub round_lookups: [[BaseFieldVec; 16 * 3 * 2]; N_ROUNDS],
    pub blake_lookups: [BaseFieldVec; 16 * 3 * 2],
}
impl BlakeSchedulerLookupData {
    fn new(log_size: u32) -> Self {
        Self {
            round_lookups: std::array::from_fn(|_| {
                std::array::from_fn(|_| BaseFieldVec::zeros(1 << log_size))
            }),
            blake_lookups: std::array::from_fn(|_| BaseFieldVec::zeros(1 << log_size)),
        }
    }
}

pub fn gen_trace(
    log_size: u32,
    inputs: &[BlakeInput],
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    BlakeSchedulerLookupData,
    Vec<BlakeRoundInput>,
) {
    let mut lookup_data = BlakeSchedulerLookupData::new(log_size);
    let mut round_inputs = Vec::with_capacity(inputs.len() * N_ROUNDS);

    let mut trace = (0..blake_scheduler_info().mask_offsets[0].len())
        .map(|_| BaseFieldVec::zeros(1 << log_size))
        .collect_vec();

    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let mut col_index = 0;

        let mut write_array = |x: [u32x16; 16], col_index: &mut usize| {
            x.iter().for_each(|x| {
                trace[*col_index].data[vec_row] =
                    unsafe { PackedBaseField::from_simd_unchecked(*x & u32x16::splat(0xffff)) };
                trace[*col_index + 1].data[vec_row] =
                    unsafe { PackedBaseField::from_simd_unchecked(*x >> 16) };
                *col_index += 2;
            });
        };

        let BlakeInput { mut v, m } = inputs.get(vec_row).copied().unwrap_or_default();
        let initial_v = v;
        write_array(m, &mut col_index);
        write_array(v, &mut col_index);

        for r in 0..N_ROUNDS {
            let prev_v = v;
            blake2s::round(&mut v, m, r);
            write_array(v, &mut col_index);

            let round_m = blake2s::SIGMA[r].map(|i| m[i as usize]);
            round_inputs.push(BlakeRoundInput {
                v: prev_v,
                m: round_m,
            });

            chain![
                prev_v.iter().copied().flat_map(to_felts),
                v.iter().copied().flat_map(to_felts),
                round_m.iter().copied().flat_map(to_felts)
            ]
            .enumerate()
            .for_each(|(i, val)| lookup_data.round_lookups[r][i].data[vec_row] = val);
        }

        chain![
            initial_v.iter().copied().flat_map(to_felts),
            v.iter().copied().flat_map(to_felts),
            m.iter().copied().flat_map(to_felts)
        ]
        .enumerate()
        .for_each(|(i, val)| lookup_data.blake_lookups[i].data[vec_row] = val);
    }

    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace = trace
        .into_iter()
        .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
        .collect_vec();

    (trace, lookup_data, round_inputs)
}
pub fn gen_interaction_trace(
    log_size: u32,
    lookup_data: BlakeSchedulerLookupData,
    round_lookup_elements: &LookupElements,
    blake_lookup_elements: &LookupElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let _span = span!(Level::INFO, "Generate scheduler interaction trace").entered();

    let mut logup_gen = LogupTraceGenerator::new(log_size);

    for [l0, l1] in lookup_data.round_lookups.array_chunks::<2>() {
        let mut col_gen = logup_gen.new_col();

        #[allow(clippy::needless_range_loop)]
        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            let p0: PackedSecureField =
                round_lookup_elements.combine(&l0.each_ref().map(|l| l.data[vec_row]));
            let p1: PackedSecureField =
                round_lookup_elements.combine(&l1.each_ref().map(|l| l.data[vec_row]));
            #[allow(clippy::eq_op)]
            col_gen.write_frac(vec_row, p0 - p0, p0 * p1);
        }

        col_gen.finalize_col();
    }

    // Last pair.
    assert_eq!(N_ROUNDS % 2, 1);
    let mut col_gen = logup_gen.new_col();
    #[allow(clippy::needless_range_loop)]
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let p0: PackedSecureField = round_lookup_elements.combine(
            &lookup_data.round_lookups[N_ROUNDS - 1]
                .each_ref()
                .map(|l| l.data[vec_row]),
        );
        let p1: PackedSecureField = blake_lookup_elements.combine(
            &lookup_data
                .blake_lookups
                .each_ref()
                .map(|l| l.data[vec_row]),
        );
        #[allow(clippy::eq_op)]
        col_gen.write_frac(vec_row, p1 - p1, p0 * p1);
    }
    col_gen.finalize_col();

    logup_gen.finalize()
}
