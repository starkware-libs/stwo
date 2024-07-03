use std::ops::{Add, AddAssign, Mul, Sub};
use std::simd::u32x16;

use itertools::Itertools;

use super::Fu32;
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::{blake2s, SimdBackend};
use crate::core::backend::{Col, Column};
use crate::core::fields::m31::BaseField;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::vcs::blake2s_ref::SIGMA;
use crate::core::ColumnVec;

const N_ROUNDS: usize = 10;
const N_COLUMNS: usize = 16 * (N_ROUNDS + 1) + 16;

pub struct BlakeSchedulerComponent {
    pub log_size: u32,
}

pub trait SchedulerEval {
    type F: FieldExpOps
        + Copy
        + AddAssign<Self::F>
        + Add<Self::F, Output = Self::F>
        + Sub<Self::F, Output = Self::F>
        + Mul<BaseField, Output = Self::F>;

    fn next_mask(&mut self) -> Self::F;

    fn next_u32(&mut self) -> Fu32<Self::F> {
        let l = self.next_mask();
        let h = self.next_mask();
        Fu32 { l, h }
    }
    fn eval(&mut self) {
        let messages: [Fu32<Self::F>; 16] = std::array::from_fn(|_| self.next_u32());
        let states: [[Fu32<Self::F>; 16]; N_ROUNDS + 1] =
            std::array::from_fn(|_| std::array::from_fn(|_| self.next_u32()));

        // Schedule.
        for i in 0..N_ROUNDS {
            let _input_state = &states[i];
            let _output_state = &states[i + 1];
            let _round_messages: [Fu32<Self::F>; 16] =
                std::array::from_fn(|j| messages[SIGMA[i][j] as usize]);
            // Use triplet in round lookup.
        }

        let _input_state = &states[0];
        let _output_state = &states[N_ROUNDS];
        // Yield triplet in hash lookup.
    }
}

pub fn gen_trace(
    log_size: u32,
    h: Vec<[u32; 16]>,
    m: Vec<[u32; 16]>,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    assert_eq!(h.len(), 1 << log_size);
    assert_eq!(m.len(), 1 << log_size);
    assert!(log_size >= LOG_N_LANES);

    let mut trace = (0..N_COLUMNS)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
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

        let messages: [_; 16] = std::array::from_fn(|i| {
            u32x16::from_array(std::array::from_fn(|j| m[(vec_row << LOG_N_LANES) + j][i]))
        });
        write_array(messages, &mut col_index);

        let mut state: [_; 16] = std::array::from_fn(|i| {
            u32x16::from_array(std::array::from_fn(|j| h[(vec_row << LOG_N_LANES) + j][i]))
        });
        write_array(state, &mut col_index);

        for i in 0..N_ROUNDS {
            blake2s::round(&mut state, messages, i);
            write_array(state, &mut col_index);
        }
    }

    let domain = CanonicCoset::new(log_size).circle_domain();
    trace
        .into_iter()
        .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}
