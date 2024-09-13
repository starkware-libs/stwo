#![allow(unused)]
use itertools::Itertools;
use num_traits::One;
use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};

use crate::core::backend::simd::column::BaseColumn;
use crate::core::backend::simd::SimdBackend;
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

pub fn gen_add_1_trace(
    log_size: u32,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let domain = CanonicCoset::new(log_size).circle_domain();
    let mut rng = SmallRng::from_seed([0; 32]);
    let a_col = BaseColumn::from_iter((0..1 << log_size).map(|_| BaseField::from(rng.next_u32())));
    let b_col = BaseColumn::from_iter(
        a_col
            .clone()
            .into_cpu_vec()
            .iter()
            .map(|&x| x + BaseField::one()),
    );
    [a_col, b_col]
        .into_iter()
        .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}

pub fn gen_add_2_trace(
    log_size: u32,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let domain = CanonicCoset::new(log_size).circle_domain();
    let mut rng = SmallRng::from_seed([1; 32]);
    let a_col = BaseColumn::from_iter((0..1 << log_size).map(|_| BaseField::from(rng.next_u32())));
    let b_col = BaseColumn::from_iter(
        a_col
            .clone()
            .into_cpu_vec()
            .iter()
            .map(|&x| x + BaseField::one() + BaseField::one()),
    );
    [a_col, b_col]
        .into_iter()
        .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}

pub fn gen_const_1_trace(
    log_size: u32,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let domain = CanonicCoset::new(log_size).circle_domain();
    let col = BaseColumn::from_iter((0..1 << log_size).map(|_| BaseField::one()));
    [col]
        .into_iter()
        .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}
