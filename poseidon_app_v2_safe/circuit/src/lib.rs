use num_traits::One;

// Poseidon computing with LogUp
pub mod poseidon_computing;

// Poseidon parameters
pub const N_STATE: usize = 16;
pub const RATE: usize = 8;
pub const CAPACITY: usize = 8;
pub const N_PARTIAL_ROUNDS: usize = 14;
pub const N_HALF_FULL_ROUNDS: usize = 4;

use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse_coset_to_circle_domain_order;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

/// Generate is_first preprocessed column (1 for first row, 0 for others)
pub fn gen_is_first_column(log_size: u32) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    let n_rows = 1 << log_size;
    let mut col = Col::<SimdBackend, BaseField>::zeros(n_rows);

    // Set first row to 1
    col.set(0, BaseField::from_u32_unchecked(1));

    // Convert to bit-reversed circle domain order
    bit_reverse_coset_to_circle_domain_order(col.as_mut_slice());

    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

pub fn is_first_column_id(log_size: u32) -> PreProcessedColumnId {
    PreProcessedColumnId {
        id: format!("is_first_{}", log_size),
    }
}

/// Generate is_active preprocessed column: 1 for rows 0..=target_element, 0 for rest
pub fn gen_is_active_column(
    log_size: u32,
    target_element: usize,
) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    let n_rows = 1 << log_size;
    let mut col = Col::<SimdBackend, BaseField>::zeros(n_rows);

    // Set 1 for all rows up to and including target_element
    for row in 0..=target_element.min(n_rows - 1) {
        col.set(row, BaseField::one());
    }

    // Convert to bit-reversed circle domain order
    bit_reverse_coset_to_circle_domain_order(col.as_mut_slice());

    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

pub fn is_active_column_id(log_size: u32, target_element: usize) -> PreProcessedColumnId {
    PreProcessedColumnId {
        id: format!("is_active_{}_upto{}", log_size, target_element),
    }
}

/// Generate is_target preprocessed column: 1 ONLY for target_element, 0 for rest (for LogUp)
pub fn gen_is_target_column(
    log_size: u32,
    target_element: usize,
) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    let n_rows = 1 << log_size;
    let mut col = Col::<SimdBackend, BaseField>::zeros(n_rows);

    // Set 1 ONLY for target_element
    if target_element < n_rows {
        col.set(target_element, BaseField::one());
    }

    // Convert to bit-reversed circle domain order
    bit_reverse_coset_to_circle_domain_order(col.as_mut_slice());

    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

pub fn is_target_column_id(log_size: u32, target_element: usize) -> PreProcessedColumnId {
    PreProcessedColumnId {
        id: format!("is_target_{}_row{}", log_size, target_element),
    }
}
