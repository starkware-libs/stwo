//! Poseidon hash with component-based architecture (like stark_appv2_safe)
//!
//! This module implements Poseidon hash using separate Computing and Scheduler components.

use stwo_constraint_framework::relation;

mod computing;
mod scheduler;
mod trace_gen;

pub use computing::{PoseidonComputingComponent, PoseidonComputingEval};
pub use scheduler::{PoseidonSchedulerComponent, PoseidonSchedulerEval};
pub use trace_gen::{
    gen_computing_trace, gen_scheduler_trace,
    gen_computing_interaction_trace, gen_scheduler_interaction_trace,
    LookupData,
};

use stwo::core::channel::{Blake2sChannel, Channel};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse_coset_to_circle_domain_order;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::core::pcs::TreeVec;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

// Poseidon constants
pub const N_STATE: usize = 16;
pub const RATE: usize = 8; // First 8 elements absorb message
pub const CAPACITY: usize = 8; // Last 8 elements for security
pub const N_PARTIAL_ROUNDS: usize = 14;
pub const N_HALF_FULL_ROUNDS: usize = 4;
pub const FULL_ROUNDS: usize = 2 * N_HALF_FULL_ROUNDS;
// Columns: 8 message + 16 initial_state + intermediate_states + 16 final_state
pub const N_COLUMNS: usize = RATE + N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS + N_STATE;
pub const LOG_CONSTRAINT_DEGREE: u32 = 2;
// TODO(shahars): Use poseidon's real constants.
pub const EXTERNAL_ROUND_CONSTS: [[BaseField; N_STATE]; 2 * N_HALF_FULL_ROUNDS] =
    [[BaseField::from_u32_unchecked(1234); N_STATE]; 2 * N_HALF_FULL_ROUNDS];
pub const INTERNAL_ROUND_CONSTS: [BaseField; N_PARTIAL_ROUNDS] =
    [BaseField::from_u32_unchecked(1234); N_PARTIAL_ROUNDS];

relation!(PoseidonElements, N_STATE);

use std::ops::{Add, AddAssign, Mul, Sub};
use stwo::core::fields::FieldExpOps;

#[inline(always)]
/// Applies the M4 MDS matrix described in <https://eprint.iacr.org/2023/323.pdf> 5.1.
pub fn apply_m4<F>(x: [F; 4]) -> [F; 4]
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    let t0 = x[0].clone() + x[1].clone();
    let t02 = t0.clone() + t0.clone();
    let t1 = x[2].clone() + x[3].clone();
    let t12 = t1.clone() + t1.clone();
    let t2 = x[1].clone() + x[1].clone() + t1.clone();
    let t3 = x[3].clone() + x[3].clone() + t0.clone();
    let t4 = t12.clone() + t12.clone() + t3.clone();
    let t5 = t02.clone() + t02.clone() + t2.clone();
    let t6 = t3.clone() + t5.clone();
    let t7 = t2.clone() + t4.clone();
    [t6, t5, t7, t4]
}

/// Applies the external round matrix.
/// See <https://eprint.iacr.org/2023/323.pdf> 5.1 and Appendix B.
pub fn apply_external_round_matrix<F>(state: &mut [F; 16])
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    // Applies circ(2M4, M4, M4, M4).
    for i in 0..4 {
        [
            state[4 * i],
            state[4 * i + 1],
            state[4 * i + 2],
            state[4 * i + 3],
        ] = apply_m4([
            state[4 * i].clone(),
            state[4 * i + 1].clone(),
            state[4 * i + 2].clone(),
            state[4 * i + 3].clone(),
        ]);
    }
    for j in 0..4 {
        let s =
            state[j].clone() + state[j + 4].clone() + state[j + 8].clone() + state[j + 12].clone();
        for i in 0..4 {
            state[4 * i + j] += s.clone();
        }
    }
}

// Applies the internal round matrix.
//   mu_i = 2^{i+1} + 1.
// See <https://eprint.iacr.org/2023/323.pdf> 5.2.
pub fn apply_internal_round_matrix<F>(state: &mut [F; 16])
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    let sum = state[1..]
        .iter()
        .cloned()
        .fold(state[0].clone(), |acc, s| acc + s);
    state.iter_mut().enumerate().for_each(|(i, s)| {
        *s = s.clone() * BaseField::from_u32_unchecked(1 << (i + 1)) + sum.clone();
    });
}

pub fn pow5<F: FieldExpOps>(x: F) -> F {
    let x2 = x.clone() * x.clone();
    let x4 = x2.clone() * x2.clone();
    x4 * x.clone()
}

/// Generate is_first preprocessed column: 1 for first row, 0 for rest
pub fn gen_is_first_column(log_size: u32) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    let n_rows = 1 << log_size;
    let mut col = Col::<SimdBackend, BaseField>::zeros(n_rows);
    col.set(0, BaseField::from_u32_unchecked(1));
    bit_reverse_coset_to_circle_domain_order(col.as_mut_slice());
    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

pub fn is_first_column_id(log_size: u32) -> PreProcessedColumnId {
    PreProcessedColumnId {
        id: format!("is_first_{}", log_size),
    }
}

/// Generate is_active preprocessed column: 1 for rows with messages, 0 for padding
pub fn gen_is_active_column(log_size: u32, n_messages: usize) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    let n_rows = 1 << log_size;
    let mut col = Col::<SimdBackend, BaseField>::zeros(n_rows);

    for row in 0..n_messages.min(n_rows) {
        col.set(row, BaseField::from_u32_unchecked(1));
    }

    bit_reverse_coset_to_circle_domain_order(col.as_mut_slice());
    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

pub fn is_active_column_id(log_size: u32, n_messages: usize) -> PreProcessedColumnId {
    PreProcessedColumnId {
        id: format!("is_active_{}_{}", log_size, n_messages),
    }
}

/// Statement 0: Component configuration (log_size, n_messages)
/// Mixed into channel before drawing PoseidonElements
#[derive(Clone, Copy, Debug)]
pub struct PoseidonStatement0 {
    pub log_size: u32,
    pub n_messages: usize,
}

impl PoseidonStatement0 {
    pub fn mix_into(&self, channel: &mut Blake2sChannel) {
        channel.mix_u64(self.log_size as u64);
        channel.mix_u64(self.n_messages as u64);
    }

    /// Returns log sizes for all trees (preprocessed, main, interaction)
    pub fn log_sizes(&self) -> TreeVec<Vec<u32>> {
        TreeVec(vec![
            // Tree 0: Preprocessed (2 columns: is_first, is_active)
            vec![self.log_size; 2],
            // Tree 1: Main traces (TODO: determine actual number of columns)
            vec![self.log_size; 8],  // Placeholder: 8 message columns
            // Tree 2: Interaction traces (TODO: determine actual number of columns)
            vec![self.log_size; 4],  // Placeholder: 4 columns per SecureColumn
        ])
    }
}

/// Statement 1: LogUp claimed sums
/// Mixed into channel after drawing PoseidonElements
#[derive(Clone, Copy, Debug)]
pub struct PoseidonStatement1 {
    pub claimed_sum_computing: SecureField,
    pub claimed_sum_scheduler: SecureField,
}

impl PoseidonStatement1 {
    pub fn mix_into(&self, channel: &mut Blake2sChannel) {
        channel.mix_felts(&[
            self.claimed_sum_computing,
            self.claimed_sum_scheduler,
        ]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poseidon_components_basic() {
        let log_size = 7; // 128 rows
        let n_messages = 2;

        // Generate preprocessed columns
        let is_first_col = gen_is_first_column(log_size);
        let is_active_col = gen_is_active_column(log_size, n_messages);

        println!("✅ Basic component structure test passed!");
        println!("   Preprocessed columns: is_first={}, is_active={}",
                 is_first_col.values.len(), is_active_col.values.len());
    }

    #[test]
    fn test_trace_generation_with_optimization() {
        println!("\n========================================");
        println!("TEST: Trace Generation with 2 messages out of 128 rows");
        println!("========================================");

        let log_size = 7; // 128 rows
        let n_messages = 2;

        // Create messages
        let messages: Vec<[BaseField; RATE]> = (0..n_messages)
            .map(|i| std::array::from_fn(|j| BaseField::from_u32_unchecked((i * RATE + j) as u32)))
            .collect();

        println!("Generating trace for {} messages out of {} rows...", n_messages, 1 << log_size);

        // Generate computing trace
        let (trace, lookup_data) = gen_computing_trace(log_size, messages.clone());

        println!("✅ Trace generated successfully!");
        println!("   Number of trace columns: {}", trace.len());
        println!("   Expected columns (N_COLUMNS): {}", N_COLUMNS);

        assert_eq!(trace.len(), N_COLUMNS, "Should have correct number of columns");

        // Generate interaction trace
        let lookup_elements = PoseidonElements::dummy();
        let (interaction_trace, claimed_sum) =
            gen_computing_interaction_trace(log_size, lookup_data, &lookup_elements, n_messages);

        println!("✅ Interaction trace generated!");
        println!("   Claimed sum: {:?}", claimed_sum);
        println!("   Interaction columns: {}", interaction_trace.len());

        // Generate scheduler trace
        let scheduler_trace = gen_scheduler_trace(log_size, n_messages);
        println!("✅ Scheduler trace generated!");
        println!("   Scheduler columns: {}", scheduler_trace.len());

        println!("\n✅ All trace generation tests passed!");
    }
}
