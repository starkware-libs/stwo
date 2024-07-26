//! Returns the computational cost of proving Poseidon with AIR vs GKR.
//!
//! The Grand Product has been removed from the calculation because Stwo is already optimized to use
//! GKR for Grand Product.

use constraint_counter::poseidon_constraint_cost;
use cost_counter::CostCount;
use stwo_prover::core::prover::LOG_BLOWUP_FACTOR;
use stwo_prover::examples::poseidon::{FULL_ROUNDS, LOG_EXPAND, N_PARTIAL_ROUNDS, N_STATE};

mod constraint_counter;
mod cost_counter;

fn poseidon_air_cost(log_n_instances: u32) -> CostCount {
    /// Number of base trace columns:
    /// * 16x trace columns (inputs)
    /// * 16x trace columns (full round 0)
    /// * 16x trace columns (full round 1)
    /// * 16x trace columns (full round 2)
    /// * 16x trace columns (full round 3)
    /// * 14x trace columns (partial rounds)
    /// * 16x trace columns (full round 4)
    /// * 16x trace columns (full round 5)
    /// * 16x trace columns (full round 6)
    /// * 16x trace columns (outputs)
    const N_TRACE_COLUMNS: usize = N_STATE + FULL_ROUNDS * N_STATE + N_PARTIAL_ROUNDS;

    let n_instances = 1 << log_n_instances;

    // Base trace commitment.
    let base_trace_commitment_cost =
        CostCount::commitment(log_n_instances, LOG_BLOWUP_FACTOR, N_TRACE_COLUMNS);

    // Composition polynomial generation and commitment.
    let log_composition_degree_bound = log_n_instances + LOG_EXPAND;
    let constraint_trace_ffts_cost = CostCount::fft(log_composition_degree_bound) * N_TRACE_COLUMNS;
    let constraint_eval_cost = poseidon_constraint_cost() * (1 << log_composition_degree_bound);
    let composition_trace_commitment_cost =
        CostCount::commitment(log_composition_degree_bound, LOG_BLOWUP_FACTOR, 1);

    // DEEP composition polynomial generation and commitment.
    let trace_ood_eval_cost = CostCount::base_eval_ext(n_instances) * N_TRACE_COLUMNS;
    let trace_ood_numer_rand_lin_comb =
        CostCount::random_linear_combination_base_values(N_TRACE_COLUMNS) * n_instances;
    let trace_quotient_cost = CostCount::ext_quotient_ext(n_instances);
    let composition_ood_eval_cost = CostCount::ext_eval_ext(1 << log_composition_degree_bound);
    // TODO: Can do degree decomposition trick and merge with quotients above.
    let composition_ood_quotient_cost =
        CostCount::ext_quotient_ext(1 << log_composition_degree_bound);

    base_trace_commitment_cost
        + constraint_trace_ffts_cost
        + constraint_eval_cost
        + composition_trace_commitment_cost
        + trace_ood_eval_cost
        + trace_ood_numer_rand_lin_comb
        + trace_quotient_cost
        + composition_ood_eval_cost
        + composition_ood_quotient_cost
}

#[allow(unreachable_code)]
fn poseidon_gkr_cost(log_n_instances: u32) -> CostCount {
    /// STARK base trace:
    /// * 16x trace columns (inputs)
    const N_TRACE_COLUMNS: usize = N_STATE;

    // Composition polynomial commitment.
    let base_trace_commitment_cost =
        CostCount::commitment(log_n_instances, LOG_BLOWUP_FACTOR, N_TRACE_COLUMNS);

    // Poseidon GKR circuit cost.
    let full_rounds_cost = CostCount::poseidon_gkr_full_round(log_n_instances) * 8;
    let partial_rounds_cost = CostCount::poseidon_gkr_partial_round(log_n_instances) * 14;

    // Note that Poseidon with all GKR needs multilinear evaluation IOPs for:
    // 1. Grand product terms column `z-in0-..-in15*alpha^15-out0*alpha^16-..-out15*alpha^31`
    //    evaluated at multilinear point r0.
    // 2. Input state columns evaluated at multilinear point r1.
    // Whereas Poseidon with AIR and GKR grand product only needs to do (1).
    // TODO: Add costs for this.

    // DEEP composition polynomial
    let n_instances = 1 << log_n_instances;
    let trace_ood_eval_cost = CostCount::base_eval_ext(n_instances) * N_TRACE_COLUMNS;
    let trace_ood_numer_rand_lin_comb =
        CostCount::random_linear_combination_base_values(N_TRACE_COLUMNS) * n_instances;
    let trace_quotient_cost = CostCount::ext_quotient_ext(n_instances);
    // TODO: Added quotients for extra multilinear eval at point.

    base_trace_commitment_cost
        + full_rounds_cost
        + partial_rounds_cost
        + trace_ood_eval_cost
        + trace_ood_numer_rand_lin_comb
        + trace_quotient_cost
}

fn main() {
    const LOG_N_INSTANCES: u32 = 16;
    println!("AIR cost: {:#?}", poseidon_air_cost(LOG_N_INSTANCES));
    println!("AIR cost: {:#?}", poseidon_gkr_cost(LOG_N_INSTANCES));
}
