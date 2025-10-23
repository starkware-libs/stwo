//! Dynamic Scheduler + Poseidon Component Composition Example
//!
//! This example demonstrates VARIABLE-LENGTH composition where:
//! - SchedulerComponent: Schedules VARIABLE number of hash computations (1 to MAX_CALLS)
//! - PoseidonComponent: Performs actual Poseidon2 hashing for active calls
//! - Uses is_active column to mark which rows are real vs padding
//!
//! Communication via LogUp lookup arguments over 16-element states.

use itertools::chain;
use num_traits::One;
use stwo::core::air::Component;
use stwo::core::channel::Channel;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::TreeVec;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::ComponentProver;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, InfoEvaluator, RelationEntry,
    TraceLocationAllocator,
};

// Import Poseidon utilities
use crate::poseidon::{PoseidonComponent, PoseidonElements, PoseidonEval};

const CONSTRAINT_EVAL_BLOWUP_FACTOR: u32 = 1;
const N_STATE: usize = 16; // Poseidon2 state size

// Dynamic scheduler configuration
#[allow(dead_code)]
const MAX_CALLS: usize = 1024; // Maximum number of Poseidon calls supported
#[allow(dead_code)]
const LOG_MAX_CALLS: u32 = 10; // log2(1024) = 10

// We no longer use fixed instances per row - each row is ONE call
// Scheduler structure per row: [is_active (1), input_state (16), output_state (16)] = 33 columns total

#[cfg(test)]
const POSEIDON_LOG_EXPAND: u32 = 2; // Poseidon uses LOG_EXPAND = 2

pub type SchedulerComponent = FrameworkComponent<SchedulerEval>;

/// Container for both components
pub struct Components {
    scheduler_component: SchedulerComponent,
    poseidon_component: PoseidonComponent,
}

impl Components {
    pub fn new(
        statement0: &ComponentsStatement0,
        lookup_elements: &PoseidonElements,
        statement1: &ComponentsStatement1,
    ) -> Self {
        let tree_span_provider =
            &mut TraceLocationAllocator::new_with_preprocessed_columns(&vec![]);

        // IMPORTANT: Create components in same order as trace commitment!
        // Poseidon first, then Scheduler
        let poseidon_component = PoseidonComponent::new(
            tree_span_provider,
            PoseidonEval {
                log_n_rows: statement0.poseidon_log_size,
                lookup_elements: lookup_elements.clone(),
                claimed_sum: statement1.poseidon_claimed_sum,
            },
            statement1.poseidon_claimed_sum,
        );

        let scheduler_component = SchedulerComponent::new(
            tree_span_provider,
            SchedulerEval {
                log_size: statement0.scheduler_log_size,
                lookup_elements: lookup_elements.clone(),
            },
            statement1.scheduler_claimed_sum,
        );

        Self {
            scheduler_component,
            poseidon_component,
        }
    }

    pub fn components(&self) -> Vec<&dyn Component> {
        chain![[
            &self.poseidon_component as &dyn Component,
            &self.scheduler_component as &dyn Component
        ]]
        .collect()
    }

    pub fn component_provers(&self) -> Vec<&dyn ComponentProver<SimdBackend>> {
        chain![[
            &self.poseidon_component as &dyn ComponentProver<SimdBackend>,
            &self.scheduler_component as &dyn ComponentProver<SimdBackend>
        ]]
        .collect()
    }
}

pub struct ComponentsStatement0 {
    scheduler_log_size: u32,  // Scheduler trace size
    poseidon_log_size: u32,   // Poseidon trace size
    num_calls: usize,          // DYNAMIC: How many actual Poseidon calls (public input)
}

impl ComponentsStatement0 {
    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_u64(self.scheduler_log_size as u64);
        channel.mix_u64(self.poseidon_log_size as u64);
        channel.mix_u64(self.num_calls as u64); // Mix num_calls into Fiat-Shamir
    }

    pub fn log_sizes(&self) -> TreeVec<Vec<u32>> {
        let mut log_sizes = vec![];

        // IMPORTANT: Same order as Components::new()!
        // Poseidon first (uses poseidon_log_size)
        log_sizes.push(
            poseidon_info()
                .mask_offsets
                .as_cols_ref()
                .map_cols(|_| self.poseidon_log_size),
        );

        // Scheduler second (uses scheduler_log_size)
        log_sizes.push(
            scheduler_info()
                .mask_offsets
                .as_cols_ref()
                .map_cols(|_| self.scheduler_log_size),
        );

        TreeVec::concat_cols(log_sizes.into_iter())
    }
}

pub struct ComponentsStatement1 {
    scheduler_claimed_sum: SecureField,
    poseidon_claimed_sum: SecureField,
}

impl ComponentsStatement1 {
    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_felts(&[self.scheduler_claimed_sum, self.poseidon_claimed_sum]);
    }
}

fn scheduler_info() -> InfoEvaluator {
    let component = SchedulerEval {
        log_size: 1,
        lookup_elements: PoseidonElements::dummy(),
    };
    component.evaluate(InfoEvaluator::empty())
}

fn poseidon_info() -> InfoEvaluator {
    use num_traits::Zero;
    let component = PoseidonEval {
        log_n_rows: 1,
        lookup_elements: PoseidonElements::dummy(),
        claimed_sum: SecureField::zero(),
    };
    component.evaluate(InfoEvaluator::empty())
}

/// Dynamic Scheduler component that requests variable number of Poseidon hashes
///
/// Trace structure per row: [is_active, input_state[16], output_state[16]]
/// - is_active: 1 if this row contains a real call, 0 if padding
/// - input_state: 16-element input to Poseidon
/// - output_state: 16-element output from Poseidon
#[derive(Clone)]
pub struct SchedulerEval {
    log_size: u32,
    lookup_elements: PoseidonElements,
}

impl FrameworkEval for SchedulerEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + CONSTRAINT_EVAL_BLOWUP_FACTOR
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // Read is_active flag (1 column)
        let is_active = eval.next_trace_mask();

        // Read input state (16 elements)
        let input_state: [_; N_STATE] = std::array::from_fn(|_| eval.next_trace_mask());

        // Read output state (16 elements)
        let output_state: [_; N_STATE] = std::array::from_fn(|_| eval.next_trace_mask());

        // CONDITIONAL LogUp: Only add to lookup if is_active = 1
        // When is_active = 0 (padding), we multiply coefficients by 0
        //
        // Poseidon (provider) does: (+1 * initial, -1 * final)
        // Scheduler (consumer) does: (-1 * input, +1 * output) when active
        //
        // Coefficient becomes: is_active * (-1) or is_active * (+1)
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::one() * is_active.clone(), // -is_active
            &input_state,
        ));

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::one() * is_active, // +is_active
            &output_state,
        ));

        eval.finalize_logup_in_pairs();
        eval
    }
}

// Test-only imports and functions
#[cfg(test)]
use stwo::core::fields::m31::M31;
#[cfg(test)]
use stwo::core::poly::circle::CanonicCoset;
#[cfg(test)]
use stwo::prover::backend::simd::m31::{PackedM31, LOG_N_LANES};
#[cfg(test)]
use stwo::prover::backend::simd::qm31::PackedSecureField;
#[cfg(test)]
use stwo::prover::backend::{Col, Column};
#[cfg(test)]
use stwo::prover::poly::circle::CircleEvaluation;
#[cfg(test)]
use stwo::prover::poly::BitReversedOrder;
#[cfg(test)]
use stwo_constraint_framework::{LogupTraceGenerator, Relation};
#[cfg(test)]
use crate::poseidon::{
    gen_interaction_trace as gen_poseidon_interaction_trace, gen_trace as gen_poseidon_trace,
    LookupData as PoseidonLookupData,
};

/// Generate dynamic scheduler trace with variable number of calls
///
/// Trace structure: [is_active (1), input_state (16), output_state (16)] = 33 columns
///
/// Args:
///   - num_calls: How many actual Poseidon calls (1 to MAX_CALLS)
///   - log_size: log2(trace rows) - must be >= log2(num_calls)
///   - poseidon_lookup_data: Data from Poseidon for the actual calls
#[cfg(test)]
fn gen_scheduler_trace(
    num_calls: usize,
    log_size: u32,
    poseidon_lookup_data: &PoseidonLookupData,
) -> Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>> {
    const POSEIDON_INSTANCES_PER_ROW: usize = 8;

    assert!(num_calls <= MAX_CALLS, "num_calls exceeds MAX_CALLS");
    assert!(num_calls <= (1 << log_size), "num_calls exceeds trace size");

    // Scheduler has 33 columns: [is_active, input_state[16], output_state[16]]
    let n_cols = 1 + 16 + 16;
    let trace_size = 1 << log_size;
    let mut trace = (0..n_cols)
        .map(|_| Col::<SimdBackend, M31>::zeros(trace_size))
        .collect::<Vec<_>>();

    // Fill active rows (0..num_calls)
    for call_idx in 0..num_calls {
        let scheduler_vec_index = call_idx >> LOG_N_LANES;
        let scheduler_lane = call_idx & ((1 << LOG_N_LANES) - 1);

        // Which Poseidon instance does this call correspond to?
        let poseidon_absolute_idx = call_idx;  // Linear index into Poseidon instances
        let poseidon_row = poseidon_absolute_idx / POSEIDON_INSTANCES_PER_ROW;
        let poseidon_instance = poseidon_absolute_idx % POSEIDON_INSTANCES_PER_ROW;

        // Convert poseidon_row to SIMD indexing
        let poseidon_vec_index = poseidon_row >> LOG_N_LANES;
        let poseidon_lane = poseidon_row & ((1 << LOG_N_LANES) - 1);

        // is_active = 1
        let mut is_active_arr = trace[0].data[scheduler_vec_index].to_array();
        is_active_arr[scheduler_lane] = M31::one();
        trace[0].data[scheduler_vec_index] = PackedM31::from_array(is_active_arr);

        // Input state (16 columns)
        for state_i in 0..N_STATE {
            let value = poseidon_lookup_data.initial_state[poseidon_instance][state_i]
                .data[poseidon_vec_index].to_array()[poseidon_lane];
            let mut arr = trace[1 + state_i].data[scheduler_vec_index].to_array();
            arr[scheduler_lane] = value;
            trace[1 + state_i].data[scheduler_vec_index] = PackedM31::from_array(arr);
        }

        // Output state (16 columns)
        for state_i in 0..N_STATE {
            let value = poseidon_lookup_data.final_state[poseidon_instance][state_i]
                .data[poseidon_vec_index].to_array()[poseidon_lane];
            let mut arr = trace[1 + 16 + state_i].data[scheduler_vec_index].to_array();
            arr[scheduler_lane] = value;
            trace[1 + 16 + state_i].data[scheduler_vec_index] = PackedM31::from_array(arr);
        }
    }

    // Padding rows (num_calls..trace_size) are already zeros:
    // - is_active = 0
    // - input_state = [0, 0, ..., 0]
    // - output_state = [0, 0, ..., 0]

    let domain = CanonicCoset::new(log_size).circle_domain();
    trace
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect()
}

/// Generate scheduler LogUp trace with conditional lookups based on is_active
///
/// Only rows with is_active=1 contribute to the LogUp sum
#[cfg(test)]
fn gen_scheduler_logup_trace(
    log_size: u32,
    scheduler_trace: &[CircleEvaluation<SimdBackend, M31, BitReversedOrder>],
    lookup_elements: &PoseidonElements,
) -> (
    Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    SecureField,
) {
    let mut logup_gen = LogupTraceGenerator::new(log_size);
    let mut col_gen = logup_gen.new_col();

    // Now we have ONE LogUp column for the entire trace (not per-instance)
    // Trace structure: [is_active (col 0), input_state (cols 1-16), output_state (cols 17-32)]
    for row in 0..(1 << (log_size - LOG_N_LANES)) {
        // Read is_active flag
        let is_active = scheduler_trace[0].data[row];

        // Read input state (cols 1-16)
        let input_state: [_; N_STATE] =
            std::array::from_fn(|i| scheduler_trace[1 + i].data[row]);
        let input_combined: PackedSecureField = lookup_elements.combine(&input_state);

        // Read output state (cols 17-32)
        let output_state: [_; N_STATE] =
            std::array::from_fn(|i| scheduler_trace[17 + i].data[row]);
        let output_combined: PackedSecureField = lookup_elements.combine(&output_state);

        // CONDITIONAL LogUp:
        // If is_active = 1: write_frac with (input - output) / (input * output)
        // If is_active = 0: write_frac with 0 / 1 (contributes nothing)
        //
        // Convert each element of is_active (PackedM31) to SecureField
        // We need to convert element-wise, not broadcast!
        let is_active_arr = is_active.to_array();
        let is_active_qm31 = PackedSecureField::from_array(
            is_active_arr.map(|m| SecureField::from(m))
        );

        // We multiply numerator by is_active to make it conditional
        let numerator = (input_combined - output_combined) * is_active_qm31;

        // For denominator, when is_active=0 we want to avoid division by zero
        // Use: is_active * (input * output) + (1 - is_active) * 1
        // When is_active=1: input * output
        // When is_active=0: 1
        let one = PackedSecureField::broadcast(SecureField::one());
        let denominator = is_active_qm31 * (input_combined * output_combined)
            + (one - is_active_qm31) * one;

        col_gen.write_frac(row, numerator, denominator);
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Zero;
    use stwo::core::channel::Blake2sChannel;
    use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
    use stwo::core::proof::StarkProof;
    use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use stwo::core::vcs::MerkleHasher;
    use stwo::core::verifier::verify;
    use stwo::prover::poly::circle::PolyOps;
    use stwo::prover::{prove, CommitmentSchemeProver};

    struct ComponentsProof<H: MerkleHasher> {
        statement0: ComponentsStatement0,
        statement1: ComponentsStatement1,
        stark_proof: StarkProof<H>,
    }

    #[test]
    fn test_dynamic_scheduler_poseidon() {
        println!("=== DYNAMIC Scheduler + Poseidon Component Composition ===");
        println!("This example shows VARIABLE-LENGTH composition\n");

        // Configure how many Poseidon calls we want to make
        // Can be any multiple of 8: 64, 128, 256, 512, 1024
        let num_calls: usize = 64;

        // LIMITATION: Current implementation uses ALL Poseidon instances
        // ------------------------------------------------------------
        // The Poseidon component generates LogUp entries for ALL its instances.
        // Since we cannot modify Poseidon (it's shared code), we must use ALL instances.
        //
        // This means:
        // - For num_calls=64, Poseidon creates 128 instances (next power of 2)
        // - Scheduler MUST mark all 128 as active for LogUp to balance
        // - Only the first 64 contain "real" data, rest are hash(0,0,...,0)
        //
        // To properly support variable-length input, we would need:
        // 1. A modified Poseidon that accepts num_calls and conditionally generates LogUp
        // 2. OR: Use num_calls that matches a power-of-2 * 8 (e.g., 64, 128, 256, 512, 1024)
        //
        // For this example, we demonstrate the composition pattern, accepting this limitation.

        const POSEIDON_INSTANCES_PER_ROW: usize = 8;

        // Calculate Poseidon log_size to fit at least num_calls
        let poseidon_rows_needed = (num_calls + POSEIDON_INSTANCES_PER_ROW - 1) / POSEIDON_INSTANCES_PER_ROW;
        let poseidon_log_size = (poseidon_rows_needed as u32).next_power_of_two().ilog2().max(LOG_N_LANES);

        // Total instances that Poseidon will create (and ALL will have LogUp entries)
        let total_poseidon_instances = (1 << poseidon_log_size) * POSEIDON_INSTANCES_PER_ROW;

        // Scheduler size must fit all Poseidon instances
        let log_size = (total_poseidon_instances as u32).next_power_of_two().ilog2().max(LOG_N_LANES);
        let config = PcsConfig::default();

        println!("Number of actual Poseidon calls: {}", num_calls);
        println!("Log size (trace rows): {} (2^{} = {} rows)", log_size, log_size, 1 << log_size);
        println!("Padding rows: {}\n", (1 << log_size) - num_calls);

        // Use maximum of scheduler and Poseidon LOG_EXPAND
        let max_log_expand = CONSTRAINT_EVAL_BLOWUP_FACTOR.max(POSEIDON_LOG_EXPAND);
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(log_size + max_log_expand + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        let channel = &mut Blake2sChannel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

        println!("Step 1: Committing preprocessed columns (empty)...");
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(vec![]);
        tree_builder.commit(channel);

        println!("Step 2: Generating Poseidon trace (this computes the actual hashes)...");
        println!("  Poseidon log_size: {} ({} rows with {} instances each = {} total instances)",
                 poseidon_log_size, 1 << poseidon_log_size, POSEIDON_INSTANCES_PER_ROW,
                 total_poseidon_instances);
        println!("  Poseidon will generate LogUp for ALL {} instances!", total_poseidon_instances);
        let (poseidon_trace, poseidon_lookup_data) = gen_poseidon_trace(poseidon_log_size);

        println!("Step 3: Generating Scheduler trace (using Poseidon's input/output)...");
        println!("  Scheduler will mark ALL {} Poseidon instances as active", total_poseidon_instances);
        println!("  (even though only first {} are 'real' calls)", num_calls);
        let scheduler_trace = gen_scheduler_trace(total_poseidon_instances, log_size, &poseidon_lookup_data);

        let statement0 = ComponentsStatement0 {
            scheduler_log_size: log_size,
            poseidon_log_size,
            num_calls,
        };
        statement0.mix_into(channel);

        println!("Step 4: Committing trace columns...");
        let mut tree_builder = commitment_scheme.tree_builder();
        // IMPORTANT: Order must match Components::new() which creates poseidon_component first
        tree_builder.extend_evals([poseidon_trace.clone(), scheduler_trace.clone()].concat());
        tree_builder.commit(channel);

        println!("Step 5: Drawing lookup elements from channel...");
        let lookup_elements = PoseidonElements::draw(channel);

        println!("Step 6: Generating LogUp interaction columns...");
        let (scheduler_logup_cols, scheduler_claimed_sum) =
            gen_scheduler_logup_trace(log_size, &scheduler_trace, &lookup_elements);

        let (poseidon_logup_cols, poseidon_claimed_sum) =
            gen_poseidon_interaction_trace(poseidon_log_size, poseidon_lookup_data, &lookup_elements);

        let statement1 = ComponentsStatement1 {
            scheduler_claimed_sum,
            poseidon_claimed_sum,
        };
        statement1.mix_into(channel);

        println!("Step 7: Committing LogUp columns...");
        println!("  Scheduler trace columns: {}", scheduler_trace.len());
        println!("  Poseidon trace columns: {}", poseidon_trace.len());
        println!("  Scheduler LogUp columns: {}", scheduler_logup_cols.len());
        println!("  Poseidon LogUp columns: {}", poseidon_logup_cols.len());

        let mut tree_builder = commitment_scheme.tree_builder();
        // IMPORTANT: Same order as components - Poseidon first, Scheduler second
        tree_builder.extend_evals([poseidon_logup_cols, scheduler_logup_cols].concat());
        tree_builder.commit(channel);

        println!("Step 8: Creating components...");
        let components = Components::new(&statement0, &lookup_elements, &statement1);

        println!("Step 9: Generating STARK proof...");
        let stark_proof = prove(&components.component_provers(), channel, commitment_scheme).unwrap();
        println!("  ✓ Proof generated successfully\n");

        let proof = ComponentsProof {
            statement0,
            statement1,
            stark_proof,
        };

        println!("Step 10: Verifying proof...");

        // Verify claimed sums balance
        assert_eq!(
            scheduler_claimed_sum + poseidon_claimed_sum,
            SecureField::zero()
        );

        let statement0 = proof.statement0;
        let statement1 = proof.statement1;
        let stark_proof = proof.stark_proof;

        let channel = &mut Blake2sChannel::default();
        let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        let log_sizes = statement0.log_sizes();

        // Preprocessed columns
        commitment_scheme.commit(stark_proof.commitments[0], &log_sizes[0], channel);
        statement0.mix_into(channel);

        // Trace columns
        commitment_scheme.commit(stark_proof.commitments[1], &log_sizes[1], channel);

        // Draw lookup elements
        let lookup_elements = PoseidonElements::draw(channel);

        statement1.mix_into(channel);

        // Interaction columns
        commitment_scheme.commit(stark_proof.commitments[2], &log_sizes[2], channel);

        let components = Components::new(&statement0, &lookup_elements, &statement1);

        verify(
            &components.components(),
            channel,
            commitment_scheme,
            stark_proof,
        )
        .unwrap();

        println!("  ✓ Proof verified successfully\n");
        println!("=== Example completed successfully! ===");
        println!("\nWhat happened:");
        println!("1. Scheduler requested {} Poseidon hashes", total_poseidon_instances);
        println!("2. Poseidon component computed all {} hashes", total_poseidon_instances);
        println!("3. LogUp verified all requests matched computations (sum balanced to zero)");
        println!("4. STARK proof proves correctness of entire composition");
    }
}
