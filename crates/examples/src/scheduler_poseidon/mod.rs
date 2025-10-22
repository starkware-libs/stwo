//! Scheduler + Poseidon Component Composition Example
//!
//! This example demonstrates realistic component composition where:
//! - SchedulerComponent: Schedules hash computations (sends 16-element inputs, expects outputs)
//! - PoseidonComponent: Performs actual Poseidon2 hashing
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
const N_INSTANCES_PER_ROW: usize = 8; // Must match Poseidon's N_INSTANCES_PER_ROW

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
                log_n_rows: statement0.log_size,
                lookup_elements: lookup_elements.clone(),
                claimed_sum: statement1.poseidon_claimed_sum,
            },
            statement1.poseidon_claimed_sum,
        );

        let scheduler_component = SchedulerComponent::new(
            tree_span_provider,
            SchedulerEval {
                log_size: statement0.log_size,
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
    log_size: u32,
}

impl ComponentsStatement0 {
    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_u64(self.log_size as u64);
    }

    pub fn log_sizes(&self) -> TreeVec<Vec<u32>> {
        let mut log_sizes = vec![];

        // IMPORTANT: Same order as Components::new()!
        // Poseidon first
        log_sizes.push(
            poseidon_info()
                .mask_offsets
                .as_cols_ref()
                .map_cols(|_| self.log_size),
        );

        // Scheduler second
        log_sizes.push(
            scheduler_info()
                .mask_offsets
                .as_cols_ref()
                .map_cols(|_| self.log_size),
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

/// Scheduler component that requests Poseidon hashes
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
        // Process N_INSTANCES_PER_ROW instances (8 instances per row)
        for _ in 0..N_INSTANCES_PER_ROW {
            // Read input state (16 elements)
            let input_state: [_; N_STATE] = std::array::from_fn(|_| eval.next_trace_mask());

            // Read output state (16 elements)
            let output_state: [_; N_STATE] = std::array::from_fn(|_| eval.next_trace_mask());

            // Poseidon (provider) does: (+1 * initial, -1 * final)
            // LogUp for Poseidon: (+1/initial - 1/final) = (final - initial)/(initial*final)
            //
            // Scheduler (consumer) must do OPPOSITE:
            // So: (-1 * input, +1 * output) where input=initial, output=final
            // LogUp for Scheduler: (-1/input + 1/output) = (output - input)/(output*input)
            eval.add_to_relation(RelationEntry::new(
                &self.lookup_elements,
                -E::EF::one(),
                &input_state,
            ));

            eval.add_to_relation(RelationEntry::new(
                &self.lookup_elements,
                E::EF::one(),
                &output_state,
            ));
        }

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
use stwo::prover::backend::simd::m31::LOG_N_LANES;
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

/// Generate scheduler trace
/// For simplicity, we'll create random 16-element inputs and compute their hashes
#[cfg(test)]
fn gen_scheduler_trace(
    log_size: u32,
    poseidon_lookup_data: &PoseidonLookupData,
) -> Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>> {
    // Scheduler has 32 * N_INSTANCES_PER_ROW columns:
    // For each instance: 16 for input_state, 16 for output_state
    let n_cols = 32 * N_INSTANCES_PER_ROW;
    let mut trace = (0..n_cols)
        .map(|_| Col::<SimdBackend, M31>::zeros(1 << log_size))
        .collect::<Vec<_>>();

    // For each row, use all Poseidon instances
    for vec_index in 0..(1 << (log_size - LOG_N_LANES)) {
        for rep_i in 0..N_INSTANCES_PER_ROW {
            let col_offset = rep_i * 32;

            // Input state (16 columns)
            for state_i in 0..N_STATE {
                trace[col_offset + state_i].data[vec_index] =
                    poseidon_lookup_data.initial_state[rep_i][state_i].data[vec_index];
            }

            // Output state (16 columns)
            for state_i in 0..N_STATE {
                trace[col_offset + 16 + state_i].data[vec_index] =
                    poseidon_lookup_data.final_state[rep_i][state_i].data[vec_index];
            }
        }
    }

    let domain = CanonicCoset::new(log_size).circle_domain();
    trace
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect()
}

/// Generate scheduler LogUp trace
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

    // Generate one LogUp column per instance (8 columns total)
    for rep_i in 0..N_INSTANCES_PER_ROW {
        let col_offset = rep_i * 32;
        let mut col_gen = logup_gen.new_col();

        for row in 0..(1 << (log_size - LOG_N_LANES)) {
            // Get input state from columns [col_offset..(col_offset+16)]
            let input_state: [_; N_STATE] =
                std::array::from_fn(|i| scheduler_trace[col_offset + i].data[row]);
            let input_combined: PackedSecureField = lookup_elements.combine(&input_state);

            // Get output state from columns [(col_offset+16)..(col_offset+32)]
            let output_state: [_; N_STATE] =
                std::array::from_fn(|i| scheduler_trace[col_offset + 16 + i].data[row]);
            let output_combined: PackedSecureField = lookup_elements.combine(&output_state);

            // LogUp: (-1 * input) + (+1 * output)
            // Formula: (-1/a + 1/b) = (a - b) / (a * b)
            // So: (-1/input + 1/output) = (input - output) / (input * output)
            col_gen.write_frac(
                row,
                input_combined - output_combined,
                input_combined * output_combined,
            );
        }
        col_gen.finalize_col();
    }

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
    fn test_scheduler_poseidon_prove_and_verify() {
        println!("=== Scheduler + Poseidon Component Composition ===");
        println!("This example shows realistic component composition with crypto primitive\n");

        let log_size = LOG_N_LANES + 1; // Use 5 instead of 4 for more rows
        let config = PcsConfig::default();

        println!("Log size (rows): {}", log_size);
        println!("Number of hash instances total: {}\n", (1 << log_size) * N_INSTANCES_PER_ROW);

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
        let (poseidon_trace, poseidon_lookup_data) = gen_poseidon_trace(log_size);

        println!("Step 3: Generating Scheduler trace (using Poseidon's input/output)...");
        let scheduler_trace = gen_scheduler_trace(log_size, &poseidon_lookup_data);

        let statement0 = ComponentsStatement0 { log_size };
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
        println!("  Scheduler claimed sum: {:?}", scheduler_claimed_sum);

        let (poseidon_logup_cols, poseidon_claimed_sum) =
            gen_poseidon_interaction_trace(log_size, poseidon_lookup_data, &lookup_elements);
        println!("  Poseidon claimed sum: {:?}", poseidon_claimed_sum);

        println!(
            "  Sum check (should be zero): {:?}\n",
            scheduler_claimed_sum + poseidon_claimed_sum
        );

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
        println!("1. Scheduler requested {} Poseidon hashes", 1 << log_size);
        println!("2. Poseidon component computed all hashes");
        println!("3. LogUp verified all requests matched computations");
        println!("4. STARK proof proves correctness of entire system");
    }
}
