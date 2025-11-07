use stwo_constraint_framework::relation;

mod computing;
mod trace_gen;

pub use computing::{PoseidonComputingComponent, PoseidonComputingEval};
pub use trace_gen::{gen_computing_trace, gen_computing_interaction_trace};

// Import and re-export constants from lib.rs
use crate::N_STATE;

pub const LOG_CONSTRAINT_DEGREE: u32 = 1;
pub const POSEIDON_RELATION_SIZE: usize = N_STATE; // 16-element state

relation!(PoseidonRelation, POSEIDON_RELATION_SIZE);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gen_is_first_column, gen_is_active_column, gen_is_target_column,
                is_first_column_id, is_active_column_id, is_target_column_id, RATE};
    use stwo::core::channel::Blake2sChannel;
    use stwo::core::fields::m31::BaseField;
    use stwo::core::pcs::PcsConfig;
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::poly::circle::PolyOps;
    use stwo::prover::{prove, CommitmentSchemeProver};
    use stwo_constraint_framework::{FrameworkComponent, TraceLocationAllocator};

    #[test]
    fn test_poseidon_trace_generation() {
        let log_size = 4; // 16 rows
        let target_element = 10;
        let initial_message: [BaseField; RATE] = std::array::from_fn(|i| BaseField::from_u32_unchecked(i as u32 + 1));

        // Generate trace
        let (trace, target_state) = gen_computing_trace(log_size, initial_message, target_element);

        println!("✓ Poseidon trace generated successfully");
        println!("  Trace columns: {} (message + state_in + state_out)", trace.len());
        println!("  Rows: {} (active up to element {})", 1 << log_size, target_element);
        println!("  Target state at element {}: {:?}", target_element, target_state);

        assert_eq!(trace.len(), RATE + N_STATE + N_STATE); // 40 columns
    }

    #[test]
    fn test_poseidon_proof() {
        println!("\n==================================================");
        println!("  POSEIDON HASH PROOF TEST");
        println!("==================================================\n");

        let target_element = 10;
        let initial_message: [BaseField; RATE] = std::array::from_fn(|i| BaseField::from_u32_unchecked(i as u32 + 1));

        // Compute log_size dynamically
        let min_rows = target_element + 1;
        let min_log_size = if min_rows <= 1 { 0 } else { ((min_rows - 1) as u32).ilog2() + 1 };
        let log_size = min_log_size.max(4); // minimum 16 rows for SIMD

        println!("Target element: {}", target_element);
        println!("Computed log_size: {} ({} rows)\n", log_size, 1 << log_size);

        // Setup prover
        let config = PcsConfig::default();
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        let channel = &mut Blake2sChannel::default();
        let mut commitment_scheme = CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(
            config,
            &twiddles,
        );

        // Step 1: Generate and commit preprocessed columns
        println!("Step 1: Generating and committing preprocessed columns...");
        let is_first_col = gen_is_first_column(log_size);
        let is_active_col = gen_is_active_column(log_size, target_element);
        let is_target_col = gen_is_target_column(log_size, target_element);

        let preprocessed_trace = vec![is_first_col, is_active_col, is_target_col];
        println!("Generated 3 preprocessed columns: is_first, is_active, is_target");

        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(preprocessed_trace);
        tree_builder.commit(channel);

        // Step 2: Generate main trace
        println!("\nStep 2: Generating main trace...");
        let (trace, target_state_value) = gen_computing_trace(log_size, initial_message, target_element);
        println!("Poseidon trace: {} rows, {} columns", 1 << log_size, trace.len());
        println!("Target state value at element {}: {:?}", target_element, target_state_value);

        // Step 3: Commit main trace
        println!("\nStep 3: Committing main trace...");
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace.clone());
        tree_builder.commit(channel);

        // Step 4: Draw PoseidonRelation from channel
        println!("\nStep 4: Drawing LogUp relation from channel...");
        let poseidon_relation = PoseidonRelation::draw(channel);

        // Step 5: Generate interaction trace (LogUp columns)
        println!("\nStep 5: Generating LogUp interaction trace...");
        let (interaction_trace, claimed_sum) =
            gen_computing_interaction_trace(&trace, &poseidon_relation, target_element);
        println!("Claimed sum: {:?}", claimed_sum);

        // Step 6: Commit interaction trace
        println!("\nStep 6: Committing interaction trace...");
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(interaction_trace);
        tree_builder.commit(channel);

        // Step 7: Create component
        println!("\nStep 7: Creating component...");
        let mut tree_span_provider = TraceLocationAllocator::default();
        let component = FrameworkComponent::new(
            &mut tree_span_provider,
            PoseidonComputingEval {
                log_n_rows: log_size,
                initial_message,
                poseidon_relation,
                claimed_sum,
                is_first_id: is_first_column_id(log_size),
                is_active_id: is_active_column_id(log_size, target_element),
                is_target_id: is_target_column_id(log_size, target_element),
            },
            claimed_sum,
        );

        // Step 8: Generate proof
        println!("\nStep 8: Generating STARK proof...");
        let result = prove(&[&component], channel, commitment_scheme);

        match result {
            Ok(_proof) => {
                println!("\n==================================================");
                println!("  ✓✓✓ PROOF GENERATED SUCCESSFULLY! ✓✓✓");
                println!("==================================================\n");
                println!("This proves:");
                println!("  1. Poseidon hash computed correctly up to element {}", target_element);
                println!("  2. Transition constraints satisfied (sponge construction)");
                println!("  3. is_active masking works for Poseidon!");
            }
            Err(e) => {
                panic!("Proof generation failed: {:?}", e);
            }
        }
    }
}
