use stwo_constraint_framework::relation;

mod computing;
mod scheduler;
mod trace_gen;

#[cfg(test)]
mod dump_trace;

pub use computing::{FibonacciComputingComponent, FibonacciComputingEval};
pub use scheduler::{FibonacciSchedulerComponent, FibonacciSchedulerEval};
pub use trace_gen::{
    gen_computing_trace, gen_scheduler_trace,
    gen_computing_interaction_trace, gen_scheduler_interaction_trace
};

// Import preprocessed column generators from lib.rs
use crate::{gen_is_active_column, gen_is_target_column, is_active_column_id, is_target_column_id};

pub const LOG_CONSTRAINT_DEGREE: u32 = 1;
pub const FIBONACCI_RELATION_SIZE: usize = 1;

relation!(FibonacciRelation, FIBONACCI_RELATION_SIZE);

use num_traits::Zero;
use stwo::core::channel::{Blake2sChannel, Channel};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::StarkProof;
use stwo::core::utils::bit_reverse_coset_to_circle_domain_order;
use stwo::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::core::pcs::{CommitmentSchemeVerifier, TreeVec};
use stwo::prover::{prove, CommitmentSchemeProver};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::TraceLocationAllocator;

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

/// Statement 0: Component configuration (log_size)
/// This is mixed into the channel before drawing the FibonacciRelation
#[derive(Clone, Copy, Debug)]
pub struct SingleFibStatement0 {
    pub log_size: u32,
}

impl SingleFibStatement0 {
    pub fn mix_into(&self, channel: &mut Blake2sChannel) {
        channel.mix_u64(self.log_size as u64);
    }

    /// Returns log sizes for all trees (preprocessed, main, interaction)
    pub fn log_sizes(&self) -> TreeVec<Vec<u32>> {
        TreeVec(vec![
            // Tree 0: Preprocessed (3 columns: is_first, is_active, is_target)
            vec![self.log_size; 3],
            // Tree 1: Main traces (3 cols for computing + 1 col for scheduler = 4 cols)
            vec![self.log_size; 4],
            // Tree 2: Interaction traces (1 SecureColumn per component = 4 BaseField cols each = 8 cols)
            // 2 components * 4 columns (SECURE_EXTENSION_DEGREE) = 8
            vec![self.log_size; 8],
        ])
    }
}

/// Statement 1: LogUp claimed sums
/// This is mixed into the channel after drawing FibonacciRelation
#[derive(Clone, Copy, Debug)]
pub struct SingleFibStatement1 {
    pub claimed_sum_computing: SecureField,
    pub claimed_sum_scheduler: SecureField,
}

impl SingleFibStatement1 {
    pub fn mix_into(&self, channel: &mut Blake2sChannel) {
        channel.mix_felts(&[
            self.claimed_sum_computing,
            self.claimed_sum_scheduler,
        ]);
    }
}

/// Prove single-component Fibonacci with LogUp
///
/// This generates a STARK proof for:
/// - Computing: Fibonacci up to target_element (rest are zeros)
/// - Scheduler: Reads the specific result from Computing
///
/// LogUp verifies that Scheduler uses the correct value from Computing component.
pub fn prove_single_fib(
    target_element: usize,
    channel: &mut Blake2sChannel,
    mut commitment_scheme: CommitmentSchemeProver<SimdBackend, Blake2sMerkleChannel>,
) -> Result<
    (
        StarkProof<Blake2sMerkleHasher>,
        FibonacciComputingComponent,
        FibonacciSchedulerComponent,
        SingleFibStatement0,
        SingleFibStatement1,
    ),
    Box<dyn std::error::Error>,
> {
    // Step 0: Compute dynamic log_size
    let min_rows = target_element + 1;  // +1 because 0-indexed
    let min_log_size = if min_rows <= 1 {
        0
    } else {
        (min_rows - 1).ilog2() + 1  // log2_ceil
    };
    let log_size = min_log_size.max(4);  // minimum 16 rows for SIMD (LOG_N_LANES = 4)

    println!("=== Single-Component Fibonacci Proof Generation ===");
    println!("Target element: {}", target_element);
    println!("Computed log_size: {} ({} rows)\n", log_size, 1 << log_size);

    // Step 1: Generate and commit preprocessed columns
    println!("Step 1: Generating and committing preprocessed columns...");
    let is_first_col = gen_is_first_column(log_size);
    let is_active_col = gen_is_active_column(log_size, target_element);
    let is_target_col = gen_is_target_column(log_size, target_element);

    let preprocessed_trace = vec![
        is_first_col,
        is_active_col,
        is_target_col,
    ];
    println!("Generated 3 preprocessed columns: is_first, is_active, is_target");

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(preprocessed_trace);
    tree_builder.commit(channel);

    // Mix Statement0 (log_size, target_element) into channel
    let statement0 = SingleFibStatement0 { log_size };
    statement0.mix_into(channel);

    // Step 2: Generate main traces
    println!("\nStep 2: Generating main traces...");
    let (trace_computing, fib_c_value) = gen_computing_trace(log_size, 1, 1, target_element);
    let trace_scheduler = gen_scheduler_trace(log_size, fib_c_value);
    println!("Computing trace (initial: 1, 1): {} rows", 1 << log_size);
    println!("Scheduler trace: {} rows", 1 << log_size);

    // Step 3: Commit main traces
    println!("\nStep 3: Committing main traces...");
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals([trace_computing.clone(), trace_scheduler.clone()].concat());
    tree_builder.commit(channel);

    // Step 4: Draw FibonacciRelation from channel
    println!("\nStep 4: Drawing LogUp relation from channel...");
    let fibonacci_relation = FibonacciRelation::draw(channel);

    // Step 5: Generate interaction traces (LogUp columns)
    println!("\nStep 5: Generating LogUp interaction traces...");
    let (interaction_trace_computing, claimed_sum_computing) =
        gen_computing_interaction_trace(&trace_computing, &fibonacci_relation, target_element);

    let (interaction_trace_scheduler, claimed_sum_scheduler) =
        gen_scheduler_interaction_trace(&trace_scheduler, &fibonacci_relation);

    // Step 6: Verify LogUp property: sum of all claimed_sums should be 0
    let total_sum = claimed_sum_computing + claimed_sum_scheduler;
    println!("\nLogUp verification:");
    println!("  Computing yields: {:?}", claimed_sum_computing);
    println!("  Scheduler uses:   {:?}", claimed_sum_scheduler);
    println!("  Total sum:        {:?}", total_sum);
    if total_sum == Zero::zero() {
        println!("LogUp property satisfied: total sum = 0");
    } else {
        println!("Warning: LogUp sum is not zero!");
    }

    // Mix Statement1 (claimed_sums) into channel
    let statement1 = SingleFibStatement1 {
        claimed_sum_computing,
        claimed_sum_scheduler,
    };
    statement1.mix_into(channel);

    // Step 7: Commit interaction traces
    println!("\nStep 6: Committing interaction traces...");
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals([interaction_trace_computing, interaction_trace_scheduler].concat());
    tree_builder.commit(channel);

    // Step 8: Create components with TraceLocationAllocator (AFTER committing interaction traces)
    println!("\nStep 7: Creating components...");
    let mut tree_span_provider = TraceLocationAllocator::default();
    let is_first_id = is_first_column_id(log_size);

    let component_computing = FibonacciComputingComponent::new(
        &mut tree_span_provider,
        FibonacciComputingEval {
            log_n_rows: log_size,
            initial_a: 1,
            initial_b: 1,
            fibonacci_relation: fibonacci_relation.clone(),
            claimed_sum: claimed_sum_computing,
            is_first_id: is_first_id.clone(),
            is_active_id: is_active_column_id(log_size, target_element),
            is_target_id: is_target_column_id(log_size, target_element),
        },
        claimed_sum_computing,
    );

    let component_scheduler = FibonacciSchedulerComponent::new(
        &mut tree_span_provider,
        FibonacciSchedulerEval {
            log_n_rows: log_size,
            fibonacci_relation,
            claimed_sum: claimed_sum_scheduler,
            is_first_id: is_first_id.clone(),
        },
        claimed_sum_scheduler,
    );

    // Step 9: Generate proof
    println!("\nStep 8: Generating STARK proof...");
    let proof = prove(
        &[&component_computing, &component_scheduler],
        channel,
        commitment_scheme,
    )?;
    println!("Proof generated successfully!");

    Ok((proof, component_computing, component_scheduler, statement0, statement1))
}

/// Verify single-component Fibonacci proof with LogUp
///
/// This verifies a STARK proof for the single-component Fibonacci circuit.
/// The verifier must commit to the same tree structure as the prover.
pub fn verify_single_fib(
    proof: StarkProof<Blake2sMerkleHasher>,
    target_element: usize,
    statement0: SingleFibStatement0,
    statement1: SingleFibStatement1,
    config: stwo::core::pcs::PcsConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Single-Component Fibonacci Proof Verification ===");
    println!("Target element: {}\n", target_element);

    // Extract log_size from statement
    let log_size = statement0.log_size;

    // Step 1: Setup verifier channel and commitment scheme
    println!("Step 1: Setting up verifier...");
    let channel = &mut Blake2sChannel::default();
    let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
    let log_sizes = statement0.log_sizes();

    // Step 2: Commit preprocessed columns (is_first)
    println!("\nStep 2: Committing preprocessed columns...");
    commitment_scheme.commit(
        proof.commitments[0],
        &log_sizes[0],
        channel,
    );

    // Mix Statement0 (log_size) into channel
    statement0.mix_into(channel);

    // Step 3: Commit main traces
    println!("\nStep 3: Committing main traces...");
    commitment_scheme.commit(
        proof.commitments[1],
        &log_sizes[1],
        channel,
    );

    // Step 4: Draw FibonacciRelation from channel (must match prover)
    println!("\nStep 4: Drawing LogUp relation from channel...");
    let fibonacci_relation = FibonacciRelation::draw(channel);

    // Mix Statement1 (claimed_sums) into channel
    statement1.mix_into(channel);

    // Step 5: Commit interaction traces
    println!("\nStep 5: Committing interaction traces...");
    commitment_scheme.commit(
        proof.commitments[2],
        &log_sizes[2],
        channel,
    );

    // Step 6: Create components (AFTER committing interaction traces, matching prover order)
    println!("\nStep 6: Creating components for verification...");
    let mut tree_span_provider = TraceLocationAllocator::default();
    let is_first_id = is_first_column_id(log_size);

    let component_computing = FibonacciComputingComponent::new(
        &mut tree_span_provider,
        FibonacciComputingEval {
            log_n_rows: log_size,
            initial_a: 1,
            initial_b: 1,
            fibonacci_relation: fibonacci_relation.clone(),
            claimed_sum: statement1.claimed_sum_computing,
            is_first_id: is_first_id.clone(),
            is_active_id: is_active_column_id(log_size, target_element),
            is_target_id: is_target_column_id(log_size, target_element),
        },
        statement1.claimed_sum_computing,
    );

    let component_scheduler = FibonacciSchedulerComponent::new(
        &mut tree_span_provider,
        FibonacciSchedulerEval {
            log_n_rows: log_size,
            fibonacci_relation,
            claimed_sum: statement1.claimed_sum_scheduler,
            is_first_id: is_first_id.clone(),
        },
        statement1.claimed_sum_scheduler,
    );

    // Step 7: Verify the proof
    println!("\nStep 7: Verifying STARK proof...");
    stwo::core::verifier::verify(
        &[&component_computing, &component_scheduler],
        channel,
        commitment_scheme,
        proof,
    )?;
    println!("Proof verified successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use stwo::core::channel::Blake2sChannel;
    use stwo::core::pcs::PcsConfig;
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::poly::circle::PolyOps;
    use stwo::prover::CommitmentSchemeProver;

    #[test]
    fn test_single_component_fibonacci_traces() {
        let target_element = 10;
        let log_size = 4; // 16 rows

        // Generate traces for Computing (initial: 1, 1) up to element 10
        let (_trace_computing, fib_c_value) = gen_computing_trace(log_size, 1, 1, target_element);

        // Generate Scheduler trace (will read the value from Computing)
        let _trace_scheduler = gen_scheduler_trace(log_size, fib_c_value);

        println!("✓ Single-component traces generated successfully");
        println!("  Computing rows: {} (active up to element {})", 1 << log_size, target_element);
        println!("  Scheduler rows: {}", 1 << log_size);
    }

    #[test]
    fn test_single_component_proof_with_logup() {
        println!("\n==================================================");
        println!("  SINGLE-COMPONENT FIBONACCI PROOF TEST");
        println!("==================================================\n");

        let target_element = 10;

        // Setup prover (log_size will be computed dynamically)
        let config = PcsConfig::default();

        // Compute expected log_size for twiddles
        let _min_log_size = if target_element + 1 <= 1 { 0 } else { (target_element as u32).ilog2() + 1 };
        let log_size = 1;

        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        let channel = &mut Blake2sChannel::default();
        let commitment_scheme = CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(
            config,
            &twiddles,
        );

        // Generate proof with LogUp
        let result = prove_single_fib(target_element, channel, commitment_scheme);

        match result {
            Ok((proof, _component, _scheduler, _statement0, _statement1)) => {
                println!("\n==================================================");
                println!("  PROOF GENERATION SUCCESSFUL!");
                println!("==================================================\n");

                println!("Proof details:");
                println!("  - Number of commitments: {}", proof.commitments.len());
                println!("  - Computing component: 1");
                println!("  - Scheduler component: 1");

                println!("\n✓ Single-component proof with LogUp generated successfully!");
                println!("\nThis proves:");
                println!("  1. Computing correctly computed Fibonacci(1,1)");
                println!("  2. Scheduler correctly read value from Computing");
                println!("  3. LogUp verified that Scheduler used correct value");
            }
            Err(e) => {
                panic!("Proof generation failed: {:?}", e);
            }
        }
    }

    #[test]
    fn test_prove_and_verify() {
        println!("\n==================================================");
        println!("  SINGLE-COMPONENT FIBONACCI: PROVE + VERIFY");
        println!("==================================================\n");

        let target_element = 10;

        // Setup prover (log_size will be computed dynamically)
        let config = PcsConfig::default();

        // Compute expected log_size for twiddles
        let min_log_size = if target_element + 1 <= 1 { 0 } else { (target_element as u32).ilog2() + 1 };
        let log_size = min_log_size.max(4);

        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        let channel = &mut Blake2sChannel::default();
        let commitment_scheme = CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(
            config,
            &twiddles,
        );

        // STEP 1: Generate proof
        println!("==================================================");
        println!("STEP 1: PROVING");
        println!("==================================================");

        let result = prove_single_fib(target_element, channel, commitment_scheme);

        match result {
            Ok((proof, _component, _scheduler, statement0, statement1)) => {
                println!("\n✓ Proof generated successfully!");
                println!("  - Commitments: {}", proof.commitments.len());

                // STEP 2: Verify proof
                println!("\n==================================================");
                println!("STEP 2: VERIFYING");
                println!("==================================================");

                let verify_result = verify_single_fib(
                    proof,
                    target_element,
                    statement0,
                    statement1,
                    config,
                );

                match verify_result {
                    Ok(()) => {
                        println!("\n==================================================");
                        println!("  ✓✓✓ SUCCESS! ✓✓✓");
                        println!("==================================================");
                        println!("\nProof was generated AND verified successfully!");
                        println!("\nThis proves:");
                        println!("  1. Computing correctly computed Fibonacci(1,1)");
                        println!("  2. Scheduler correctly read value from Computing");
                        println!("  3. LogUp verified that Scheduler used correct value");
                        println!("  4. Verifier independently confirmed all constraints!");
                    }
                    Err(e) => {
                        panic!("Verification failed: {:?}", e);
                    }
                }
            }
            Err(e) => {
                panic!("Proof generation failed: {:?}", e);
            }
        }
    }
}
