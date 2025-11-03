//! Improved Dual Component Example - Proper Pattern
//!
//! 🎯 KEY CONCEPT: Component Composition in STARKs
//!
//! This example demonstrates the CORRECT pattern for component cooperation:
//!
//! 1. **ComputingComponent** (The "Worker"):
//!    - Generates inputs: [10, 10, 10, ...]
//!    - Computes outputs: [10^5+1, 10^5+1, ...] = [100001, 100001, ...]
//!    - Has CONSTRAINTS to verify: output = input^5 + 1
//!    - Uses LogUp to PROVIDE verified (input, output) pairs
//!
//! 2. **SchedulingComponent** (The "Requester"):
//!    - Does NOT compute anything!
//!    - Simply COPIES input/output from Computing
//!    - Has NO constraints (only LogUp)
//!    - Uses LogUp to REQUEST (input, output) pairs
//!
//! 3. **LogUp Verification**:
//!    - Ensures both components use IDENTICAL data
//!    - Claimed sums must cancel out (sum = 0)
//!    - This proves Scheduling's data matches Computing's verified data
//!
//! 🔑 WHY THIS PATTERN?
//! - Computing has complex logic → keep it in one place
//! - Scheduling just needs results → copy via LogUp
//! - Same pattern as scheduler_poseidon (professional approach)
//!
//! 📊 COMPARISON TO ORIGINAL docs_component:
//! - Original (WRONG): Scheduling computes, Computing copies
//! - This (RIGHT): Computing computes, Scheduling copies
//! - This matches production pattern (scheduler_poseidon)

use itertools::chain;
use num_traits::One;
use stwo::core::air::Component;
use stwo::core::channel::Channel;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::TreeVec;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::ComponentProver;
use stwo_constraint_framework::{
    relation, EvalAtRow, FrameworkComponent, FrameworkEval, InfoEvaluator, RelationEntry,
    TraceLocationAllocator, PREPROCESSED_TRACE_IDX,
};

const CONSTRAINT_EVAL_BLOWUP_FACTOR: u32 = 1;

// ============================================================================
// CONCEPT: Lookup Relations (LogUp)
// ============================================================================
// This defines the "language" both components use to communicate.
// Think of it as a shared ID system:
//   - Computing says: "I provide (input, output) pair with ID X"
//   - Scheduling says: "I request (input, output) pair with ID X"
//   - LogUp verifies IDs match!
//
// The "1" means we hash 1 field element to create the lookup ID.
// For Poseidon (16 elements), we'd use relation!(PoseidonElements, 16).
relation!(ComputationLookupElements, 1);

pub type SchedulingComponent = FrameworkComponent<SchedulingEval>;
pub type ComputingComponent = FrameworkComponent<ComputingEval>;

// ============================================================================
// CONCEPT: Component Container
// ============================================================================
// This struct holds both components together.
// In a real application, you might have many components:
//   - Scheduler, Poseidon, Range checks, Bitwise ops, Memory access, etc.
// Each component proves a piece of the computation.
pub struct Components {
    scheduling_component: SchedulingComponent,
    computing_component: ComputingComponent,
}

impl Components {
    pub fn new(
        statement0: &ComponentsStatement0,
        lookup_elements: &ComputationLookupElements,
        statement1: &ComponentsStatement1,
    ) -> Self {
        let tree_span_provider = &mut TraceLocationAllocator::default();

        let scheduling_component = SchedulingComponent::new(
            tree_span_provider,
            SchedulingEval {
                log_size: statement0.log_size,
                lookup_elements: lookup_elements.clone(),
            },
            statement1.scheduling_claimed_sum,
        );

        let computing_component = ComputingComponent::new(
            tree_span_provider,
            ComputingEval {
                log_size: statement0.log_size,
                lookup_elements: lookup_elements.clone(),
            },
            statement1.computing_claimed_sum,
        );

        Self {
            scheduling_component,
            computing_component,
        }
    }

    pub fn components(&self) -> Vec<&dyn Component> {
        chain![[
            &self.scheduling_component as &dyn Component,
            &self.computing_component as &dyn Component
        ]]
        .collect()
    }

    pub fn component_provers(&self) -> Vec<&dyn ComponentProver<SimdBackend>> {
        chain![[
            &self.scheduling_component as &dyn ComponentProver<SimdBackend>,
            &self.computing_component as &dyn ComponentProver<SimdBackend>
        ]]
        .collect()
    }
}

// ============================================================================
// CONCEPT: Public Statement (Part 0 - Before LogUp)
// ============================================================================
// This is PUBLIC data that both prover and verifier know.
// Statement0 contains info needed BEFORE drawing lookup elements:
//   - log_size: How many rows in the trace (2^log_size rows)
//   - This gets mixed into Fiat-Shamir (affects randomness)
//
// Think of it as: "I'm proving computation over N rows"
pub struct ComponentsStatement0 {
    log_size: u32,
}

impl ComponentsStatement0 {
    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_u64(self.log_size as u64);
    }

    pub fn log_sizes(&self) -> TreeVec<Vec<u32>> {
        let mut log_sizes = vec![];

        log_sizes.push(
            scheduling_info()
                .mask_offsets
                .as_cols_ref()
                .map_cols(|_| self.log_size),
        );

        log_sizes.push(
            computing_info()
                .mask_offsets
                .as_cols_ref()
                .map_cols(|_| self.log_size),
        );

        let mut tree_vec = TreeVec::concat_cols(log_sizes.into_iter());
        tree_vec[PREPROCESSED_TRACE_IDX] = vec![];
        tree_vec
    }
}

// ============================================================================
// CONCEPT: Public Statement (Part 1 - After LogUp)
// ============================================================================
// Statement1 contains the LogUp claimed sums:
//   - Each component claims: "My LogUp sum is X"
//   - For valid proof: scheduling_sum + computing_sum MUST = 0
//
// Why must they sum to zero?
//   - Scheduling does: +1/input - 1/output  (requesting)
//   - Computing does:  -1/input + 1/output  (providing)
//   - If data matches: (+1-1) + (-1+1) = 0 ✓
pub struct ComponentsStatement1 {
    scheduling_claimed_sum: SecureField,
    computing_claimed_sum: SecureField,
}

impl ComponentsStatement1 {
    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_felts(&[self.scheduling_claimed_sum, self.computing_claimed_sum]);
    }
}

fn scheduling_info() -> InfoEvaluator {
    let component = SchedulingEval {
        log_size: 1,
        lookup_elements: ComputationLookupElements::dummy(),
    };
    component.evaluate(InfoEvaluator::empty())
}

fn computing_info() -> InfoEvaluator {
    let component = ComputingEval {
        log_size: 1,
        lookup_elements: ComputationLookupElements::dummy(),
    };
    component.evaluate(InfoEvaluator::empty())
}

// ============================================================================
// CONCEPT: Scheduling Component (The "Requester")
// ============================================================================
// This component:
//   1. Has NO constraints (no mathematical verification)
//   2. Only uses LogUp to REQUEST data from Computing
//   3. Trace structure: [input_col, output_col]
//
// 🔑 KEY INSIGHT: Scheduling doesn't verify anything!
//    It just says: "I need (10, 100001) pairs"
//    Computing verifies that 100001 = 10^5 + 1
//    LogUp ensures they use the same pairs
/// Scheduling component - ONLY copies data, NO computation, NO constraints
#[derive(Clone)]
pub struct SchedulingEval {
    log_size: u32,
    lookup_elements: ComputationLookupElements,
}

impl FrameworkEval for SchedulingEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + CONSTRAINT_EVAL_BLOWUP_FACTOR
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // Read trace columns (2 columns total)
        let input_col = eval.next_trace_mask();
        let output_col = eval.next_trace_mask();

        // ========================================================================
        // CONCEPT: LogUp Requesting (Consumer Side)
        // ========================================================================
        // add_to_relation() adds to LogUp sum:
        //   coefficient / lookup_id
        //
        // We do:
        //   +1 / hash(input)  → requests "input"
        //   -1 / hash(output) → requests "output"
        //
        // This creates fraction: (hash(output) - hash(input)) / (hash(input) * hash(output))
        //
        // Computing will do the OPPOSITE signs (-1/input, +1/output)
        // When summed: our fraction CANCELS with Computing's → total = 0 ✓
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::one(), // +1 coefficient
            &[input_col], // value to hash
        ));

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::one(), // -1 coefficient
            &[output_col], // value to hash
        ));

        // finalize_logup_in_pairs() processes the two entries together
        // This is more efficient than processing them separately
        eval.finalize_logup_in_pairs();
        eval
    }
}

// ============================================================================
// CONCEPT: Computing Component (The "Worker/Provider")
// ============================================================================
// This component:
//   1. Has CONSTRAINTS to verify: output = input^5 + 1
//   2. Uses LogUp to PROVIDE verified (input, output) pairs
//   3. Trace structure: [input_col, intermediate_col, output_col]
//
// 🔑 KEY INSIGHT: Computing does ALL the work!
//    - Computes x^5 + 1 in trace generation (off-chain)
//    - Proves correctness with constraints (on-chain)
//    - Provides verified pairs to Scheduling via LogUp
/// Computing component - Generates data, computes x^5+1, AND has all constraints
#[derive(Clone)]
pub struct ComputingEval {
    log_size: u32,
    lookup_elements: ComputationLookupElements,
}

impl FrameworkEval for ComputingEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + CONSTRAINT_EVAL_BLOWUP_FACTOR
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // Read trace columns (3 columns total)
        let input_col = eval.next_trace_mask();
        let intermediate_col = eval.next_trace_mask();
        let output_col = eval.next_trace_mask();

        // ========================================================================
        // CONCEPT: Algebraic Constraints (The Math!)
        // ========================================================================
        // These constraints verify the computation is correct.
        // Verifier checks these hold for EVERY row.
        //
        // Constraint 1: intermediate = input^3
        //   We check: intermediate - input^3 = 0
        //   Example: 1000 - 10*10*10 = 1000 - 1000 = 0 ✓
        eval.add_constraint(
            intermediate_col.clone() - input_col.clone() * input_col.clone() * input_col.clone(),
        );

        // Constraint 2: output = input^5 + 1 = intermediate * input^2 + 1
        //   We check: output - (intermediate * input^2 + 1) = 0
        //   Example: 100001 - (1000*10*10 + 1) = 100001 - 100001 = 0 ✓
        eval.add_constraint(
            output_col.clone()
                - intermediate_col.clone() * input_col.clone() * input_col.clone()
                - E::F::one(),
        );

        // ========================================================================
        // CONCEPT: LogUp Providing (Provider Side)
        // ========================================================================
        // Now that we've PROVEN output = input^5 + 1 with constraints,
        // we offer these verified (input, output) pairs to other components.
        //
        // We use OPPOSITE signs from Scheduling:
        //   -1 / hash(input)  → provides "input"
        //   +1 / hash(output) → provides "output"
        //
        // This creates fraction: (hash(input) - hash(output)) / (hash(input) * hash(output))
        //
        // When Scheduling requests same pairs with opposite signs:
        //   Their fraction + Our fraction = 0 ✓
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::one(), // -1 coefficient (opposite of Scheduling's +1)
            &[input_col],
        ));

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::one(), // +1 coefficient (opposite of Scheduling's -1)
            &[output_col],
        ));

        eval.finalize_logup_in_pairs();
        eval
    }
}

#[cfg(test)]
use stwo::core::fields::m31::M31;
#[cfg(test)]
use stwo::core::fields::FieldExpOps;
#[cfg(test)]
use stwo::core::poly::circle::CanonicCoset;
#[cfg(test)]
use stwo::prover::backend::simd::column::BaseColumn;
#[cfg(test)]
use stwo::prover::backend::simd::m31::LOG_N_LANES;
#[cfg(test)]
use stwo::prover::backend::simd::qm31::PackedSecureField;
#[cfg(test)]
use stwo::prover::poly::circle::CircleEvaluation;
#[cfg(test)]
use stwo::prover::poly::BitReversedOrder;
#[cfg(test)]
use stwo_constraint_framework::{LogupTraceGenerator, Relation};

// ============================================================================
// CONCEPT: Lookup Data (Sharing Results Between Components)
// ============================================================================
// This struct holds the (input, output) pairs that Computing generates.
// Think of it as a "results package" that Computing hands to Scheduling.
//
// In real apps (like scheduler_poseidon), this would be:
//   - initial_state[8][16]: 8 instances × 16 elements
//   - final_state[8][16]: 8 instances × 16 elements
/// Data structure to pass computed values from Computing to Scheduling
#[cfg(test)]
pub struct ComputationLookupData {
    pub inputs: CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
    pub outputs: CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
}

// ============================================================================
// CONCEPT: Trace Generation (Off-Chain Computation)
// ============================================================================
// This function generates the Computing component's trace.
// This runs on the PROVER side ONLY (not verified on-chain).
//
// Steps:
//   1. Generate inputs (we use all 10s for simplicity)
//   2. Compute x^3 (intermediate helper for constraints)
//   3. Compute x^5 + 1 (the actual output)
//   4. Return trace columns for Computing (3 columns)
//   5. Return lookup data for Scheduling to copy
//
// 🔑 KEY: This is normal Rust code! Use .pow(), loops, whatever you need.
//         Constraints will verify this computation is correct.
/// Computing component: Generates inputs, computes x^5+1, returns data for Scheduling
#[cfg(test)]
fn gen_computing_trace(
    log_size: u32,
) -> (
    Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    ComputationLookupData,
) {
    // Step 1: Generate inputs (all 10s for educational simplicity)
    // In production: might be transaction data, user inputs, etc.
    let input_col = BaseColumn::from_iter((0..(1 << log_size)).map(|_| M31::from(10)));

    // Step 2: Compute intermediate = x^3
    // This helps split x^5 into x^3 * x^2 (reduces constraint degree)
    let intermediate_col = BaseColumn::from_iter(input_col.as_slice().iter().map(|&v| v.pow(3)));

    // Step 3: Compute output = x^5 + 1
    // This is the actual computation we're proving
    // Example: 10^5 + 1 = 100,000 + 1 = 100,001
    let output_col = BaseColumn::from_iter(
        input_col
            .as_slice()
            .iter()
            .map(|&v| v.pow(5) + M31::from(1)),
    );

    let domain = CanonicCoset::new(log_size).circle_domain();

    // Create trace evaluations for Computing component
    let input_eval = CircleEvaluation::new(domain, input_col.clone());
    let intermediate_eval = CircleEvaluation::new(domain, intermediate_col);
    let output_eval = CircleEvaluation::new(domain, output_col.clone());

    // Package results for Scheduling to copy
    let lookup_data = ComputationLookupData {
        inputs: CircleEvaluation::new(domain, input_col),
        outputs: CircleEvaluation::new(domain, output_col),
    };

    // Return both trace and lookup data
    (
        vec![input_eval, intermediate_eval, output_eval],
        lookup_data,
    )
}

// ============================================================================
// CONCEPT: Trace Copying (The "Requester" Pattern)
// ============================================================================
// Scheduling's trace generation is TRIVIAL - just copy!
//
// This demonstrates the key pattern:
//   - Computing does heavy lifting (x^5 + 1 computation + constraints)
//   - Scheduling just copies results
//   - LogUp ensures they match
//
// 🔑 WHY? Scheduling might be part of a larger system that needs these
//         results but doesn't want to duplicate Computing's constraints.
//         Think: a VM that needs Poseidon hashes but doesn't want to
//         implement Poseidon constraints itself.
/// Scheduling component: ONLY copies data from Computing (doesn't compute!)
#[cfg(test)]
fn gen_scheduling_trace(
    _log_size: u32,
    computation_lookup_data: &ComputationLookupData,
) -> Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>> {
    // Simply COPY - no computation!
    // In scheduler_poseidon, this copies 8 × (16 input + 16 output) per row
    vec![
        computation_lookup_data.inputs.clone(),  // COPY input
        computation_lookup_data.outputs.clone(), // COPY output
    ]
}

#[cfg(test)]
fn gen_scheduling_logup_trace(
    log_size: u32,
    scheduling_col_1: &CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
    scheduling_col_2: &CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
    lookup_elements: &ComputationLookupElements,
) -> (
    Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    SecureField,
) {
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    let mut col_gen = logup_gen.new_col();
    for row in 0..(1 << (log_size - LOG_N_LANES)) {
        let scheduling_input: PackedSecureField =
            lookup_elements.combine(&[scheduling_col_1.data[row]]);
        let scheduling_output: PackedSecureField =
            lookup_elements.combine(&[scheduling_col_2.data[row]]);
        col_gen.write_frac(
            row,
            scheduling_output - scheduling_input,
            scheduling_input * scheduling_output,
        );
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
}

#[cfg(test)]
fn gen_computing_logup_trace(
    log_size: u32,
    computing_col_1: &CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
    computing_col_3: &CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
    lookup_elements: &ComputationLookupElements,
) -> (
    Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    SecureField,
) {
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    let mut col_gen = logup_gen.new_col();
    for row in 0..(1 << (log_size - LOG_N_LANES)) {
        let computing_input: PackedSecureField =
            lookup_elements.combine(&[computing_col_1.data[row]]);
        let computing_output: PackedSecureField =
            lookup_elements.combine(&[computing_col_3.data[row]]);
        col_gen.write_frac(
            row,
            computing_input - computing_output,
            computing_input * computing_output,
        );
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;
    use stwo::core::channel::Blake2sChannel;
    use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
    use stwo::core::proof::StarkProof;
    use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use stwo::core::vcs::MerkleHasher;
    use stwo::core::verifier::verify;
    use stwo::prover::backend::Column;
    use stwo::prover::poly::circle::PolyOps;
    use stwo::prover::{prove, CommitmentSchemeProver};

    use super::*;

    struct ComponentsProof<H: MerkleHasher> {
        statement0: ComponentsStatement0,
        statement1: ComponentsStatement1,
        stark_proof: StarkProof<H>,
    }

    #[test]
    fn test_docs_component_proper_prove_and_verify() {
        println!("=== Improved Dual Component Example ===");
        println!("Computing generates data, Scheduling only copies");
        println!("Log size: {}", LOG_N_LANES);
        println!("Number of rows: {}\n", 1 << LOG_N_LANES);

        let log_size = LOG_N_LANES;
        let config = PcsConfig::default();

        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(
                log_size + CONSTRAINT_EVAL_BLOWUP_FACTOR + config.fri_config.log_blowup_factor,
            )
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

        println!("Step 2: Computing generates inputs and computes x^5+1...");
        let (computing_trace, computation_lookup_data) = gen_computing_trace(log_size);
        println!("  ✓ Computing generated {} rows", 1 << log_size);

        // Show first few rows
        println!("\n  First 5 rows of Computing trace:");
        for i in 0..5.min(1 << log_size) {
            let input = computing_trace[0].values.at(i);
            let intermediate = computing_trace[1].values.at(i);
            let output = computing_trace[2].values.at(i);
            println!(
                "    Row {}: input={}, intermediate={}^3={}, output={}^5+1={}",
                i, input, input, intermediate, input, output
            );
        }
        println!();

        println!("Step 3: Scheduling COPIES data from Computing...");
        let scheduling_trace = gen_scheduling_trace(log_size, &computation_lookup_data);
        println!("  ✓ Scheduling copied data (no computation!)");

        // Show that Scheduling has same data as Computing
        println!("\n  First 5 rows of Scheduling trace (should match Computing):");
        for i in 0..5.min(1 << log_size) {
            let input = scheduling_trace[0].values.at(i);
            let output = scheduling_trace[1].values.at(i);
            println!("    Row {}: input={}, output={}", i, input, output);
        }
        println!();

        let statement0 = ComponentsStatement0 { log_size };
        statement0.mix_into(channel);

        println!("Step 4: Committing trace columns...");
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals([scheduling_trace.clone(), computing_trace.clone()].concat());
        tree_builder.commit(channel);

        println!("Step 5: Drawing lookup elements from channel...");
        let lookup_elements = ComputationLookupElements::draw(channel);

        println!("Step 6: Generating LogUp interaction columns...");
        let (scheduling_logup_cols, scheduling_claimed_sum) = gen_scheduling_logup_trace(
            log_size,
            &scheduling_trace[0],
            &scheduling_trace[1],
            &lookup_elements,
        );
        let (computing_logup_cols, computing_claimed_sum) = gen_computing_logup_trace(
            log_size,
            &computing_trace[0],
            &computing_trace[2],
            &lookup_elements,
        );

        println!("  Scheduling claimed sum: {:?}", scheduling_claimed_sum);
        println!("  Computing claimed sum: {:?}", computing_claimed_sum);
        println!(
            "  Sum check (should be zero): {:?}\n",
            scheduling_claimed_sum + computing_claimed_sum
        );

        let statement1 = ComponentsStatement1 {
            scheduling_claimed_sum,
            computing_claimed_sum,
        };
        statement1.mix_into(channel);

        println!("Step 7: Committing LogUp columns...");
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals([scheduling_logup_cols, computing_logup_cols].concat());
        tree_builder.commit(channel);

        println!("Step 8: Creating components...");
        let components = Components::new(&statement0, &lookup_elements, &statement1);

        println!("Step 9: Generating STARK proof...");
        let stark_proof =
            prove(&components.component_provers(), channel, commitment_scheme).unwrap();
        println!("  ✓ Proof generated successfully\n");

        let proof = ComponentsProof {
            statement0,
            statement1,
            stark_proof,
        };

        println!("Step 10: Verifying proof...");

        assert_eq!(
            scheduling_claimed_sum + computing_claimed_sum,
            SecureField::zero()
        );

        let statement0 = proof.statement0;
        let statement1 = proof.statement1;
        let stark_proof = proof.stark_proof;

        let channel = &mut Blake2sChannel::default();
        let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        let log_sizes = statement0.log_sizes();

        commitment_scheme.commit(stark_proof.commitments[0], &log_sizes[0], channel);
        statement0.mix_into(channel);
        commitment_scheme.commit(stark_proof.commitments[1], &log_sizes[1], channel);

        let lookup_elements = ComputationLookupElements::draw(channel);

        statement1.mix_into(channel);
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
        println!("=== PROPER PATTERN: Computing does work, Scheduling only copies ===");
    }
}
