//! Scheduler + Poseidon Component Composition Example
//!
//! 🎯 WHAT THIS PROVES
//! This example proves the correctness of 256 Poseidon2 hash computations.
//! Each hash takes a 16-element input state and produces a 16-element output state.
//!
//! Example of ONE hash:
//!   Input:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
//!   Output: [462565134, 1755139106, 2142661998, ..., 1004506017]
//!
//! 🏗️ ARCHITECTURE: Two Components Working Together
//!
//! **PoseidonComponent** (The Provider - Does the work):
//!   - GENERATES input states: [0,1,2,...,15], [16,17,...,31], etc.
//!   - COMPUTES Poseidon2 hash using:
//!     * 4 full rounds (linear layer + pow5 S-box)
//!     * 14 partial rounds (partial linear layer + pow5)
//!     * 4 more full rounds
//!   - HAS CONSTRAINTS that verify every step of the hash is correct
//!   - PROVIDES verified (input_state, output_state) pairs via LogUp
//!   - Think of it as: "I computed these hashes correctly, here's proof"
//!
//! **SchedulerComponent** (The Consumer - Just uses results):
//!   - DOES NOT compute any hashes!
//!   - Simply RECEIVES and COPIES input/output states from Poseidon
//!   - HAS NO CONSTRAINTS (completely constraint-free!)
//!   - Only has LogUp entries to REQUEST (input_state, output_state) pairs
//!   - Think of it as: "I need these hash results, give them to me"
//!
//! 🚀 BATCHING OPTIMIZATION: 8 Hashes Per Row
//!
//! Instead of 1 hash per row (naive approach):
//!   Row structure: [input(16), output(16)] = 32 columns
//!   For 256 hashes: 256 rows × 32 columns
//!
//! We pack 8 INDEPENDENT hashes into each row:
//!   Row structure: [input₀(16), output₀(16), input₁(16), output₁(16), ..., input₇(16),
//! output₇(16)]   Total: 256 columns per row
//!   For 256 hashes: only 32 rows × 256 columns
//!
//! **Why batching matters**:
//!   - STARK proof size grows with NUMBER OF ROWS, not columns
//!   - 32 rows → ~8× smaller proof than 256 rows
//!   - Same computation, dramatically smaller proof!
//!   - This is a STARK optimization, not part of Poseidon spec
//!
//! 🔐 LOGUP VERIFICATION: Ensuring Data Matches
//!
//! LogUp (Logarithmic Derivative Lookup) ensures both components use IDENTICAL data:
//!
//! For each (input_state, output_state) pair:
//!   - Poseidon adds to sum:  -1/hash(input) + 1/hash(output)
//!   - Scheduler adds to sum:  +1/hash(input) - 1/hash(output)
//!   - If data matches: (-1+1) + (1-1) = 0 ✓
//!   - If data differs: sums DON'T cancel → proof fails!
//!
//! Final verification:
//!   poseidon_claimed_sum + scheduler_claimed_sum = 0
//!
//! This cryptographically proves Scheduler's data matches Poseidon's verified computations.
//!
//! 💡 WHY THIS PATTERN?
//!
//! **Separation of Concerns**:
//!   - Poseidon: Complex hash logic + verification (hundreds of constraints)
//!   - Scheduler: Simple data management (zero constraints)
//!   - Each component does one thing well
//!
//! **Reusability**:
//!   - Many components might need Poseidon hashes (VM, memory, etc.)
//!   - They all just "request" via LogUp, don't duplicate hash logic
//!   - Poseidon component can be tested and optimized independently
//!
//! **Production Pattern**:
//!   - This is how real systems work (Cairo VM, Polygon Miden, etc.)
//!   - Scheduler = your main computation
//!   - Poseidon = reusable crypto primitive
//!
//! 🎓 HOW TO READ THIS CODE
//!
//! 1. Start with the test function `test_scheduler_poseidon_prove_and_verify()`
//!    - See the 10-step proof generation process
//!    - Observe actual input/output values
//!
//! 2. Read `SchedulerEval::evaluate()`
//!    - Understand the batching loop (8 iterations)
//!    - See LogUp requesting pattern
//!
//! 3. Read `gen_scheduler_trace()`
//!    - See how Scheduler copies data from Poseidon
//!    - Understand column layout (256 columns per row)
//!
//! 4. For Poseidon internals, see `crates/examples/src/poseidon/mod.rs`

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

// ============================================================================
// CONCEPT: Poseidon State and Batching Configuration
// ============================================================================
// N_STATE = 16: Poseidon2 operates on 16-element states
//   - This is part of the Poseidon2 specification
//   - Each state element is a field element (M31)
//   - Example: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
//
// N_INSTANCES_PER_ROW = 8: Batching optimization
//   - Each row contains 8 INDEPENDENT Poseidon hash calls
//   - This is a STARK optimization, NOT part of Poseidon spec
//   - Reduces proof size by ~8× compared to 1 instance per row
//   - Scheduler MUST match this number to align with Poseidon
const N_STATE: usize = 16; // Poseidon2 state size
const N_INSTANCES_PER_ROW: usize = 8; // Must match Poseidon's N_INSTANCES_PER_ROW

#[cfg(test)]
const POSEIDON_LOG_EXPAND: u32 = 2; // Poseidon uses LOG_EXPAND = 2

pub type SchedulerComponent = FrameworkComponent<SchedulerEval>;

// ============================================================================
// FUNCTION: Components Container
// ============================================================================
// PURPOSE: Holds both Scheduler and Poseidon components together.
//
// WHY: In STARK proofs, you often have multiple components working together.
//      This struct manages both components and provides unified access.
//
// ORDERING: The order fields are declared here MUST match the order they're
//           created in Components::new() for trace commitment to work correctly.
//
pub struct Components {
    scheduler_component: SchedulerComponent,
    poseidon_component: PoseidonComponent,
}

impl Components {
    // ========================================================================
    // FUNCTION: Components::new()
    // ========================================================================
    // PURPOSE: Creates both components with their configurations.
    //
    // PARAMETERS:
    //   - statement0: Public info (log_size) known before LogUp
    //   - lookup_elements: Random challenges from Fiat-Shamir for LogUp
    //   - statement1: LogUp claimed sums from both components
    //
    // WHAT IT DOES:
    //   1. Creates TraceLocationAllocator - assigns column ranges to components
    //   2. Creates PoseidonComponent with its eval logic and claimed sum
    //   3. Creates SchedulerComponent with its eval logic and claimed sum
    //   4. Returns both wrapped in Components struct
    //
    // WHY THIS ORDER: Poseidon FIRST, Scheduler SECOND
    //   - This matches the trace commitment order in the test
    //   - Verifier must commit traces in the same order
    //   - If you swap order here, swap it everywhere!
    pub fn new(
        statement0: &ComponentsStatement0,
        lookup_elements: &PoseidonElements,
        statement1: &ComponentsStatement1,
    ) -> Self {
        let tree_span_provider =
            &mut TraceLocationAllocator::new_with_preprocessed_columns(&vec![]);

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

// ============================================================================
// CONCEPT: Public Statement (Part 0 - Before LogUp)
// ============================================================================
// Same concept as docs_component_proper, but simpler:
//   - Only one log_size (both components use the same size here)
//   - Gets mixed into Fiat-Shamir transcript
//
// In new_example, this has TWO log_sizes (scheduler_log_size, poseidon_log_size)
// because they can be different sizes.
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

// ============================================================================
// CONCEPT: Public Statement (Part 1 - After LogUp)
// ============================================================================
// LogUp claimed sums - MUST sum to zero for valid proof!
//
// Why must they sum to zero?
//   - Scheduler requests: +1/input_state - 1/output_state (for each of 8 instances)
//   - Poseidon provides: -1/input_state + 1/output_state (for each of 8 instances)
//   - If data matches: scheduler_sum + poseidon_sum = 0 ✓
//
// This is the SAME concept as docs_component_proper, just with:
//   - 16-element states instead of single values
//   - 8 instances per row instead of 1
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

// ============================================================================
// CONCEPT: Scheduler Component (The "Requester")
// ============================================================================
// Same pattern as SchedulingEval in docs_component_proper, but:
//   - 16-element states instead of single values
//   - 8 instances per row (batching optimization)
//   - Total: 256 columns per row (8 × 32)
//
// Trace structure per row:
//   [input₀(16), output₀(16), input₁(16), output₁(16), ..., input₇(16), output₇(16)]
//
// 🔑 KEY: Scheduler has NO constraints! Only LogUp requests.
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
        // ========================================================================
        // CONCEPT: Batched LogUp Requesting (8 instances per row)
        // ========================================================================
        // This loop processes 8 INDEPENDENT Poseidon calls in ONE row.
        // Each iteration handles one (input_state, output_state) pair.
        //
        // Compare to docs_component_proper:
        //   - There: 1 iteration, reads [input, output] (2 columns)
        //   - Here: 8 iterations, reads [input_state(16), output_state(16)] × 8 (256 columns)
        //
        // The LogUp logic is IDENTICAL to docs_component_proper:
        //   - Request input_state with coefficient -1
        //   - Request output_state with coefficient +1
        //   - Opposite signs from Poseidon → they cancel when summed

        // Process N_INSTANCES_PER_ROW instances (8 instances per row)
        for _ in 0..N_INSTANCES_PER_ROW {
            // Read input state (16 elements)
            // Example row 0, instance 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            let input_state: [_; N_STATE] = std::array::from_fn(|_| eval.next_trace_mask());

            // Read output state (16 elements after Poseidon hash)
            // Example: [462565134, 1755139106, ..., 1004506017]
            let output_state: [_; N_STATE] = std::array::from_fn(|_| eval.next_trace_mask());

            // ====================================================================
            // CONCEPT: LogUp with Multi-Element States
            // ====================================================================
            // Instead of hashing 1 value (like docs_component_proper),
            // we hash 16 values together to create the lookup ID.
            //
            // PoseidonElements.combine([v0, v1, ..., v15]) creates:
            //   lookup_id = α + v0·β + v1·β² + v2·β³ + ... + v15·β¹⁶
            // where α, β are the lookup elements drawn from Fiat-Shamir.
            //
            // Poseidon (provider) does: -1/lookup_id(input) + 1/lookup_id(output)
            // Scheduler (consumer) does: +1/lookup_id(input) - 1/lookup_id(output)
            // When summed: (-1+1) + (1-1) = 0 ✓

            eval.add_to_relation(RelationEntry::new(
                &self.lookup_elements,
                -E::EF::one(), // -1 coefficient (opposite of Poseidon's +1)
                &input_state,  // 16-element array
            ));

            eval.add_to_relation(RelationEntry::new(
                &self.lookup_elements,
                E::EF::one(),  // +1 coefficient (opposite of Poseidon's -1)
                &output_state, // 16-element array
            ));
        }

        // Process all pairs together for efficiency
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

// ============================================================================
// FUNCTION: gen_scheduler_trace()
// ============================================================================
// PURPOSE: Generate the Scheduler component's execution trace by COPYING
//          data from Poseidon's pre-computed results.
//
// PARAMETERS:
//   - log_size: log2(number of rows), e.g., 5 means 32 rows
//   - poseidon_lookup_data: Pre-computed Poseidon input/output states
//
// WHAT IT DOES:
//   1. Allocates 256 columns (32 per instance × 8 instances)
//   2. For each row, for each of 8 instances:
//      - Copies 16 input state elements from Poseidon
//      - Copies 16 output state elements from Poseidon
//   3. Returns trace as CircleEvaluations (polynomial evaluations)
//
// WHY COPY? Scheduler doesn't know HOW to compute Poseidon hash!
//   - It just needs the results
//   - Poseidon component handles all computation and verification
//   - Scheduler only proves it USES the correct results (via LogUp)
//
// COLUMN LAYOUT per row (256 columns total):
//   Instance 0: [input(16), output(16)]   cols 0-31
//   Instance 1: [input(16), output(16)]   cols 32-63
//   Instance 2: [input(16), output(16)]   cols 64-95
//   ...
//   Instance 7: [input(16), output(16)]   cols 224-255
//
// EXAMPLE for row 0, instance 0:
//   Input cols 0-15:   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
//   Output cols 16-31: [462565134, 1755139106, ..., 1004506017]
/// Generate scheduler trace
#[cfg(test)]
fn gen_scheduler_trace(
    log_size: u32,
    poseidon_lookup_data: &PoseidonLookupData,
) -> Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>> {
    // Calculate total columns: 32 (16 input + 16 output) × 8 instances = 256
    let n_cols = 32 * N_INSTANCES_PER_ROW;
    let mut trace = (0..n_cols)
        .map(|_| Col::<SimdBackend, M31>::zeros(1 << log_size))
        .collect::<Vec<_>>();

    // For each row, COPY data for all 8 instances
    for vec_index in 0..(1 << (log_size - LOG_N_LANES)) {
        for rep_i in 0..N_INSTANCES_PER_ROW {
            let col_offset = rep_i * 32;

            for state_i in 0..N_STATE {
                trace[col_offset + state_i].data[vec_index] =
                    poseidon_lookup_data.initial_state[rep_i][state_i].data[vec_index];
            }

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

// ============================================================================
// FUNCTION: gen_scheduler_logup_trace()
// ============================================================================
// PURPOSE: Generate LogUp interaction columns for the Scheduler component.
//          These columns accumulate the "LogUp sum" that will be checked
//          against Poseidon's sum.
//
// PARAMETERS:
//   - log_size: log2(number of rows)
//   - scheduler_trace: The 256-column trace we generated
//   - lookup_elements: Random challenges (α, β) from Fiat-Shamir
//
// WHAT IT DOES:
//   For each of 8 instances:
//     1. Creates one LogUp accumulation column
//     2. For each row: a. Reads 16-element input state from trace b. Reads 16-element output state
//        from trace c. Combines each state into single field element using lookup_elements d.
//        Computes LogUp fraction: (input - output) / (input * output) e. Writes fraction to
//        accumulation column
//     3. Finalizes column (computes running sum)
//   Returns: (8 LogUp columns, total claimed sum)
//
// LOGUP MATH:
//   We want to add: -1/hash(input) + 1/hash(output)
//   This equals:    (hash(input) - hash(output)) / (hash(input) * hash(output))
//
//   But we swap numerator sign to match Poseidon's opposite signs:
//   We compute:     (input_combined - output_combined) / (input_combined * output_combined)
//
//   Where input_combined = α + input[0]·β + input[1]·β² + ... + input[15]·β¹⁶
//
// WHY 8 COLUMNS? One per instance in the batched row.
//   Each instance has independent LogUp tracking.
//
// CLAIMED SUM: The final sum returned here will be added to Poseidon's sum.
//   If they cancel (sum = 0), proof is valid!
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
        let col_offset = rep_i * 32; // Each instance starts at col_offset
        let mut col_gen = logup_gen.new_col();

        for row in 0..(1 << (log_size - LOG_N_LANES)) {
            // Extract input state (16 elements) from trace columns
            let input_state: [_; N_STATE] =
                std::array::from_fn(|i| scheduler_trace[col_offset + i].data[row]);
            // Combine into single field element: α + v0·β + v1·β² + ... + v15·β¹⁶
            let input_combined: PackedSecureField = lookup_elements.combine(&input_state);

            // Extract output state (16 elements) from trace columns
            let output_state: [_; N_STATE] =
                std::array::from_fn(|i| scheduler_trace[col_offset + 16 + i].data[row]);
            // Combine into single field element
            let output_combined: PackedSecureField = lookup_elements.combine(&output_state);

            // Compute LogUp fraction for this row
            // We add: -1/input + 1/output = (input - output) / (input * output)
            col_gen.write_frac(
                row,
                input_combined - output_combined, // Numerator
                input_combined * output_combined, // Denominator
            );
        }
        col_gen.finalize_col(); // Compute running sum for this instance
    }

    // Returns: (8 LogUp columns, total claimed sum across all instances and rows)
    logup_gen.finalize_last()
}
// poseiodn (circuit) -> [1][4][6][8][2][56][3][5][][][]
// scheduling {poseidon impl} -> [inputs, outputs]
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
    fn test_scheduler_poseidon_prove_and_verify() {
        println!("=== Scheduler + Poseidon Component Composition ===");

        let log_size = LOG_N_LANES + 1;
        let config = PcsConfig::default();

        let max_log_expand = CONSTRAINT_EVAL_BLOWUP_FACTOR.max(POSEIDON_LOG_EXPAND);
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(log_size + max_log_expand + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        let channel = &mut Blake2sChannel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(vec![]);
        tree_builder.commit(channel);

        let (poseidon_trace, poseidon_lookup_data) = gen_poseidon_trace(log_size);

        for i in 0..16 {
            print!("{}", poseidon_lookup_data.initial_state[0][i].at(0));
            if i < 15 {
                print!(", ");
            }
        }
        println!("]");
        println!("    Output state (16 elements) after Poseidon hash:");
        print!("      [");
        for i in 0..16 {
            print!("{}", poseidon_lookup_data.final_state[0][i].at(0));
            if i < 15 {
                print!(", ");
            }
        }
        println!("]\n");

        let scheduler_trace = gen_scheduler_trace(log_size, &poseidon_lookup_data);

        let statement0 = ComponentsStatement0 { log_size };
        statement0.mix_into(channel);

        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals([poseidon_trace.clone(), scheduler_trace.clone()].concat());
        tree_builder.commit(channel);

        let lookup_elements = PoseidonElements::draw(channel);
        // A + (-A) = 0
        let (scheduler_logup_cols, scheduler_claimed_sum) =
            gen_scheduler_logup_trace(log_size, &scheduler_trace, &lookup_elements);

        let (poseidon_logup_cols, poseidon_claimed_sum) =
            gen_poseidon_interaction_trace(log_size, poseidon_lookup_data, &lookup_elements);

        let statement1 = ComponentsStatement1 {
            scheduler_claimed_sum,
            poseidon_claimed_sum,
        };
        statement1.mix_into(channel);

        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals([poseidon_logup_cols, scheduler_logup_cols].concat());
        tree_builder.commit(channel);

        let components = Components::new(&statement0, &lookup_elements, &statement1);

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
    }
}
