//! AIR for Poseidon2 hash function from <https://eprint.iacr.org/2023/323.pdf>.

use std::ops::{Add, AddAssign, Mul, Sub};

use itertools::Itertools;
use num_traits::One;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::StarkProof;
use stwo::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::{prove, CommitmentSchemeProver};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    relation, EvalAtRow, FrameworkComponent, FrameworkEval, LogupTraceGenerator, Relation,
    RelationEntry, TraceLocationAllocator,
};
use tracing::{info, span, Level};

// Vertical sponge construction - one Poseidon per row
const N_STATE: usize = 16;
pub const RATE: usize = 8; // First 8 elements absorb message
#[allow(dead_code)]
const CAPACITY: usize = 8; // Last 8 elements for security
const N_PARTIAL_ROUNDS: usize = 14;
const N_HALF_FULL_ROUNDS: usize = 4;
const FULL_ROUNDS: usize = 2 * N_HALF_FULL_ROUNDS;
// Columns: 8 message + 16 initial_state + intermediate_states + 16 final_state
// NOTE: multiplicity is NOT a trace column - it's only in LookupData for gen_interaction_trace
const N_COLUMNS: usize = RATE + N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS + N_STATE;
const LOG_EXPAND: u32 = 2;
// TODO(shahars): Use poseidon's real constants.
const EXTERNAL_ROUND_CONSTS: [[BaseField; N_STATE]; 2 * N_HALF_FULL_ROUNDS] =
    [[BaseField::from_u32_unchecked(1234); N_STATE]; 2 * N_HALF_FULL_ROUNDS];
const INTERNAL_ROUND_CONSTS: [BaseField; N_PARTIAL_ROUNDS] =
    [BaseField::from_u32_unchecked(1234); N_PARTIAL_ROUNDS];

pub type PoseidonComponent = FrameworkComponent<PoseidonEval>;

relation!(PoseidonElements, N_STATE);

#[derive(Clone)]
pub struct PoseidonEval {
    pub log_n_rows: u32,
    pub lookup_elements: PoseidonElements,
    pub claimed_sum: SecureField,
    pub is_first_id: PreProcessedColumnId,
    pub is_active_id: PreProcessedColumnId,
    pub n_active_rows: usize,
}
impl FrameworkEval for PoseidonEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + LOG_EXPAND
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        eval_poseidon_sponge_constraints(
            &mut eval,
            &self.lookup_elements,
            &self.is_first_id,
            &self.is_active_id,
        );
        eval
    }
}

#[inline(always)]
/// Applies the M4 MDS matrix described in <https://eprint.iacr.org/2023/323.pdf> 5.1.
fn apply_m4<F>(x: [F; 4]) -> [F; 4]
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
fn apply_external_round_matrix<F>(state: &mut [F; 16])
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
fn apply_internal_round_matrix<F>(state: &mut [F; 16])
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    // TODO(shahars): Check that these coefficients are good according to section  5.3 of Poseidon2
    // paper.
    let sum = state[1..]
        .iter()
        .cloned()
        .fold(state[0].clone(), |acc, s| acc + s);
    state.iter_mut().enumerate().for_each(|(i, s)| {
        // TODO(andrew): Change to rotations.
        *s = s.clone() * BaseField::from_u32_unchecked(1 << (i + 1)) + sum.clone();
    });
}

fn pow5<F: FieldExpOps>(x: F) -> F {
    let x2 = x.clone() * x.clone();
    let x4 = x2.clone() * x2.clone();
    x4 * x.clone()
}

/// Evaluates Poseidon sponge constraints with vertical chaining.
///
/// Trace structure per row:
/// - 8 message columns: new message to absorb
/// - 16 initial_state columns: state before permutation
/// - N intermediate columns: intermediate states during permutation
/// - 16 final_state columns: state after permutation (for chaining to next row)
///
/// Constraints:
/// 1. First row: capacity must be zero (initial_state[8..16] = 0)
/// 2. Transition constraints (disabled for first row AND padding rows):
///    - Rate: initial_state[0..8] = prev_final_state[0..8] + message[0..8]
///    - Capacity: initial_state[8..16] = prev_final_state[8..16] (unchanged)
/// 3. Poseidon permutation: final_state = P(initial_state)
/// 4. LogUp: Only active rows participate (multiplicity = is_active)
///
/// Padding rows (is_active = 0):
/// - Chaining constraints are DISABLED (no dependency on previous row)
/// - LogUp is DISABLED (multiplicity = 0, they don't participate)
/// - Poseidon constraints remain ENABLED (for simplicity, padding still computes valid Poseidon)
/// - This provides full isolation: padding rows cannot affect active rows
pub fn eval_poseidon_sponge_constraints<E: EvalAtRow>(
    eval: &mut E,
    lookup_elements: &PoseidonElements,
    is_first_id: &PreProcessedColumnId,
    is_active_id: &PreProcessedColumnId,
) {
    use stwo_constraint_framework::ORIGINAL_TRACE_IDX;

    let is_first_val = eval.get_preprocessed_column(is_first_id.clone());
    let is_active_val = eval.get_preprocessed_column(is_active_id.clone());

    // Read ALL columns using next_interaction_mask so we can access previous row values
    // Column layout: [message(8), initial_state(16), intermediate_states, final_state(16)]

    // Read message (8 elements) - current row only
    let message: [E::F; RATE] = std::array::from_fn(|_| {
        let [curr, _prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
        curr
    });

    // Read initial state (16 elements) - current and previous row
    let initial_state_curr: [E::F; N_STATE] = std::array::from_fn(|_| {
        let [curr, _prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
        curr
    });

    // Read intermediate states from first 4 full rounds
    let intermediate_full1: [[E::F; N_STATE]; N_HALF_FULL_ROUNDS] = std::array::from_fn(|_| {
        std::array::from_fn(|_| {
            let [curr, _prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
            curr
        })
    });

    // Read partial round intermediate states
    let intermediate_partial: [E::F; N_PARTIAL_ROUNDS] = std::array::from_fn(|_| {
        let [curr, _prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
        curr
    });

    // Read intermediate states from last 4 full rounds
    let intermediate_full2: [[E::F; N_STATE]; N_HALF_FULL_ROUNDS] = std::array::from_fn(|_| {
        std::array::from_fn(|_| {
            let [curr, _prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
            curr
        })
    });

    // Read final state (16 elements) - current and PREVIOUS row
    let mut final_state_curr_vec = Vec::with_capacity(N_STATE);
    let mut final_state_prev_vec = Vec::with_capacity(N_STATE);
    for _ in 0..N_STATE {
        let [curr, prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
        final_state_curr_vec.push(curr);
        final_state_prev_vec.push(prev);
    }
    let final_state_curr: [E::F; N_STATE] =
        std::array::from_fn(|i| final_state_curr_vec[i].clone());
    let final_state_prev: [E::F; N_STATE] =
        std::array::from_fn(|i| final_state_prev_vec[i].clone());

    // Constraint 1: First row capacity must be zero
    for i in RATE..N_STATE {
        eval.add_constraint(is_first_val.clone() * initial_state_curr[i].clone());
    }

    // Constraint 2: Transition constraints (chaining between rows)
    // CRITICAL: Only enabled for active rows (excluding first row)
    // Padding rows (is_active = 0) have chaining DISABLED for isolation
    let enable_chaining = is_active_val.clone() * (E::F::one() - is_first_val.clone());

    // Rate part: initial_state[0..8] = final_state_prev[0..8] + message[0..8]
    for i in 0..RATE {
        let expected = final_state_prev[i].clone() + message[i].clone();
        eval.add_constraint(enable_chaining.clone() * (initial_state_curr[i].clone() - expected));
    }

    // Capacity part: initial_state[8..16] = final_state_prev[8..16]
    for i in RATE..N_STATE {
        eval.add_constraint(
            enable_chaining.clone() * (initial_state_curr[i].clone() - final_state_prev[i].clone()),
        );
    }

    // Constraint 3: Poseidon permutation correctness
    // Verify that the intermediate states match the permutation computation
    let mut state = initial_state_curr.clone();

    // 4 full rounds
    for round in 0..N_HALF_FULL_ROUNDS {
        for i in 0..N_STATE {
            state[i] = state[i].clone() + E::F::from(EXTERNAL_ROUND_CONSTS[round][i]);
        }
        apply_external_round_matrix(&mut state);
        state = std::array::from_fn(|i| pow5_expr(state[i].clone()));

        // Verify intermediate state matches trace
        for i in 0..N_STATE {
            eval.add_constraint(state[i].clone() - intermediate_full1[round][i].clone());
        }
        state = intermediate_full1[round].clone();
    }

    // Partial rounds
    for round in 0..N_PARTIAL_ROUNDS {
        state[0] = state[0].clone() + E::F::from(INTERNAL_ROUND_CONSTS[round]);
        apply_internal_round_matrix(&mut state);
        state[0] = pow5_expr(state[0].clone());

        // Verify intermediate state matches trace
        eval.add_constraint(state[0].clone() - intermediate_partial[round].clone());
        state[0] = intermediate_partial[round].clone();
    }

    // 4 full rounds
    for round in 0..N_HALF_FULL_ROUNDS {
        for i in 0..N_STATE {
            state[i] =
                state[i].clone() + E::F::from(EXTERNAL_ROUND_CONSTS[round + N_HALF_FULL_ROUNDS][i]);
        }
        apply_external_round_matrix(&mut state);
        state = std::array::from_fn(|i| pow5_expr(state[i].clone()));

        // Verify intermediate state matches trace
        for i in 0..N_STATE {
            eval.add_constraint(state[i].clone() - intermediate_full2[round][i].clone());
        }
        state = intermediate_full2[round].clone();
    }

    // Verify final state matches computed state
    for i in 0..N_STATE {
        eval.add_constraint(state[i].clone() - final_state_curr[i].clone());
    }

    // LogUp: Provide initial and final state lookups
    // IMPORTANT: Use is_active as multiplicity in BOTH places:
    // 1. gen_interaction_trace: multiplies fractions by multiplicity from LookupData
    // 2. constraints: multiplies relation entries by is_active from preprocessed column
    //
    // Interaction trace: Σ(multiplicity * (1/(initial) - 1/(final)))
    // Constraints: Σ(is_active * (1/(initial) - 1/(final)))
    //
    // Since multiplicity = is_active, these match!
    // For active rows (is_active=1): both contribute full fraction
    // For padding rows (is_active=0): both contribute 0

    let is_active_ef = E::EF::from(is_active_val.clone());

    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        is_active_ef.clone(), // multiplicity = is_active
        &initial_state_curr,
    ));
    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        -is_active_ef, // multiplicity = -is_active
        &final_state_curr,
    ));

    eval.finalize_logup_in_pairs();
}

/// Helper function to compute x^5 for constraint expressions
fn pow5_expr<F: Clone + std::ops::Mul<Output = F>>(x: F) -> F {
    let x2 = x.clone() * x.clone();
    let x4 = x2.clone() * x2.clone();
    x4 * x
}

pub struct LookupData {
    pub initial_state: [Col<SimdBackend, BaseField>; N_STATE],
    pub final_state: [Col<SimdBackend, BaseField>; N_STATE],
    pub multiplicity: Col<SimdBackend, BaseField>, /* is_active values (1 for active, 0 for
                                                    * padding) */
}

/// Dumps trace to a file for inspection
pub fn dump_trace_to_file(
    trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    filename: &str,
) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(filename)?;
    let n_rows = trace[0].values.len();
    let n_cols = trace.len();

    writeln!(file, "Poseidon Sponge Trace Dump")?;
    writeln!(file, "==========================")?;
    writeln!(file, "Rows: {}, Columns: {}\n", n_rows, n_cols)?;
    writeln!(file, "Column layout:")?;
    writeln!(file, "  Columns 0-7:     message (RATE)")?;
    writeln!(file, "  Columns 8-23:    initial_state (16 elements)")?;
    writeln!(file, "  Columns 24-...:  intermediate states")?;
    writeln!(
        file,
        "  Columns ...-{}: final_state (16 elements)\n",
        n_cols - 1
    )?;

    // Print first few rows in detail
    let rows_to_print = std::cmp::min(8, n_rows);
    for row in 0..rows_to_print {
        writeln!(file, "\n=== ROW {} ===", row)?;

        // Message (columns 0-7)
        write!(file, "Message:       [")?;
        for col in 0..RATE {
            write!(file, "{:8}", trace[col].values.at(row).0)?;
            if col < RATE - 1 {
                write!(file, ", ")?;
            }
        }
        writeln!(file, "]")?;

        // Initial state (columns 8-23)
        write!(file, "Initial state: [")?;
        for i in 0..N_STATE {
            let col = RATE + i;
            write!(file, "{:8}", trace[col].values.at(row).0)?;
            if i < N_STATE - 1 {
                write!(file, ", ")?;
            }
        }
        writeln!(file, "]")?;

        // Final state (last 16 columns)
        write!(file, "Final state:   [")?;
        for i in 0..N_STATE {
            let col = n_cols - N_STATE + i;
            write!(file, "{:8}", trace[col].values.at(row).0)?;
            if i < N_STATE - 1 {
                write!(file, ", ")?;
            }
        }
        writeln!(file, "]")?;

        // Show chaining
        if row > 0 {
            writeln!(file, "\nChaining verification:")?;
            write!(file, "  prev_final[0..8] + msg = ")?;
            for i in 0..RATE {
                let prev_final = trace[n_cols - N_STATE + i].values.at(row - 1);
                let curr_msg = trace[i].values.at(row);
                let expected = prev_final + curr_msg;
                write!(file, "{:8} ", expected.0)?;
            }
            writeln!(file)?;
            write!(file, "  curr_initial[0..8]     = ")?;
            for i in 0..RATE {
                write!(file, "{:8} ", trace[RATE + i].values.at(row).0)?;
            }
            writeln!(file)?;
        }
    }

    writeln!(file, "\n\nDone! Check the file for complete trace.")?;
    Ok(())
}

/// Generates the is_first preprocessed column
pub fn gen_is_first_column(
    log_size: u32,
) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    use stwo::core::utils::bit_reverse_coset_to_circle_domain_order;

    let n_rows = 1 << log_size;
    let mut col = Col::<SimdBackend, BaseField>::zeros(n_rows);
    col.set(0, BaseField::from_u32_unchecked(1));
    bit_reverse_coset_to_circle_domain_order(col.as_mut_slice());

    let domain = CanonicCoset::new(log_size).circle_domain();
    CircleEvaluation::new(domain, col)
}

/// Returns the PreProcessedColumnId for is_first column
pub fn is_first_column_id(log_size: u32) -> PreProcessedColumnId {
    PreProcessedColumnId {
        id: format!("is_first_{}", log_size),
    }
}

/// Generates the is_active preprocessed column.
///
/// This column marks which rows are "active" (contain real messages) vs "padding" (unused).
/// Active rows have value 1, padding rows have value 0.
///
/// # Arguments
/// * `log_size` - Log2 of total number of rows
/// * `n_active` - Number of active rows (must be <= 2^log_size and multiple of SIMD lane size)
///
/// # Panics
/// Panics if n_active is not a multiple of (1 << LOG_N_LANES), which is required for SIMD
/// efficiency Helper function to round up n_active to nearest SIMD lane boundary
pub fn round_up_to_simd_lanes(n_active: usize) -> usize {
    let simd_lane_size = 1 << LOG_N_LANES; // 16
    if n_active == 0 {
        return 0;
    }
    ((n_active + simd_lane_size - 1) / simd_lane_size) * simd_lane_size
}

/// Generates the `is_active` preprocessed column.
///
/// # Arguments
/// * `log_size` - Log2 of trace size
/// * `n_active` - Number of active rows (should already be rounded to SIMD boundary)
///
/// # Returns
/// Column where first `n_active` rows are 1, rest are 0 (in sequential order before bit-reversal)
pub fn gen_is_active_column(
    log_size: u32,
    n_active: usize,
) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    use stwo::core::utils::bit_reverse_coset_to_circle_domain_order;

    let n_rows = 1 << log_size;

    assert!(
        n_active <= n_rows,
        "n_active ({}) exceeds n_rows ({}). Use larger log_size.",
        n_active,
        n_rows
    );

    let mut col = Col::<SimdBackend, BaseField>::zeros(n_rows);

    // Set first n_active rows to 1
    for i in 0..n_active {
        col.set(i, BaseField::from_u32_unchecked(1));
    }

    bit_reverse_coset_to_circle_domain_order(col.as_mut_slice());

    let domain = CanonicCoset::new(log_size).circle_domain();
    CircleEvaluation::new(domain, col)
}

/// Returns the PreProcessedColumnId for is_active column
///
/// # Arguments
/// * `log_size` - Log2 of trace size
/// * `n_active` - Number of active rows (should already be rounded to SIMD boundary)
pub fn is_active_column_id(log_size: u32, n_active: usize) -> PreProcessedColumnId {
    PreProcessedColumnId {
        id: format!("is_active_{}_{}", log_size, n_active),
    }
}
/// Generates trace for vertical sponge construction.
///
/// Takes a vector of messages where each message is RATE (8) elements.
/// Each row processes one message and chains the output to the next row.
///
/// Input: messages - vector of messages, each RATE elements
/// Output: trace columns + lookup data
/// Dumps trace in SEQUENTIAL order (before bit-reversal) to see true chaining
pub fn dump_trace_sequential(
    trace: &[Col<SimdBackend, BaseField>],
    filename: &str,
) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(filename)?;
    let n_rows = trace[0].len();

    writeln!(
        file,
        "=== POSEIDON SPONGE TRACE (SEQUENTIAL ORDER - BEFORE BIT-REVERSAL) ==="
    )?;
    writeln!(file, "Total rows: {}\n", n_rows)?;

    // Show ALL rows (not just first 8)
    let rows_to_show = n_rows;

    for row in 0..rows_to_show {
        writeln!(file, "Row {}: {{", row)?;

        let mut col_idx = 0;

        // Message (8 elements)
        write!(file, "  Message: [")?;
        for i in 0..RATE {
            write!(file, "{}", trace[col_idx].at(row).0)?;
            if i < RATE - 1 {
                write!(file, ", ")?;
            }
            col_idx += 1;
        }
        writeln!(file, "]")?;

        // Initial state (16 elements)
        write!(file, "  Initial state: [")?;
        for i in 0..N_STATE {
            write!(file, "{}", trace[col_idx].at(row).0)?;
            if i < N_STATE - 1 {
                write!(file, ", ")?;
            }
            col_idx += 1;
        }
        writeln!(file, "]")?;

        // Skip intermediate states
        col_idx += N_STATE * FULL_ROUNDS + N_PARTIAL_ROUNDS;

        // Final state (16 elements)
        write!(file, "  Final state: [")?;
        for i in 0..N_STATE {
            write!(file, "{}", trace[col_idx].at(row).0)?;
            if i < N_STATE - 1 {
                write!(file, ", ")?;
            }
            col_idx += 1;
        }
        writeln!(file, "]")?;

        // Chaining verification for next row
        if row < rows_to_show - 1 {
            writeln!(file, "\n  Chaining verification (for next row):")?;

            // Get current row's final state
            let final_col_start = RATE + N_STATE + N_STATE * FULL_ROUNDS + N_PARTIAL_ROUNDS;

            // Get next row's message and initial state
            let next_msg_start = 0;
            let next_initial_start = RATE;

            write!(file, "    prev_final[0..8] (rate):       [")?;
            for i in 0..RATE {
                write!(file, "{}", trace[final_col_start + i].at(row).0)?;
                if i < RATE - 1 {
                    write!(file, ", ")?;
                }
            }
            writeln!(file, "]")?;

            write!(file, "    prev_final[8..16] (capacity):  [")?;
            for i in RATE..N_STATE {
                write!(file, "{}", trace[final_col_start + i].at(row).0)?;
                if i < N_STATE - 1 {
                    write!(file, ", ")?;
                }
            }
            writeln!(file, "]")?;

            write!(file, "    next_msg:                      [")?;
            for i in 0..RATE {
                write!(file, "{}", trace[next_msg_start + i].at(row + 1).0)?;
                if i < RATE - 1 {
                    write!(file, ", ")?;
                }
            }
            writeln!(file, "]")?;

            write!(file, "    prev_final[0..8] + next_msg:   [")?;
            for i in 0..RATE {
                let sum =
                    trace[final_col_start + i].at(row) + trace[next_msg_start + i].at(row + 1);
                write!(file, "{}", sum.0)?;
                if i < RATE - 1 {
                    write!(file, ", ")?;
                }
            }
            writeln!(file, "]")?;

            write!(file, "    next_initial[0..8]:            [")?;
            for i in 0..RATE {
                write!(file, "{}", trace[next_initial_start + i].at(row + 1).0)?;
                if i < RATE - 1 {
                    write!(file, ", ")?;
                }
            }
            writeln!(file, "]")?;

            write!(
                file,
                "    next_initial[8..16] (should = prev_final[8..16]): ["
            )?;
            for i in RATE..N_STATE {
                write!(file, "{}", trace[next_initial_start + i].at(row + 1).0)?;
                if i < N_STATE - 1 {
                    write!(file, ", ")?;
                }
            }
            writeln!(file, "]")?;
        }

        writeln!(file, "}}\n")?;
    }

    // All rows are now shown, no "... more rows ..." message needed

    Ok(())
}

/// Generates trace for vertical sponge construction.
///
/// # Arguments
/// * `log_size` - Log2 of total number of rows
/// * `n_active` - Number of active rows (remaining are padding)
/// * `messages` - Vector of messages (will be extended with zeros if needed)
///
/// # Padding behavior
/// Rows >= n_active_rounded are padding rows:
/// - initial_state = [0, 0, ..., 0] (no chaining from previous row)
/// - Poseidon is still computed (for simplicity)
/// - final_state computed but not used for chaining
///
/// # SIMD Alignment
/// n_active is automatically rounded up to nearest multiple of 16 (SIMD lane size)
/// for LogUp consistency. This means a few extra rows may be treated as "active".
pub fn gen_trace(
    log_size: u32,
    n_active: usize,
    messages: Vec<[BaseField; RATE]>,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    LookupData,
) {
    use stwo::core::utils::bit_reverse_coset_to_circle_domain_order;

    let _span = span!(Level::INFO, "Generation").entered();
    assert!(log_size >= LOG_N_LANES);
    let n_rows = 1 << log_size;

    // n_active can be any value from 0 to n_rows (no SIMD alignment required)

    // Extend messages to fill all rows if needed
    let mut extended_messages = messages;
    while extended_messages.len() < n_rows {
        extended_messages.push([BaseField::from_u32_unchecked(0); RATE]);
    }

    let mut trace = (0..N_COLUMNS)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(n_rows))
        .collect_vec();
    let mut lookup_data = LookupData {
        initial_state: std::array::from_fn(|_| Col::<SimdBackend, BaseField>::zeros(n_rows)),
        final_state: std::array::from_fn(|_| Col::<SimdBackend, BaseField>::zeros(n_rows)),
        multiplicity: Col::<SimdBackend, BaseField>::zeros(n_rows),
    };

    // Generate trace row by row
    let mut prev_output: Option<[BaseField; N_STATE]> = None;

    for row in 0..n_rows {
        let mut col_index = 0;
        let message = extended_messages[row];

        // Debug: Print first few rows
        if row < 4 {
            println!("\n=== Generating Row {} ===", row);
            print!("Message: [");
            for i in 0..RATE {
                print!("{}", message[i].0);
                if i < RATE - 1 {
                    print!(", ");
                }
            }
            println!("]");
        }

        // Write message columns (8 elements)
        for i in 0..RATE {
            trace[col_index].set(row, message[i]);
            col_index += 1;
        }

        // Compute initial state
        let is_padding_row = row >= n_active;
        let mut state: [BaseField; N_STATE] = if !is_padding_row && prev_output.is_some() {
            // Not first row AND not padding: state = [prev_rate + message, prev_capacity]
            let prev = prev_output.unwrap();
            if row < 4 {
                println!("Previous output exists, computing chaining...");
                print!("  prev_output[0..8] (rate):     [");
                for i in 0..RATE {
                    print!("{}", prev[i].0);
                    if i < RATE - 1 {
                        print!(", ");
                    }
                }
                println!("]");
                print!("  prev_output[8..16] (capacity): [");
                for i in RATE..N_STATE {
                    print!("{}", prev[i].0);
                    if i < N_STATE - 1 {
                        print!(", ");
                    }
                }
                println!("]");
            }

            let new_state = std::array::from_fn(|i| {
                if i < RATE {
                    prev[i] + message[i]
                } else {
                    prev[i]
                }
            });

            if row < 4 {
                print!("  initial_state = prev_output + [message, 0...]: [");
                for i in 0..N_STATE {
                    print!("{}", new_state[i].0);
                    if i < N_STATE - 1 {
                        print!(", ");
                    }
                }
                println!("]");
            }

            new_state
        } else {
            // First row OR padding row: no chaining
            if row < 4 || (row == n_active && n_active < 20) {
                if is_padding_row {
                    println!("Padding row {}: NO CHAINING, state = ALL ZEROS", row);
                } else {
                    println!("First row: state = [message, zeros]");
                }
            }

            let new_state = if is_padding_row {
                // Padding row: ALL ZEROS (no message, no chaining)
                std::array::from_fn(|_| BaseField::from_u32_unchecked(0))
            } else {
                // First row: [message, zeros]
                std::array::from_fn(|i| {
                    if i < RATE {
                        message[i]
                    } else {
                        BaseField::from_u32_unchecked(0)
                    }
                })
            };

            if row < 4 {
                print!("  initial_state: [");
                for i in 0..N_STATE {
                    print!("{}", new_state[i].0);
                    if i < N_STATE - 1 {
                        print!(", ");
                    }
                }
                println!("]");
            }

            new_state
        };

        // Write initial state columns (16 elements)
        for i in 0..N_STATE {
            trace[col_index].set(row, state[i]);
            lookup_data.initial_state[i].set(row, state[i]);
            col_index += 1;
        }

        // Poseidon permutation
        // 4 full rounds
        (0..N_HALF_FULL_ROUNDS).for_each(|round| {
            (0..N_STATE).for_each(|i| {
                state[i] += EXTERNAL_ROUND_CONSTS[round][i];
            });
            apply_external_round_matrix(&mut state);
            state = std::array::from_fn(|i| pow5(state[i]));
            state.iter().for_each(|&s| {
                trace[col_index].set(row, s);
                col_index += 1;
            });
        });

        // Partial rounds
        (0..N_PARTIAL_ROUNDS).for_each(|round| {
            state[0] += INTERNAL_ROUND_CONSTS[round];
            apply_internal_round_matrix(&mut state);
            state[0] = pow5(state[0]);
            trace[col_index].set(row, state[0]);
            col_index += 1;
        });

        // 4 full rounds
        (0..N_HALF_FULL_ROUNDS).for_each(|round| {
            (0..N_STATE).for_each(|i| {
                state[i] += EXTERNAL_ROUND_CONSTS[round + N_HALF_FULL_ROUNDS][i];
            });
            apply_external_round_matrix(&mut state);
            state = std::array::from_fn(|i| pow5(state[i]));
            state.iter().for_each(|&s| {
                trace[col_index].set(row, s);
                col_index += 1;
            });
        });

        // Write final state columns (16 elements)
        for i in 0..N_STATE {
            trace[col_index].set(row, state[i]);
            lookup_data.final_state[i].set(row, state[i]);
            col_index += 1;
        }

        // Write multiplicity to LookupData only (not to trace!)
        // Multiplicity is used in gen_interaction_trace to weight fractions
        let is_active = if row < n_active {
            BaseField::from_u32_unchecked(1)
        } else {
            BaseField::from_u32_unchecked(0)
        };
        lookup_data.multiplicity.set(row, is_active);

        // Debug: Print final state
        if row < 4 {
            print!("After Poseidon permutation:\n  final_state (output): [");
            for i in 0..N_STATE {
                print!("{}", state[i].0);
                if i < N_STATE - 1 {
                    print!(", ");
                }
            }
            println!("]");
            println!("  → This output will be used in next row!");
        }

        // Store output for next row
        prev_output = Some(state);
    }

    if n_rows > 4 {
        println!("\n... (remaining {} rows generated) ...\n", n_rows - 4);
    }

    // Save sequential trace before bit-reversal for debugging
    let sequential_trace = trace.clone();

    // Apply bit_reverse_coset_to_circle_domain_order to main trace columns
    for col in trace.iter_mut() {
        bit_reverse_coset_to_circle_domain_order(col.as_mut_slice());
    }

    // Also bit-reverse lookup_data to match trace order
    // LogupTraceGenerator expects bit-reversed input
    for col in lookup_data.initial_state.iter_mut() {
        bit_reverse_coset_to_circle_domain_order(col.as_mut_slice());
    }
    for col in lookup_data.final_state.iter_mut() {
        bit_reverse_coset_to_circle_domain_order(col.as_mut_slice());
    }
    bit_reverse_coset_to_circle_domain_order(lookup_data.multiplicity.as_mut_slice());

    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace = trace
        .into_iter()
        .map(|eval| CircleEvaluation::new(domain, eval))
        .collect();

    // Dump sequential trace to file for verification
    println!("Dumping sequential trace (before bit-reversal)...");
    dump_trace_sequential(&sequential_trace, "poseidon_sponge_sequential.txt")
        .expect("Failed to dump sequential trace");
    println!("✅ Sequential trace dumped to: poseidon_sponge_sequential.txt");

    (trace, lookup_data)
}

pub fn gen_interaction_trace(
    log_size: u32,
    _n_active: usize,
    lookup_data: LookupData,
    lookup_elements: &PoseidonElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    use stwo::prover::backend::simd::qm31::PackedSecureField;

    let _span = span!(Level::INFO, "Generate interaction trace").entered();
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    {
        let mut col_gen = logup_gen.new_col();

        // SOLUTION: Use multiplicity column from main trace!
        // For each SIMD pack, compute fraction with multiplicity as numerator:
        // fraction = multiplicity * (1/(initial) - 1/(final))
        //          = multiplicity * (denom1 - denom0) / (denom0 * denom1)
        //
        // For active rows: multiplicity = 1, so fraction contributes to claimed_sum
        // For padding rows: multiplicity = 0, so fraction = 0 (doesn't contribute)
        //
        // This ensures: claimed_sum (from trace) == actual_sum (from constraints)

        let n_packs = 1 << (log_size - LOG_N_LANES);

        for vec_row in 0..n_packs {
            // Read multiplicity for this SIMD pack (bit-reversed data)
            let multiplicity_packed = lookup_data.multiplicity.data[vec_row];

            // Read state values
            let initial_state_packed: [PackedBaseField; N_STATE] =
                std::array::from_fn(|i| lookup_data.initial_state[i].data[vec_row]);
            let final_state_packed: [PackedBaseField; N_STATE] =
                std::array::from_fn(|i| lookup_data.final_state[i].data[vec_row]);

            let denom0: PackedSecureField = lookup_elements.combine(&initial_state_packed);
            let denom1: PackedSecureField = lookup_elements.combine(&final_state_packed);

            // Numerator = multiplicity * (denom1 - denom0)
            // For padding rows (multiplicity=0), this gives numerator=0
            let numerator = PackedSecureField::from(multiplicity_packed) * (denom1 - denom0);
            let denominator = denom0 * denom1;

            col_gen.write_frac(vec_row, numerator, denominator);
        }

        col_gen.finalize_col();
    }

    logup_gen.finalize_last()
}

/// Proves Poseidon sponge with vertical chaining.
///
/// # Arguments
/// * `log_n_rows` - Log2 of total number of rows (trace size = 2^log_n_rows)
/// * `n_real_messages` - Number of real messages (rest will be padding)
/// * `messages` - Vector of RATE-element messages to hash sequentially
/// * `config` - PCS configuration
///
/// # Padding behavior
/// If `n_real_messages < 2^log_n_rows`, remaining rows are padding rows:
/// - Padding rows have `is_active = 0`
/// - Chaining is disabled for padding rows (they don't depend on previous rows)
/// - LogUp multiplicity = 0 for padding rows (they don't participate in lookups)
/// - This allows proving with minimal computation for unused rows
pub fn prove_poseidon(
    log_n_rows: u32,
    n_real_messages: usize,
    messages: Vec<[BaseField; RATE]>,
    config: PcsConfig,
) -> (PoseidonComponent, StarkProof<Blake2sMerkleHasher>) {
    // Use exact n_real_messages without rounding
    let n_active = n_real_messages;

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + LOG_EXPAND + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    span.exit();

    // Setup protocol.
    let channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<_, Blake2sMerkleChannel>::new(config, &twiddles);

    // Preprocessed trace.
    let span = span!(Level::INFO, "Constant").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    let is_first_col = gen_is_first_column(log_n_rows);
    let is_active_col = gen_is_active_column(log_n_rows, n_active);
    let constant_trace = vec![is_first_col, is_active_col];
    tree_builder.extend_evals(constant_trace);
    tree_builder.commit(channel);
    span.exit();

    // Trace.
    let span = span!(Level::INFO, "Trace").entered();
    let (trace, lookup_data) = gen_trace(log_n_rows, n_active, messages);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();

    // Draw lookup elements.
    let lookup_elements = PoseidonElements::draw(channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let (trace, claimed_sum) =
        gen_interaction_trace(log_n_rows, n_active, lookup_data, &lookup_elements);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();

    // Prove constraints.
    let is_first_id = is_first_column_id(log_n_rows);
    let is_active_id = is_active_column_id(log_n_rows, n_active);
    let component = PoseidonComponent::new(
        &mut TraceLocationAllocator::default(),
        PoseidonEval {
            log_n_rows,
            lookup_elements,
            claimed_sum,
            is_first_id,
            is_active_id,
            n_active_rows: n_active,
        },
        claimed_sum,
    );
    info!("Poseidon component info:\n{}", component);
    let proof = prove(&[&component], channel, commitment_scheme).unwrap();

    (component, proof)
}

#[cfg(test)]
mod tests {
    use std::{array, env};

    use itertools::Itertools;
    use stwo::core::air::Component;
    use stwo::core::channel::Blake2sChannel;
    use stwo::core::fields::m31::{BaseField, M31};
    use stwo::core::fri::FriConfig;
    use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use stwo::core::verifier::verify;
    use stwo_constraint_framework::assert_constraints_on_polys;

    use crate::poseidon_uaciasv2::{
        apply_internal_round_matrix, apply_m4, eval_poseidon_sponge_constraints,
        gen_interaction_trace, gen_is_active_column, gen_is_first_column, gen_trace,
        is_active_column_id, is_first_column_id, prove_poseidon, PoseidonElements, RATE,
    };

    #[cfg(all(target_family = "wasm", not(target_os = "wasi")))]
    #[wasm_bindgen_test::wasm_bindgen_test]
    fn test_poseidon_prove_wasm() {
        const LOG_N_ROWS: u32 = 10;
        const N_ROWS: usize = 1 << LOG_N_ROWS;
        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(5, 1, 64),
        };

        // Generate some test messages (all rows active)
        let messages: Vec<[BaseField; RATE]> = (0..N_ROWS)
            .map(|i| std::array::from_fn(|j| BaseField::from_u32_unchecked((i * RATE + j) as u32)))
            .collect();

        // Prove.
        prove_poseidon(LOG_N_ROWS, N_ROWS, messages, config);
    }

    #[test]
    fn test_apply_m4() {
        let m4 = ndarray::arr2(&[
            [5, 7, 1, 3].map(M31),
            [4, 6, 1, 1].map(M31),
            [1, 3, 5, 7].map(M31),
            [1, 1, 4, 6].map(M31),
        ]);
        let state = [0, 1, 2, 3].map(M31);
        let expected_dot = m4.dot(&ndarray::arr2(&[state]).t());
        let expected_dot: [_; 4] = expected_dot.into_raw_vec_and_offset().0.try_into().unwrap();

        let actual_dot = apply_m4(state);

        assert_eq!(expected_dot, actual_dot);
    }

    #[test]
    fn test_apply_internal() {
        const W: usize = 16;
        let mut state = array::from_fn(|i| M31((i * 3 + 187) as u32));
        let mut internal_matrix = ndarray::arr2(&[[M31(1); W]; W]);
        for (i, elem) in internal_matrix.diag_mut().iter_mut().enumerate() {
            *elem += M31((1 << (i + 1)) as u32);
        }
        let expected_state = internal_matrix.dot(&ndarray::arr2(&[state]).t());
        let expected_state: [_; W] = expected_state
            .into_raw_vec_and_offset()
            .0
            .try_into()
            .unwrap();

        apply_internal_round_matrix(&mut state);

        assert_eq!(state, expected_state);
    }

    #[test]
    fn test_poseidon_constraints() {
        const LOG_N_ROWS: u32 = 8;
        const N_ROWS: usize = 1 << LOG_N_ROWS; // 256

        // Generate test messages (all rows active)
        let messages: Vec<[BaseField; RATE]> = (0..N_ROWS)
            .map(|i| std::array::from_fn(|j| BaseField::from_u32_unchecked((i * RATE + j) as u32)))
            .collect();

        // Trace.
        let is_first_col = gen_is_first_column(LOG_N_ROWS);
        let is_first_id = is_first_column_id(LOG_N_ROWS);
        let is_active_col = gen_is_active_column(LOG_N_ROWS, N_ROWS); // All rows active
        let is_active_id = is_active_column_id(LOG_N_ROWS, N_ROWS);
        let trace_preprocessed = vec![is_first_col, is_active_col];
        let (trace0, interaction_data) = gen_trace(LOG_N_ROWS, N_ROWS, messages);
        let lookup_elements = PoseidonElements::dummy();
        let (trace1, claimed_sum) =
            gen_interaction_trace(LOG_N_ROWS, N_ROWS, interaction_data, &lookup_elements);

        let traces = TreeVec::new(vec![trace_preprocessed, trace0, trace1]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());
        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(LOG_N_ROWS),
            |mut eval| {
                eval_poseidon_sponge_constraints(
                    &mut eval,
                    &lookup_elements,
                    &is_first_id,
                    &is_active_id,
                );
            },
            claimed_sum,
        );
    }

    #[test_log::test]
    fn test_simd_poseidon_prove() {
        // Note: To see time measurement, run test with
        //   RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info RUST_BACKTRACE=1 RUSTFLAGS="
        //   -C target-cpu=native -C target-feature=+avx512f -C opt-level=3" cargo test
        //   test_simd_poseidon_prove -- --nocapture

        // Get from environment variable:
        let log_n_rows = env::var("LOG_N_ROWS")
            .unwrap_or_else(|_| "10".to_string())
            .parse::<u32>()
            .unwrap();
        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(5, 1, 64),
        };

        let n_rows = 1 << log_n_rows;

        // Generate test messages (all rows active)
        let messages: Vec<[BaseField; RATE]> = (0..n_rows)
            .map(|i| std::array::from_fn(|j| BaseField::from_u32_unchecked((i * RATE + j) as u32)))
            .collect();

        // Prove.
        let (component, proof) = prove_poseidon(log_n_rows, n_rows, messages, config);

        // Verify.
        // TODO: Create Air instance independently.
        let channel = &mut Blake2sChannel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(proof.config);

        // Decommit.
        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = component.trace_log_degree_bounds();

        // Preprocessed columns.
        commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
        // Trace columns.
        commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);
        // Draw lookup element.
        let lookup_elements = PoseidonElements::draw(channel);
        assert_eq!(lookup_elements, component.lookup_elements);
        // Interaction columns.
        commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

        verify(&[&component], channel, commitment_scheme, proof).unwrap();
    }

    #[cfg(feature = "tracing")]
    #[test]
    fn trace_simd_poseidon_prove() {
        use stwo::tracing::SpanAccumulator;
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::Registry;

        let collector = SpanAccumulator::default();
        let layer = collector.clone();
        let subscriber = Registry::default().with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        let log_n_rows = env::var("LOG_N_ROWS")
            .unwrap_or_else(|_| "10".to_string())
            .parse::<u32>()
            .unwrap();
        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(5, 1, 64),
        };

        let n_rows = 1 << log_n_rows;

        // Generate test messages (all rows active)
        let messages: Vec<[BaseField; RATE]> = (0..n_rows)
            .map(|i| std::array::from_fn(|j| BaseField::from_u32_unchecked((i * RATE + j) as u32)))
            .collect();

        // Prove.
        let _ = prove_poseidon(log_n_rows, n_rows, messages, config);

        let csv = collector.export_csv();

        println!("{csv}");
    }

    /// Helper function to create messages from input vector for arbitrary length tests
    fn create_messages_from_input(input: Vec<u32>) -> Vec<[BaseField; RATE]> {
        input
            .chunks(8)
            .map(|chunk| {
                let mut msg = [BaseField::from_u32_unchecked(0); RATE];
                for (i, &val) in chunk.iter().enumerate() {
                    msg[i] = BaseField::from_u32_unchecked(val);
                }
                msg
            })
            .collect()
    }

    /// Test with full prove + verify flow for specific input size
    fn test_arbitrary_length_prove_verify(input_size: usize) {
        let log_n_rows = 8; // 256 rows
        let n_rows = 1 << log_n_rows; // 256
        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(5, 1, 64),
        };

        // Generate input
        let input: Vec<u32> = (0..input_size as u32).collect();
        let messages = create_messages_from_input(input);

        // Prove (all rows are active since we fill to capacity)
        let (component, proof) = prove_poseidon(log_n_rows, n_rows, messages, config);

        // Verify
        let channel = &mut Blake2sChannel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(proof.config);

        // Retrieve the expected column sizes in each commitment interaction
        let sizes = component.trace_log_degree_bounds();

        // Preprocessed columns
        commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
        // Trace columns
        commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);
        // Draw lookup element
        let lookup_elements = PoseidonElements::draw(channel);
        assert_eq!(lookup_elements, component.lookup_elements);
        // Interaction columns
        commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

        // Final verification
        verify(&[&component], channel, commitment_scheme, proof).unwrap();
    }

    #[test]
    fn test_arbitrary_length_8_elements() {
        // Single full message (8 elements)
        test_arbitrary_length_prove_verify(8);
    }

    #[test]
    fn test_arbitrary_length_10_elements() {
        // Partial message (10 elements = 1 full + 1 partial)
        test_arbitrary_length_prove_verify(10);
    }

    #[test]
    fn test_arbitrary_length_16_elements() {
        // Two full messages
        test_arbitrary_length_prove_verify(16);
    }

    #[test]
    fn test_arbitrary_length_24_elements() {
        // Three full messages
        test_arbitrary_length_prove_verify(24);
    }

    #[test]
    fn test_arbitrary_length_32_elements() {
        // Four full messages
        test_arbitrary_length_prove_verify(32);
    }

    #[test]
    fn test_arbitrary_length_40_elements() {
        // Five full messages
        test_arbitrary_length_prove_verify(40);
    }

    #[test]
    fn test_arbitrary_length_48_elements() {
        // Six full messages
        test_arbitrary_length_prove_verify(48);
    }

    #[test]
    fn test_arbitrary_length_64_elements() {
        // Eight full messages
        test_arbitrary_length_prove_verify(64);
    }

    #[test]
    fn test_arbitrary_length_100_elements() {
        // Many messages (13 chunks)
        test_arbitrary_length_prove_verify(100);
    }

    #[test]
    fn test_arbitrary_length_1_element() {
        // Minimal input (single element)
        test_arbitrary_length_prove_verify(1);
    }

    #[test]
    fn test_arbitrary_length_empty() {
        // Edge case: empty input (all padding)
        test_arbitrary_length_prove_verify(0);
    }

    #[test]
    fn test_arbitrary_length_256_elements() {
        test_arbitrary_length_prove_verify(256);
    }

    /// Test with padding rows: only 16 active messages out of 256 total rows
    /// This demonstrates the efficiency gain from is_active column
    /// Note: N_ACTIVE must be multiple of 16 (SIMD lane size) for LogUp consistency
    #[test]
    fn test_with_padding_rows() {
        const LOG_N_ROWS: u32 = 8; // 256 total rows
        const N_ACTIVE: usize = 16; // Only 16 active messages (must be multiple of 16)
        const N_ROWS: usize = 1 << LOG_N_ROWS;

        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(5, 1, 64),
        };

        // Generate only 10 real messages
        let messages: Vec<[BaseField; RATE]> = (0..N_ACTIVE)
            .map(|i| std::array::from_fn(|j| BaseField::from_u32_unchecked((i * RATE + j) as u32)))
            .collect();

        println!("\n=== Testing with Padding Rows ===");
        println!("Total rows: {}", N_ROWS);
        println!("Active rows: {}", N_ACTIVE);
        println!("Padding rows: {} (will be isolated)", N_ROWS - N_ACTIVE);

        // Prove - padding rows (10-255) will have:
        // - is_active = 0
        // - Chaining disabled (no dependency on row 9)
        // - LogUp disabled (multiplicity = 0)
        let (component, proof) = prove_poseidon(LOG_N_ROWS, N_ACTIVE, messages, config);

        // Verify
        let channel = &mut Blake2sChannel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(proof.config);

        let sizes = component.trace_log_degree_bounds();

        // Preprocessed columns (is_first, is_active)
        commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
        // Trace columns
        commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);
        // Draw lookup element
        let lookup_elements = PoseidonElements::draw(channel);
        assert_eq!(lookup_elements, component.lookup_elements);
        // Interaction columns
        commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

        // Final verification - should pass with padding rows isolated!
        verify(&[&component], channel, commitment_scheme, proof).unwrap();
        println!(
            "✅ Proof verified successfully with {} padding rows!",
            N_ROWS - N_ACTIVE
        );
    }

    /// Test with constraints directly - verify padding isolation
    /// Note: N_ACTIVE must be multiple of 16 (SIMD lane size) for LogUp consistency
    #[test]
    fn test_padding_constraints() {
        const LOG_N_ROWS: u32 = 8;
        const N_ACTIVE: usize = 16; // Must be multiple of 16
        const N_ROWS: usize = 1 << LOG_N_ROWS;

        // Generate only 16 messages
        let messages: Vec<[BaseField; RATE]> = (0..N_ACTIVE)
            .map(|i| std::array::from_fn(|j| BaseField::from_u32_unchecked((i * RATE + j) as u32)))
            .collect();

        // Trace with padding
        let is_first_col = gen_is_first_column(LOG_N_ROWS);
        let is_first_id = is_first_column_id(LOG_N_ROWS);
        let is_active_col = gen_is_active_column(LOG_N_ROWS, N_ACTIVE); // Only first 10 active
        let is_active_id = is_active_column_id(LOG_N_ROWS, N_ACTIVE);
        let trace_preprocessed = vec![is_first_col, is_active_col];

        let (trace0, interaction_data) = gen_trace(LOG_N_ROWS, N_ACTIVE, messages);
        let lookup_elements = PoseidonElements::dummy();
        let (trace1, claimed_sum) =
            gen_interaction_trace(LOG_N_ROWS, N_ACTIVE, interaction_data, &lookup_elements);

        let traces = TreeVec::new(vec![trace_preprocessed, trace0, trace1]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        // This should pass because:
        // - Rows 0-9: active, full constraints
        // - Rows 10-255: padding, chaining/LogUp disabled
        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(LOG_N_ROWS),
            |mut eval| {
                eval_poseidon_sponge_constraints(
                    &mut eval,
                    &lookup_elements,
                    &is_first_id,
                    &is_active_id,
                );
            },
            claimed_sum,
        );
        println!(
            "✅ Constraints verified with {} active and {} padding rows!",
            N_ACTIVE,
            N_ROWS - N_ACTIVE
        );
    }

    #[test]
    fn test_poseidon_127_of_128() {
        let log_n_rows = 7; // 128 rows
        let n_real_messages = 127; // 127 active messages, 1 padding row
        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(5, 1, 64),
        };

        // Generate 127 messages
        let messages: Vec<[BaseField; RATE]> = (0..n_real_messages)
            .map(|i| std::array::from_fn(|j| BaseField::from_u32_unchecked((i * RATE + j) as u32)))
            .collect();

        println!(
            "Testing {} messages in {}-row table ({} padding row)",
            n_real_messages,
            1 << log_n_rows,
            (1 << log_n_rows) - n_real_messages
        );

        // Prove
        let (component, proof) = prove_poseidon(log_n_rows, n_real_messages, messages, config);

        // Verify
        let channel = &mut Blake2sChannel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(proof.config);

        let sizes = component.trace_log_degree_bounds();

        // Preprocessed columns (is_first, is_active)
        commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
        // Trace columns
        commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);
        // Draw lookup elements (must match what prover drew!)
        let lookup_elements = PoseidonElements::draw(channel);
        assert_eq!(lookup_elements, component.lookup_elements);
        // Interaction columns
        commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

        verify(&[&component], channel, commitment_scheme, proof).unwrap();
        println!(
            "✅ Test passed: 127 messages with 1 padding row (row 127 has initial_state = all zeros)"
        );
    }
}
