use itertools::{chain, Itertools};
use num_traits::{One, Zero};

use super::components::{State, STATE_SIZE};
use crate::constraint_framework::logup::{LogupTraceGenerator, LookupElements};
use crate::core::backend::simd::column::BaseColumn;
use crate::core::backend::simd::m31::{PackedM31, LOG_N_LANES};
use crate::core::backend::simd::qm31::PackedQM31;
use crate::core::backend::simd::SimdBackend;
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::QM31;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

// Generates the trace for the state machine transition.
// These are 2 * STATE_SIZE columns where each row contains [input_state, output_state] with the
// following:
//              output_state[i] = input_state[i] + 1 for i == coordinate; and
//              output_state[i] = input_state[i] otherwise.
pub fn gen_trace(
    log_size: u32,
    initial_state: State,
    coordinate: usize,
) -> ColumnVec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>> {
    let domain = CanonicCoset::new(log_size).circle_domain();

    // Initialize empty data for the resulted columns.
    let mut input_state_cols = (0..STATE_SIZE)
        .map(|_| vec![M31::zero(); 1 << log_size])
        .collect_vec();
    let mut output_state_cols = (0..STATE_SIZE)
        .map(|_| vec![M31::zero(); 1 << log_size])
        .collect_vec();

    // Fill columns with the state transitions.
    let mut curr_state = initial_state;
    for i in 0..1 << log_size {
        for j in 0..STATE_SIZE {
            input_state_cols[j][i] = curr_state[j];
            output_state_cols[j][i] = curr_state[j];
            if j == coordinate {
                output_state_cols[j][i] += M31::one();
            }
        }
        // Increment the state to the next state row.
        curr_state[coordinate] += M31::one();
    }

    // Collect and return the columns.
    chain![input_state_cols, output_state_cols]
        .map(|col| {
            CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(
                domain,
                BaseColumn::from_iter(col),
            )
        })
        .collect_vec()
}

// Returns the interaction trace columns for the state machine transition.
// This contain an ExtensionField columns, each stored as 4 BaseField columns.
pub fn gen_interaction_trace(
    log_size: u32,
    trace: &ColumnVec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    lookup_elements: &LookupElements<STATE_SIZE>,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    QM31,
) {
    let mut logup_gen = LogupTraceGenerator::new(log_size);
    let (input_cols, output_cols) = trace.split_at(STATE_SIZE);

    // Write interaction column for use input state.
    let mut input_lookup_col = logup_gen.new_col();
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let input_state: [PackedM31; STATE_SIZE] = input_cols
            .iter()
            .map(|col| col.data[vec_row])
            .collect_vec()
            .try_into()
            .unwrap();
        let input_denom: PackedQM31 = lookup_elements.combine(&input_state);
        input_lookup_col.write_frac(vec_row, PackedQM31::broadcast(QM31::one()), input_denom);
    }
    input_lookup_col.finalize_col();

    // Write interaction column for yield output state.
    let mut output_lookup_col = logup_gen.new_col();
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let output_state: [PackedM31; STATE_SIZE] = output_cols
            .iter()
            .map(|col| col.data[vec_row])
            .collect_vec()
            .try_into()
            .unwrap();
        let output_denom: PackedQM31 = lookup_elements.combine(&output_state);
        output_lookup_col.write_frac(vec_row, PackedQM31::broadcast(-QM31::one()), output_denom);
    }
    output_lookup_col.finalize_col();

    logup_gen.finalize_last()
}

#[cfg(test)]
mod tests {
    use itertools::zip_eq;
    use num_traits::One;

    use crate::core::backend::Column;
    use crate::core::fields::m31::M31;
    use crate::core::fields::qm31::QM31;
    use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
    use crate::core::fields::FieldExpOps;
    use crate::examples::state_machine::components::{StateMachineElements, STATE_SIZE};
    use crate::examples::state_machine::gen::{gen_interaction_trace, gen_trace};

    #[test]
    fn test_gen_trace() {
        let log_size = 8;
        let initial_state = [M31::from_u32_unchecked(17), M31::from_u32_unchecked(16)];
        let coordinate = 1;
        let row = 123;

        let trace = gen_trace(log_size, initial_state, coordinate);

        assert_eq!(trace.len(), 4);
        let (input_cols, output_cols) = trace.split_at(STATE_SIZE);
        zip_eq(input_cols, output_cols)
            .enumerate()
            .for_each(|(i, (input, output))| {
                if i == coordinate {
                    assert_eq!(input.at(row) + M31::one(), output.at(row));
                } else {
                    assert_eq!(input.at(row), output.at(row));
                }
            });
    }

    #[test]
    fn test_gen_interaction_trace() {
        let log_size = 8;
        let coordinate = 1;
        // Prepare the first and the last states.
        let first_state = [M31::from_u32_unchecked(17), M31::from_u32_unchecked(12)];
        let mut last_state = first_state;
        last_state[coordinate] += M31::from_u32_unchecked(1 << log_size);

        let trace = gen_trace(log_size, first_state, coordinate);
        let lookup_elements = StateMachineElements::dummy();
        let first_state_comb: QM31 = lookup_elements.combine(&first_state);
        let last_state_comb: QM31 = lookup_elements.combine(&last_state);

        let (interaction_trace, total_sum) =
            gen_interaction_trace(log_size, &trace, &lookup_elements);

        assert_eq!(interaction_trace.len(), SECURE_EXTENSION_DEGREE * 2); // Two extension column.
        assert_eq!(
            total_sum,
            first_state_comb.inverse() - last_state_comb.inverse()
        );
    }
}
