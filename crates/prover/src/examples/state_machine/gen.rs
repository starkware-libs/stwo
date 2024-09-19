use itertools::Itertools;
use num_traits::One;

use super::components::{State, N_STATE};
use crate::constraint_framework::logup::{LogupTraceGenerator, LookupElements};
use crate::core::backend::simd::column::BaseColumn;
use crate::core::backend::simd::m31::{PackedM31, LOG_N_LANES, N_LANES};
use crate::core::backend::simd::qm31::PackedQM31;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::QM31;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

// Given `initial state`, generate a trace that row `i` is the initial state plus `i` in the
// `inc_index` dimension.
// E.g. [x, y] -> [x, y + 1] -> [x, y + 2] -> [x, y + 1 << log_size].
pub fn gen_trace(
    log_size: u32,
    initial_state: State,
    inc_index: usize,
) -> ColumnVec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>> {
    let n_lanes = PackedM31::broadcast(M31::from_u32_unchecked(N_LANES as u32));
    let domain = CanonicCoset::new(log_size).circle_domain();

    // Prepare the state for the first packed row.
    let mut packed_state = initial_state.map(PackedM31::broadcast);
    let inc = PackedM31::from_array(std::array::from_fn(|i| M31::from_u32_unchecked((i) as u32)));
    packed_state[inc_index] += inc;

    let mut trace = (0..N_STATE)
        .map(|_| unsafe { BaseColumn::uninitialized(1 << log_size) })
        .collect_vec();
    for i in 0..(1 << (log_size - LOG_N_LANES)) {
        for j in 0..N_STATE {
            trace[j].data[i] = packed_state[j];
        }
        // Increment the state to the next packed row.
        packed_state[inc_index] += n_lanes;
    }
    trace
        .into_iter()
        .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}

pub fn gen_interaction_trace(
    log_size: u32,
    initial_state: [M31; N_STATE],
    inc_index: usize,
    lookup_elements: &LookupElements<N_STATE>,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    QM31,
) {
    let ones = PackedM31::broadcast(M31::one());
    let n_lanes_minus_one = PackedM31::broadcast(M31::from_u32_unchecked(N_LANES as u32)) - ones;

    // Prepare the state.
    let mut packed_state = initial_state.map(PackedM31::broadcast);
    let inc = PackedM31::from_array(std::array::from_fn(|i| M31::from_u32_unchecked((i) as u32)));
    packed_state[inc_index] += inc;

    let mut logup_gen = LogupTraceGenerator::new(log_size);
    let mut col_gen = logup_gen.new_col();

    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let q0: PackedQM31 = lookup_elements.combine(&packed_state);
        packed_state[inc_index] += ones;
        let q1: PackedQM31 = lookup_elements.combine(&packed_state);
        packed_state[inc_index] += n_lanes_minus_one;
        col_gen.write_frac(vec_row, q1 - q0, q0 * q1);
    }
    col_gen.finalize_col();

    let (trace, [total_sum]) = logup_gen.finalize([(1 << log_size) - 1]);
    (trace, total_sum)
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::One;

    use crate::core::backend::Column;
    use crate::core::channel::Blake2sChannel;
    use crate::core::fields::m31::M31;
    use crate::core::fields::qm31::QM31;
    use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
    use crate::core::fields::FieldExpOps;
    use crate::examples::state_machine::components::StateMachineElements;
    use crate::examples::state_machine::gen::{gen_interaction_trace, gen_trace};

    #[test]
    fn test_gen_trace() {
        let log_size = 8;
        let initial_state = [M31::from_u32_unchecked(17), M31::from_u32_unchecked(16)];
        let inc_index = 1;
        let row = 123;

        let trace = gen_trace(log_size, initial_state, inc_index);

        assert_eq!(trace.len(), 2);
        assert_eq!(trace[0].at(row), initial_state[0]);
        assert_eq!(
            trace[1].at(row),
            initial_state[1] + M31::from_u32_unchecked(row as u32)
        );
    }

    #[test]
    fn test_gen_interaction_trace() {
        let log_size = 8;
        let inc_index = 1;
        // Prepare state and next state.
        let state = [M31::from_u32_unchecked(17), M31::from_u32_unchecked(12)];
        let mut next_state = state;
        next_state[inc_index] += M31::one();

        let lookup_elements = StateMachineElements::dummy();
        let comb_state: QM31 = lookup_elements.combine(&state);
        let comb_next_state: QM31 = lookup_elements.combine(&next_state);

        let (trace, _) = gen_interaction_trace(log_size, state, inc_index, &lookup_elements);
        let first_log_up_row = QM31::from_m31_array(
            trace
                .iter()
                .map(|col| col.at(0))
                .collect_vec()
                .try_into()
                .unwrap(),
        );

        assert_eq!(trace.len(), SECURE_EXTENSION_DEGREE); // One quadradic extension column.
        assert_eq!(
            first_log_up_row,
            comb_state.inverse() - comb_next_state.inverse()
        );
    }

    #[test]
    fn test_state_machine_total_sum() {
        let log_n_rows = 8;
        let lookup_elements = StateMachineElements::draw(&mut Blake2sChannel::default());
        let inc_index = 0;

        let initial_state = [M31::from(123), M31::from(456)];
        let initial_state_comb: QM31 = lookup_elements.combine(&initial_state);

        let mut last_state = initial_state;
        last_state[inc_index] += M31::from_u32_unchecked(1 << log_n_rows);
        let last_state_comb: QM31 = lookup_elements.combine(&last_state);

        let (_, total_sum) =
            gen_interaction_trace(log_n_rows, initial_state, inc_index, &lookup_elements);

        // Assert total sum is `(1 / initial_state_comb) - (1 / last_state_comb)`.
        assert_eq!(
            total_sum * initial_state_comb * last_state_comb,
            last_state_comb - initial_state_comb
        );
    }
}
