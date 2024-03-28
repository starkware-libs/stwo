use itertools::Itertools;
use num_traits::One;
use tracing::{span, Level};

use super::structs::WideFibComponent;
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Air, Component, ComponentTrace, Mask};
use crate::core::backend::avx512::qm31::PackedQM31;
use crate::core::backend::avx512::{AVX512Backend, PackedBaseField, VECS_LOG_SIZE};
use crate::core::backend::{Col, Column};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::bit_reverse_index;
use crate::core::ColumnVec;

const N_COLS: usize = 1 << 8;

pub struct WideFibAir {
    component: WideFibComponent,
}
impl Air<AVX512Backend> for WideFibAir {
    fn components(&self) -> Vec<&dyn Component<AVX512Backend>> {
        vec![&self.component]
    }
}

pub fn gen_trace(
    log_size: usize,
) -> ColumnVec<CircleEvaluation<AVX512Backend, BaseField, BitReversedOrder>> {
    let _span = span!(Level::INFO, "Trace generation").entered();
    assert!(log_size >= VECS_LOG_SIZE);
    let mut trace = (0..N_COLS)
        .map(|_| Col::<AVX512Backend, BaseField>::zeros(1 << log_size))
        .collect_vec();
    for vec_index in 0..(1 << (log_size - VECS_LOG_SIZE)) {
        let mut a = PackedBaseField::one();
        let mut b = PackedBaseField::one();
        trace[0].data[vec_index] = a;
        trace[1].data[vec_index] = b;
        trace.iter_mut().take(log_size).skip(2).for_each(|col| {
            (a, b) = (b, a.square() + b.square());
            col.data[vec_index] = b;
        });
    }
    let domain = CanonicCoset::new(log_size as u32).circle_domain();
    trace
        .into_iter()
        .map(|eval| CircleEvaluation::<AVX512Backend, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}

impl Component<AVX512Backend> for WideFibComponent {
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    fn trace_log_degree_bounds(&self) -> Vec<u32> {
        vec![self.log_size; N_COLS]
    }

    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, AVX512Backend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<AVX512Backend>,
    ) {
        assert_eq!(trace.polys.len(), N_COLS);
        // TODO(spapini): Steal evaluation from commitment.
        let eval_domain = CanonicCoset::new(self.log_size + 1).circle_domain();
        let trace_eval = &trace.evals;

        let _span = span!(Level::INFO, "Constraint eval evaluation").entered();
        let random_coeff = PackedQM31::broadcast(evaluation_accumulator.random_coeff);
        let column_coeffs = (0..N_COLS)
            .scan(PackedQM31::one(), |state, _| {
                let res = *state;
                *state *= random_coeff;
                Some(res)
            })
            .collect_vec();

        let constraint_log_degree_bound = self.log_size + 1;
        let [accum] = evaluation_accumulator.columns([(constraint_log_degree_bound, N_COLS - 2)]);

        for vec_row in 0..(1 << (eval_domain.log_size() - VECS_LOG_SIZE as u32)) {
            // Numerator.
            let mut row_res = PackedQM31::zero();
            let mut a = trace_eval[0].data[vec_row];
            let mut b = trace_eval[1].data[vec_row];
            #[allow(clippy::needless_range_loop)]
            for i in 0..(N_COLS - 2) {
                unsafe {
                    let c = *trace_eval.get_unchecked(i + 2).data.get_unchecked(vec_row);
                    row_res = row_res + column_coeffs[i] * (a.square() + b.square() - c);
                    (a, b) = (b, c);
                }
            }

            // Denominator.
            // TODO(spapini): Optimized this, for the small number of columns case.
            let points = std::array::from_fn(|i| {
                eval_domain.at(bit_reverse_index(
                    (vec_row << VECS_LOG_SIZE) + i,
                    eval_domain.log_size(),
                ) + 1)
            });
            let mut shifted_xs = PackedBaseField::from_array(points.map(|p| p.x));
            for _ in 1..self.log_size {
                shifted_xs = shifted_xs.square() - PackedBaseField::one();
            }

            accum.col.set_packed(
                vec_row,
                accum.col.packed_at(vec_row) * PackedQM31::broadcast(accum.random_coeff_pow)
                    + row_res,
            )
        }
    }

    fn mask(&self) -> Mask {
        Mask(vec![vec![0]; N_COLS])
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &ColumnVec<Vec<SecureField>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    ) {
        let constraint_zero_domain = Coset::subgroup(self.log_size);
        let constraint_log_degree_bound = self.log_size + 1;
        for i in 0..(N_COLS - 2) {
            let numerator = mask[i][0].square() + mask[i + 1][0].square() - mask[i + 2][0];
            let denominator = coset_vanishing(constraint_zero_domain, point);
            evaluation_accumulator.accumulate(constraint_log_degree_bound, numerator / denominator);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use rayon::prelude::*;

    use crate::commitment_scheme::blake2_hash::Blake2sHasher;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::IntoSlice;
    use crate::core::prover::prove;
    use crate::examples::wide_fibonacci::avx::{gen_trace, WideFibAir};
    use crate::examples::wide_fibonacci::structs::WideFibComponent;

    #[test_log::test]
    fn test_avx_wide_fib_prove() {
        // Note: To see time measurement, run test with
        //   RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info RUST_BACKTRACE=1 RUSTFLAGS="-Awarnings
        //   -C target-cpu=native -C target-feature=+avx512f -C opt-level=2" cargo test
        //   test_avx_wide_fib_prove -- --nocapture

        // TODO(spapini): Increase to 20, to get 1GB.
        const LOG_SIZE: u32 = 20;
        (0..1).into_par_iter().for_each(|_| {
            let component = WideFibComponent { log_size: LOG_SIZE };
            let air = WideFibAir { component };
            let trace = gen_trace(LOG_SIZE as usize);
            let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
            // TODO(spapini): Fix the constraints.
            prove(&air, channel, trace).unwrap_err();
        });
    }
}
