use itertools::Itertools;
use num_traits::{One, Zero};
use tracing::{span, Level};

use super::component::{WideFibAir, WideFibComponent};
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::mask::fixed_mask_points;
use crate::core::air::{Air, Component, ComponentTrace};
use crate::core::backend::avx512::qm31::PackedSecureField;
use crate::core::backend::avx512::{AVX512Backend, BaseFieldVec, PackedBaseField, VECS_LOG_SIZE};
use crate::core::backend::{Col, Column, ColumnOps};
use crate::core::circle::CirclePoint;
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{FieldExpOps, FieldOps};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

const N_COLS: usize = 1 << 8;

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
        let mut b = PackedBaseField::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked((vec_index * 16 + i) as u32)
        }));
        trace[0].data[vec_index] = a;
        trace[1].data[vec_index] = b;
        trace.iter_mut().skip(2).for_each(|col| {
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
    fn n_constraints(&self) -> usize {
        N_COLS - 1
    }

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
        let span = span!(Level::INFO, "Constraint eval extension").entered();
        assert_eq!(trace.columns.len(), N_COLS);
        // TODO(spapini): Steal evaluation from commitment.
        let eval_domain = CanonicCoset::new(self.log_size + 1).circle_domain();
        let trace_eval = trace
            .columns
            .iter()
            .map(|poly| poly.evaluate(eval_domain))
            .collect_vec();

        // Denoms.
        // TODO(spapini): Make this prettier.
        let zero_domain = CanonicCoset::new(self.log_size).coset;
        let mut denoms =
            BaseFieldVec::from_iter(eval_domain.iter().map(|p| coset_vanishing(zero_domain, p)));
        <AVX512Backend as ColumnOps<BaseField>>::bit_reverse_column(&mut denoms);
        let mut denom_inverses = BaseFieldVec::zeros(denoms.len());
        <AVX512Backend as FieldOps<BaseField>>::batch_inverse(&denoms, &mut denom_inverses);

        span.exit();

        let _span = span!(Level::INFO, "Constraint pointwise eval").entered();

        let constraint_log_degree_bound = self.log_size + 1;
        let [accum] = evaluation_accumulator.columns([(constraint_log_degree_bound, N_COLS - 1)]);

        for vec_row in 0..(1 << (eval_domain.log_size() - VECS_LOG_SIZE as u32)) {
            // Numerator.
            let a = trace_eval[0].data[vec_row];
            let mut row_res =
                PackedSecureField::from_packed_m31s([
                    a - PackedBaseField::one(),
                    PackedBaseField::zero(),
                    PackedBaseField::zero(),
                    PackedBaseField::zero(),
                ]) * PackedSecureField::broadcast(accum.random_coeff_powers[N_COLS - 2]);

            let mut a_sq = a.square();
            let mut b_sq = trace_eval[1].data[vec_row].square();
            #[allow(clippy::needless_range_loop)]
            for i in 0..(N_COLS - 2) {
                unsafe {
                    let c = *trace_eval.get_unchecked(i + 2).data.get_unchecked(vec_row);
                    row_res +=
                        PackedSecureField::broadcast(accum.random_coeff_powers[N_COLS - 3 - i])
                            * (a_sq + b_sq - c);
                    (a_sq, b_sq) = (b_sq, c.square());
                }
            }

            accum.col.set_packed(
                vec_row,
                accum.col.packed_at(vec_row) + row_res * denom_inverses.data[vec_row],
            )
        }
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> ColumnVec<Vec<CirclePoint<SecureField>>> {
        fixed_mask_points(&vec![vec![0_usize]; 256], point)
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &ColumnVec<Vec<SecureField>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    ) {
        let zero_domain = CanonicCoset::new(self.log_size).coset;
        let denominator = coset_vanishing(zero_domain, point);
        evaluation_accumulator.accumulate((mask[0][0] - SecureField::one()) / denominator);
        for i in 0..(N_COLS - 2) {
            let numerator = mask[i][0].square() + mask[i + 1][0].square() - mask[i + 2][0];
            evaluation_accumulator.accumulate(numerator / denominator);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use crate::commitment_scheme::blake2_hash::Blake2sHasher;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::IntoSlice;
    use crate::core::prover::{prove, verify};
    use crate::examples::wide_fibonacci::avx::{gen_trace, WideFibAir};
    use crate::examples::wide_fibonacci::component::WideFibComponent;

    #[test_log::test]
    fn test_avx_wide_fib_prove() {
        // Note: To see time measurement, run test with
        //   RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info RUST_BACKTRACE=1 RUSTFLAGS="-Awarnings
        //   -C target-cpu=native -C target-feature=+avx512f -C opt-level=2" cargo test
        //   test_avx_wide_fib_prove -- --nocapture

        // Note: 17 means 128MB of trace.
        const LOG_SIZE: u32 = 17;
        let component = WideFibComponent { log_size: LOG_SIZE };
        let air = WideFibAir { component };
        let trace = gen_trace(LOG_SIZE as usize);
        let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
        let proof = prove(&air, channel, trace).unwrap();

        let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
        verify(proof, &air, channel).unwrap();
    }
}
