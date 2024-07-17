use crate::constraint_framework::EvalAtRow;
use crate::core::fields::qm31::SecureField;

/// Mask for an inclusive prefix sum.
#[derive(Debug, Clone, Copy)]
pub struct PrefixSumMask<E: EvalAtRow> {
    pub curr: E::EF,
    pub prev: E::EF,
}

impl<E: EvalAtRow> PrefixSumMask<E> {
    pub fn draw<const TRACE: usize>(eval: &mut E) -> Self {
        let [curr, prev] = eval.next_extension_interaction_mask(TRACE, [0, -1]);
        Self { curr, prev }
    }

    pub fn eval(self, eval: &mut E, row_diff: E::EF, cumulative_sum_shift: SecureField) {
        eval.add_constraint(self.curr - self.prev - row_diff + cumulative_sum_shift);
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use itertools::{chain, Itertools};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use test_log::test;

    use crate::constraint_framework::{assert_constraints, EvalAtRow};
    use crate::core::backend::simd::prefix_sum::inclusive_prefix_sum;
    use crate::core::backend::simd::qm31::PackedSecureField;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::secure_column::SecureColumnByCoords;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use crate::core::poly::BitReversedOrder;
    use crate::core::utils::{bit_reverse, coset_order_to_circle_domain_order};
    use crate::examples::xor::prefix_sum_constraints::PrefixSumMask;

    const SUM_TRACE: usize = 0;

    #[test]
    fn inclusive_prefix_sum_constraints_with_log_size_5() {
        const LOG_SIZE: u32 = 5;
        let mut rng = SmallRng::seed_from_u64(0);
        let vals = (0..1 << LOG_SIZE).map(|_| rng.gen()).collect_vec();
        let cumulative_sum = vals.iter().sum::<SecureField>();
        let cumulative_sum_shift = cumulative_sum / BaseField::from(vals.len());
        let trace = TreeVec::new(vec![gen_trace(vals)]);
        let trace_polys = trace.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(LOG_SIZE);

        assert_constraints(&trace_polys, trace_domain, |mut eval| {
            let [row_diff] = eval.next_extension_interaction_mask(SUM_TRACE, [0]);
            let mask = PrefixSumMask::draw::<SUM_TRACE>(&mut eval);
            mask.eval(&mut eval, row_diff, cumulative_sum_shift)
        });
    }

    /// Generates a trace.
    ///
    /// Trace structure:
    ///
    /// ```text
    /// ---------------------------------------------------------
    /// |           Values          |     Values prefix sum     |
    /// ---------------------------------------------------------
    /// |  c0  |  c1  |  c2  |  c3  |  c0  |  c1  |  c2  |  c3  |
    /// ---------------------------------------------------------
    /// ```
    fn gen_trace(
        vals: Vec<SecureField>,
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        assert!(vals.len().is_power_of_two());

        let vals_circle_domain_order = coset_order_to_circle_domain_order(&vals);
        let mut vals_bit_rev_circle_domain_order = vals_circle_domain_order;
        bit_reverse(&mut vals_bit_rev_circle_domain_order);
        let vals_secure_col: SecureColumnByCoords<SimdBackend> =
            vals_bit_rev_circle_domain_order.into_iter().collect();
        let vals_cols = vals_secure_col.columns;

        let cumulative_sum = vals.iter().sum::<SecureField>();
        let cumulative_sum_shift = cumulative_sum / BaseField::from(vals.len());
        let packed_cumulative_sum_shift = PackedSecureField::broadcast(cumulative_sum_shift);
        let packed_shifts = packed_cumulative_sum_shift.into_packed_m31s();
        let mut shifted_cols = vals_cols.clone();
        zip(&mut shifted_cols, packed_shifts)
            .for_each(|(col, packed_shift)| col.data.iter_mut().for_each(|v| *v -= packed_shift));
        let shifted_prefix_sum_cols = shifted_cols.map(inclusive_prefix_sum);

        let log_size = vals.len().ilog2();
        let trace_domain = CanonicCoset::new(log_size).circle_domain();

        chain![vals_cols, shifted_prefix_sum_cols]
            .map(|c| CircleEvaluation::new(trace_domain, c))
            .collect()
    }
}
