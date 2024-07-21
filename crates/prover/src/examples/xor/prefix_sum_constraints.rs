use crate::constraint_framework::EvalAtRow;
use crate::core::fields::qm31::SecureField;

/// Inclusive prefix sum constraint.
pub fn inclusive_prefix_sum_check<E: EvalAtRow>(
    eval: &mut E,
    row_diff: E::EF,
    final_sum: SecureField,
    is_first: E::F,
    at: &PrefixSumMask<E>,
) {
    let prev = at.prev - is_first * final_sum;
    eval.add_constraint(at.curr - prev - row_diff);
}

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
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use test_log::test;

    use super::inclusive_prefix_sum_check;
    use crate::constraint_framework::{assert_constraints, EvalAtRow};
    use crate::core::backend::simd::prefix_sum::inclusive_prefix_sum;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::{Col, Column};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::secure_column::SecureColumnByCoords;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use crate::core::poly::BitReversedOrder;
    use crate::core::utils::{bit_reverse, coset_order_to_circle_domain_order};
    use crate::examples::xor::prefix_sum_constraints::PrefixSumMask;

    const SUM_TRACE: usize = 0;
    const CONST_TRACE: usize = 1;

    #[test]
    fn inclusive_prefix_sum_constraints_with_log_size_5() {
        const LOG_SIZE: u32 = 5;
        let mut rng = SmallRng::seed_from_u64(0);
        let vals = (0..1 << LOG_SIZE).map(|_| rng.gen()).collect_vec();
        let final_sum = vals.iter().sum();
        let base_trace = gen_base_trace(vals);
        let constants_trace = gen_constants_trace(LOG_SIZE);
        let traces = TreeVec::new(vec![base_trace, constants_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(LOG_SIZE);

        assert_constraints(&trace_polys, trace_domain, |mut eval| {
            let [is_first] = eval.next_interaction_mask(CONST_TRACE, [0]);
            let [row_diff] = eval.next_extension_interaction_mask(SUM_TRACE, [0]);
            let at_mask = PrefixSumMask::draw::<SUM_TRACE>(&mut eval);
            inclusive_prefix_sum_check(&mut eval, row_diff, final_sum, is_first, &at_mask);
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
    fn gen_base_trace(
        vals: Vec<SecureField>,
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        assert!(vals.len().is_power_of_two());

        let vals_circle_domain_order = coset_order_to_circle_domain_order(&vals);
        let mut vals_bit_rev_circle_domain_order = vals_circle_domain_order;
        bit_reverse(&mut vals_bit_rev_circle_domain_order);
        let vals_secure_col: SecureColumnByCoords<SimdBackend> =
            vals_bit_rev_circle_domain_order.into_iter().collect();
        let [vals_col0, vals_col1, vals_col2, vals_col3] = vals_secure_col.columns;

        let prefix_sum_col0 = inclusive_prefix_sum(vals_col0.clone());
        let prefix_sum_col1 = inclusive_prefix_sum(vals_col1.clone());
        let prefix_sum_col2 = inclusive_prefix_sum(vals_col2.clone());
        let prefix_sum_col3 = inclusive_prefix_sum(vals_col3.clone());

        let log_size = vals.len().ilog2();
        let trace_domain = CanonicCoset::new(log_size).circle_domain();

        vec![
            CircleEvaluation::new(trace_domain, vals_col0),
            CircleEvaluation::new(trace_domain, vals_col1),
            CircleEvaluation::new(trace_domain, vals_col2),
            CircleEvaluation::new(trace_domain, vals_col3),
            CircleEvaluation::new(trace_domain, prefix_sum_col0),
            CircleEvaluation::new(trace_domain, prefix_sum_col1),
            CircleEvaluation::new(trace_domain, prefix_sum_col2),
            CircleEvaluation::new(trace_domain, prefix_sum_col3),
        ]
    }

    fn gen_constants_trace(
        log_size: u32,
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        let trace_domain = CanonicCoset::new(log_size).circle_domain();
        // Column is `1` at the first trace point and `0` on all other trace points.
        let mut is_first = Col::<SimdBackend, BaseField>::zeros(1 << log_size);
        is_first.as_mut_slice()[0] = BaseField::one();
        vec![CircleEvaluation::new(trace_domain, is_first)]
    }
}
