#![allow(dead_code)]

use num_traits::One;

use crate::constraint_framework::EvalAtRow;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;

/// Inclusive prefix sum constraint.
///
/// Note the prefix sum is index inclusive i.e. `prefix_sum([3, 4, 5, 1]) = [3, 7, 12, 13]`.
///
/// Checks:
/// - First row equals `claimed_row_diff`
/// - All rows (except first) equal `prev_sum + claimed_row_diff`
/// - Last row equals `claimed_sum`
pub fn prefix_sum_check<E: EvalAtRow>(
    eval: &mut E,
    row_diff: E::EF,
    final_sum: SecureField,
    at: &PrefixSumMaskAt<E>,
    is: &PrefixSumMaskIs<E>,
) {
    let is_not_first = E::F::from(BaseField::one()) - is.first;

    // TODO: Why won't this work?
    // `eval.add_constraint(at_curr - (at_prev * is_not_first + row_diff) +
    // (row_diff - final_sum) * is_last);`
    eval.add_constraint((at.curr - row_diff) * is.first);
    eval.add_constraint((at.curr - (at.prev + row_diff)) * is_not_first);
    eval.add_constraint((at.curr - final_sum) * is.last);
}

#[derive(Debug, Clone, Copy)]
pub struct PrefixSumMaskAt<E: EvalAtRow> {
    pub curr: E::EF,
    pub prev: E::EF,
}

impl<E: EvalAtRow> PrefixSumMaskAt<E> {
    pub fn draw<const TRACE: usize>(eval: &mut E) -> Self {
        let [curr, prev] = eval.next_extension_interaction_mask(TRACE, [0, -1]);
        Self { curr, prev }
    }
}

pub struct PrefixSumMaskIs<E: EvalAtRow> {
    pub first: E::F,
    pub last: E::F,
}

impl<E: EvalAtRow> PrefixSumMaskIs<E> {
    pub fn draw<const TRACE: usize>(eval: &mut E) -> Self {
        let [first, last] = eval.next_interaction_mask(TRACE, [0, 1]);
        Self { first, last }
    }
}

/// Returns the prefix sum.
///
/// The prefix sum is index inclusive i.e. `prefix_sum([3, 4, 5, 1]) = [3, 7, 12, 13]`.
pub fn prefix_sum<F: Field>(v: &[F]) -> Vec<F> {
    v.iter()
        .scan(F::zero(), |acc, &v| {
            *acc += v;
            Some(*acc)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use itertools::Itertools;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use test_log::test;

    use super::{prefix_sum, prefix_sum_check};
    use crate::constraint_framework::constant_cols::gen_is_first;
    use crate::constraint_framework::{assert_constraints, EvalAtRow};
    use crate::core::backend::simd::SimdBackend;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::Field;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use crate::core::poly::BitReversedOrder;
    use crate::core::utils::{
        bit_reverse, circle_domain_order_to_coset_order, coset_order_to_circle_domain_order,
    };
    use crate::examples::xor::prefix_sum::{PrefixSumMaskAt, PrefixSumMaskIs};

    const SUM_TRACE: usize = 0;
    const CONST_TRACE: usize = 1;

    #[test]
    fn test_prefix_sum_constraints_with_log_size_5() {
        const LOG_SIZE: u32 = 5;
        let mut rng = SmallRng::seed_from_u64(0);
        let vals = (0..1 << LOG_SIZE).map(|_| rng.gen()).collect_vec();
        let prefix_sum = prefix_sum(&vals);
        let claim = vals.iter().sum();
        assert_eq!(prefix_sum.last(), Some(&claim));
        let base_trace = gen_base_trace(vals, prefix_sum);
        let constants_trace = gen_constants_trace(LOG_SIZE);
        let traces = TreeVec::new(vec![base_trace, constants_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(LOG_SIZE);

        assert_constraints(&trace_polys, trace_domain, |mut eval| {
            let [row_diff] = eval.next_extension_interaction_mask(SUM_TRACE, [0]);
            let at_mask = PrefixSumMaskAt::draw::<SUM_TRACE>(&mut eval);
            let is_mask = PrefixSumMaskIs::draw::<CONST_TRACE>(&mut eval);
            prefix_sum_check(&mut eval, row_diff, claim, &at_mask, &is_mask);
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
        prefix_sum: Vec<SecureField>,
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        assert!(vals.len().is_power_of_two());
        assert_eq!(vals.len(), prefix_sum.len());

        let log_size = vals.len().ilog2();

        let vals_circle_domain_order = coset_order_to_circle_domain_order(&vals);
        let mut vals_bit_rev_circle_domain_order = vals_circle_domain_order;
        bit_reverse(&mut vals_bit_rev_circle_domain_order);

        let prefix_sum_circle_domain_order = coset_order_to_circle_domain_order(&prefix_sum);
        let mut prefix_sum_bit_rev_circle_domain_order = prefix_sum_circle_domain_order;
        bit_reverse(&mut prefix_sum_bit_rev_circle_domain_order);

        let mut vals_evals_col0 = Vec::new();
        let mut vals_evals_col1 = Vec::new();
        let mut vals_evals_col2 = Vec::new();
        let mut vals_evals_col3 = Vec::new();

        for v in vals_bit_rev_circle_domain_order {
            let [v0, v1, v2, v3] = v.to_m31_array();
            vals_evals_col0.push(v0);
            vals_evals_col1.push(v1);
            vals_evals_col2.push(v2);
            vals_evals_col3.push(v3);
        }

        let mut prefix_sum_col0 = Vec::new();
        let mut prefix_sum_col1 = Vec::new();
        let mut prefix_sum_col2 = Vec::new();
        let mut prefix_sum_col3 = Vec::new();

        for v in prefix_sum_bit_rev_circle_domain_order {
            let [v0, v1, v2, v3] = v.to_m31_array();
            prefix_sum_col0.push(v0);
            prefix_sum_col1.push(v1);
            prefix_sum_col2.push(v2);
            prefix_sum_col3.push(v3);
        }

        let trace_domain = CanonicCoset::new(log_size).circle_domain();

        vec![
            CircleEvaluation::new(trace_domain, vals_evals_col0.into_iter().collect()),
            CircleEvaluation::new(trace_domain, vals_evals_col1.into_iter().collect()),
            CircleEvaluation::new(trace_domain, vals_evals_col2.into_iter().collect()),
            CircleEvaluation::new(trace_domain, vals_evals_col3.into_iter().collect()),
            CircleEvaluation::new(trace_domain, prefix_sum_col0.into_iter().collect()),
            CircleEvaluation::new(trace_domain, prefix_sum_col1.into_iter().collect()),
            CircleEvaluation::new(trace_domain, prefix_sum_col2.into_iter().collect()),
            CircleEvaluation::new(trace_domain, prefix_sum_col3.into_iter().collect()),
        ]
    }

    #[test]
    fn prefix_sum_algo() {
        let log_size = 3;
        let vals = (1..=1 << log_size).map(BaseField::from).collect_vec();
        println!("{:?}", prefix_sum(&vals));
        println!("{:?}", prefix_sum_opt(&vals));
        println!("{:?}", prefix_sum_opt2(&vals));
    }

    //
    pub fn prefix_sum_opt<F: Field>(v: &[F]) -> Vec<F> {
        let n = v.len();
        assert!(n.is_power_of_two());
        let log_n = n.ilog2();

        let mut res = v.to_vec();

        // Upsweep
        for log_step in 1..=log_n as usize {
            let step = 1 << log_step;
            for i in (step - 1..n).step_by(step) {
                res[i] = res[i] + res[i - step / 2];
            }
        }

        let last = res.last().copied();

        if let Some(last) = res.last_mut() {
            *last = F::zero();
        }

        // Downsweep
        for log_step in (1..=log_n as usize).rev() {
            let step = 1 << log_step;

            for i in (step - 1..n).step_by(step) {
                let tmp = res[i - step / 2];
                res[i - step / 2] = res[i];
                res[i] += tmp;
            }
        }

        if let (Some(first), Some(last)) = (res.first_mut(), last) {
            *first = last;
        }

        res
    }

    pub fn prefix_sum_opt2<F: Field>(v: &[F]) -> Vec<F> {
        let n = v.len();
        assert!(n.is_power_of_two());
        let log_n = n.ilog2();

        let mut res = coset_order_to_circle_domain_order(v);
        bit_reverse(&mut res);

        for log_chunk_size in (0..log_n).rev() {
            let chunk_size = 1 << log_chunk_size;
            let (lhs_chunk, rhs_chunk) = res[n - chunk_size * 2..].split_at_mut(chunk_size);
            zip(lhs_chunk, rhs_chunk).for_each(|(lhs, rhs)| *rhs += *lhs);
        }

        let last = res.last().copied();

        if let Some(last) = res.last_mut() {
            *last = F::zero();
        }

        // Downsweep
        for log_chunk_size in 0..log_n {
            let chunk_size = 1 << log_chunk_size;
            let (lhs_chunk, rhs_chunk) = res[n - chunk_size * 2..].split_at_mut(chunk_size);
            zip(lhs_chunk, rhs_chunk).for_each(|(lhs, rhs)| {
                let tmp = *lhs;
                *lhs = *rhs;
                *rhs += tmp;
            });
        }

        if let (Some(first), Some(last)) = (res.first_mut(), last) {
            *first = last;
        }

        bit_reverse(&mut res);
        circle_domain_order_to_coset_order(&res)
    }

    /// Generates a single column trace. Column is `1` at the first trace point and `0` on all
    /// other trace points.
    fn gen_constants_trace(
        log_size: u32,
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        vec![gen_is_first(log_size)]
    }
}
