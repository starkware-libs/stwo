use crate::core::air::evaluation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentTrace};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::constraints::{coset_vanishing, pair_excluder, point_vanishing};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::QM31;
use crate::core::fields::ExtensionOf;
use crate::core::poly::circle::{CanonicCoset, CircleDomain};

pub struct FibonacciComponent {
    pub log_size: u32,
    pub claim: BaseField,
}

impl FibonacciComponent {
    pub fn new(log_size: u32, claim: BaseField) -> Self {
        Self { log_size, claim }
    }
    fn step_constraint_eval_quotient_by_mask<F: ExtensionOf<BaseField>>(
        &self,
        point: CirclePoint<F>,
        mask: &[F],
    ) -> F {
        let constraint_zero_domain = Coset::subgroup(self.log_size);
        let constraint_value = mask[0].square() + mask[1].square() - mask[2];
        let selector = pair_excluder(
            constraint_zero_domain
                .at(constraint_zero_domain.size() - 2)
                .into_ef(),
            constraint_zero_domain
                .at(constraint_zero_domain.size() - 1)
                .into_ef(),
            point,
        );
        let num = constraint_value * selector;
        let denom = coset_vanishing(constraint_zero_domain, point);
        num / denom
    }

    fn boundary_constraint_eval_quotient_by_mask<F: ExtensionOf<BaseField>>(
        &self,
        point: CirclePoint<F>,
        mask: &[F],
    ) -> F {
        // TODO(spapini): Boundary constraint on the first 1 as well.
        let constraint_zero_domain = Coset::subgroup(self.log_size);
        let num = mask[0] - self.claim;
        let denom = point_vanishing(
            constraint_zero_domain.at(constraint_zero_domain.size() - 1),
            point,
        );
        num / denom
    }
}

impl Component for FibonacciComponent {
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace,
        evaluation_accumulator: &mut DomainEvaluationAccumulator,
    ) {
        let poly = &trace.0[0];
        let trace_domain = CanonicCoset::new(self.log_size);
        let trace_eval_domain = trace_domain.evaluation_domain(self.log_size + 1);
        let trace_eval = poly.evaluate(trace_eval_domain);

        // Step constraint.
        let constraint_log_degree_bound = trace_domain.log_size() + 1;
        let [mut accum] = evaluation_accumulator.columns([(constraint_log_degree_bound, 1)]);
        let constraint_eval_domain =
            CircleDomain::constraint_evaluation_domain(constraint_log_degree_bound);
        for (off, point_coset, mul) in [
            (0, constraint_eval_domain.half_coset, 1isize),
            (
                constraint_eval_domain.half_coset.size(),
                constraint_eval_domain.half_coset.conjugate(),
                -1,
            ),
        ] {
            let eval = trace_eval.fetch_eval_on_coset(point_coset.shift(trace_domain.index_at(0)));
            for (i, point) in point_coset.iter().enumerate() {
                let mask = [eval[i], eval[i as isize + mul], eval[i as isize + 2 * mul]];
                assert_eq!(
                    mask[0],
                    poly.eval_at_point(point + trace_domain.at(0).into_ef())
                );
                assert_eq!(
                    mask[1],
                    poly.eval_at_point(point + trace_domain.at(1).into_ef())
                );
                assert_eq!(
                    mask[2],
                    poly.eval_at_point(point + trace_domain.at(2).into_ef())
                );
                let res = self.step_constraint_eval_quotient_by_mask(point, &mask);
                accum.accumulate(i + off, res);
            }
        }

        // Boundary constraint.
        let constraint_log_degree_bound = trace_domain.log_size();
        let [mut accum] = evaluation_accumulator.columns([(constraint_log_degree_bound, 1)]);
        let constraint_eval_domain =
            CircleDomain::constraint_evaluation_domain(constraint_log_degree_bound);
        for (off, point_coset) in [
            (0, constraint_eval_domain.half_coset),
            (
                constraint_eval_domain.half_coset.size(),
                constraint_eval_domain.half_coset.conjugate(),
            ),
        ] {
            let eval = trace_eval.fetch_eval_on_coset(point_coset.shift(trace_domain.index_at(0)));
            for (i, point) in point_coset.iter().enumerate() {
                let mask = [eval[i]];
                let res = self.boundary_constraint_eval_quotient_by_mask(point, &mask);
                accum.accumulate(i + off, res);
            }
        }
    }

    fn mask_values_at_point(&self, point: CirclePoint<QM31>, trace: &ComponentTrace) -> Vec<QM31> {
        let poly = &trace.0[0];
        let trace_domain = CanonicCoset::new(self.log_size);
        vec![
            poly.eval_at_point(point + trace_domain.at(0).into_ef()),
            poly.eval_at_point(point + trace_domain.at(1).into_ef()),
            poly.eval_at_point(point + trace_domain.at(2).into_ef()),
        ]
    }

    fn evaluate_quotients_by_mask(
        &self,
        point: CirclePoint<QM31>,
        mask: &[QM31],
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    ) {
        let res = self.step_constraint_eval_quotient_by_mask(point, mask);
        let constraint_log_degree_bound = self.log_size + 1;
        evaluation_accumulator.accumulate(constraint_log_degree_bound, res);
        let res = self.boundary_constraint_eval_quotient_by_mask(point, &mask[..1]);
        let constraint_log_degree_bound = self.log_size;
        evaluation_accumulator.accumulate(constraint_log_degree_bound, res);
    }
}
