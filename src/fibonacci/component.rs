use num_traits::One;

use crate::core::air::evaluation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentTrace};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::constraints::{coset_vanishing, pair_excluder};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{ExtensionOf, Field};
use crate::core::poly::circle::{CanonicCoset, CircleDomain};

pub struct FibonacciComponent {
    pub log_size: u32,
    pub claim: BaseField,
}

impl FibonacciComponent {
    pub fn new(log_size: u32, claim: BaseField) -> Self {
        Self { log_size, claim }
    }

    /// Evaluates the step constraint quotient polynomial on a single point.
    /// The step constraint is defined as:
    ///   mask[0]^2 + mask[1]^2 - mask[2]
    fn step_constraint_eval_quotient_by_mask<F: ExtensionOf<BaseField>>(
        &self,
        point: CirclePoint<F>,
        mask: &[F; 3],
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

    /// Evaluates the boundary constraint quotient polynomial on a single point.
    fn boundary_constraint_eval_quotient_by_mask<F: ExtensionOf<BaseField>>(
        &self,
        point: CirclePoint<F>,
        mask: &[F; 1],
    ) -> F {
        let constraint_zero_domain = Coset::subgroup(self.log_size);
        let p = constraint_zero_domain.at(constraint_zero_domain.size() - 1);
        // On (1,0), we should get 1.
        // On p, we should get self.claim.
        // 1 + y * (1 - self.claim) * p.y^-1
        // TODO(spapini): Cache the constant.
        let linear = F::one() + point.y * (self.claim - BaseField::one()) * p.y.inverse();

        let num = mask[0] - linear;
        let denom = pair_excluder(p.into_ef(), CirclePoint::zero(), point);
        num / denom
    }
}

impl Component for FibonacciComponent {
    fn max_constraint_log_degree_bound(&self) -> u32 {
        // Step constraint is of degree 2.
        self.log_size + 1
    }

    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator,
    ) {
        let poly = &trace.columns[0];
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

    fn mask_values_at_point(
        &self,
        point: CirclePoint<SecureField>,
        trace: &ComponentTrace<'_>,
    ) -> Vec<SecureField> {
        let poly = &trace.columns[0];
        let trace_domain = CanonicCoset::new(self.log_size);
        vec![
            poly.eval_at_point(point + trace_domain.at(0).into_ef()),
            poly.eval_at_point(point + trace_domain.at(1).into_ef()),
            poly.eval_at_point(point + trace_domain.at(2).into_ef()),
        ]
    }

    fn evaluate_quotients_by_mask(
        &self,
        point: CirclePoint<SecureField>,
        mask: &[SecureField],
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    ) {
        let res = self.step_constraint_eval_quotient_by_mask(point, mask.try_into().unwrap());
        let constraint_log_degree_bound = self.log_size + 1;
        evaluation_accumulator.accumulate(constraint_log_degree_bound, res);
        let res =
            self.boundary_constraint_eval_quotient_by_mask(point, &mask[..1].try_into().unwrap());
        let constraint_log_degree_bound = self.log_size;
        evaluation_accumulator.accumulate(constraint_log_degree_bound, res);
    }
}
