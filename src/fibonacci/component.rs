use std::ops::Div;

use num_traits::{One, Zero};

use crate::core::air::evaluation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentTrace, Mask};
use crate::core::backend::CPUBackend;
use crate::core::circle::{CirclePoint, Coset};
use crate::core::constraints::{coset_vanishing, pair_vanishing};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{ExtensionOf, FieldExpOps};
use crate::core::poly::circle::{CanonicCoset, CircleDomain, CircleEvaluation};
use crate::core::utils::bit_reverse_index;
use crate::core::ColumnVec;

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
        let num = self.step_constraint_nominator(point, mask);
        let denom = self.step_constraint_denominator(point);
        num / denom
    }

    fn step_constraint_denominator<F: ExtensionOf<BaseField>>(&self, point: CirclePoint<F>) -> F {
        let constraint_zero_domain = Coset::subgroup(self.log_size);
        coset_vanishing(constraint_zero_domain, point)
    }

    fn step_constraint_nominator<F: ExtensionOf<BaseField>>(
        &self,
        point: CirclePoint<F>,
        mask: &[F; 3],
    ) -> F {
        let constraint_zero_domain = Coset::subgroup(self.log_size);
        let constraint_value = mask[0].square() + mask[1].square() - mask[2];
        let selector = pair_vanishing(
            constraint_zero_domain
                .at(constraint_zero_domain.size() - 2)
                .into_ef(),
            constraint_zero_domain
                .at(constraint_zero_domain.size() - 1)
                .into_ef(),
            point,
        );
        constraint_value * selector
    }

    /// Evaluates the boundary constraint quotient polynomial on a single point.
    fn boundary_constraint_eval_quotient_by_mask<F: ExtensionOf<BaseField>>(
        &self,
        point: CirclePoint<F>,
        mask: &[F; 1],
    ) -> F {
        let num = self.boundary_constraint_nominator(point, mask);
        let denom = self.boundary_constraint_denominator(point);
        num / denom
    }

    fn boundary_constraint_nominator<F: ExtensionOf<BaseField>>(
        &self,
        point: CirclePoint<F>,
        mask: &[F; 1],
    ) -> F {
        let constraint_zero_domain = Coset::subgroup(self.log_size);
        let p = constraint_zero_domain.at(constraint_zero_domain.size() - 1);

        // On (1,0), we should get 1.
        // On p, we should get self.claim.
        // 1 + y * (self.claim - 1) * p.y^-1
        // TODO(spapini): Cache the constant.
        let linear = F::one() + point.y * (self.claim - BaseField::one()) * p.y.inverse();

        mask[0] - linear
    }

    fn boundary_constraint_denominator<F: ExtensionOf<BaseField>>(
        &self,
        point: CirclePoint<F>,
    ) -> F {
        let constraint_zero_domain = Coset::subgroup(self.log_size);
        let p = constraint_zero_domain.at(constraint_zero_domain.size() - 1);
        pair_vanishing(p.into_ef(), CirclePoint::zero(), point)
    }
}

const CPU_CHUNK_LOG_SIZE: usize = 2;
const CPU_CHUNK_SIZE: usize = 1 << CPU_CHUNK_LOG_SIZE;

impl Component<CPUBackend> for FibonacciComponent {
    fn max_constraint_log_degree_bound(&self) -> u32 {
        // Step constraint is of degree 2.
        self.log_size + 1
    }

    fn trace_log_degree_bounds(&self) -> Vec<u32> {
        vec![self.log_size]
    }

    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, CPUBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CPUBackend>,
    ) {
        let poly = &trace.columns[0];
        let trace_domain = CanonicCoset::new(self.log_size);
        let trace_eval_domain = trace_domain.evaluation_domain(self.log_size + 1);
        let trace_eval = poly.evaluate(trace_eval_domain).bit_reverse();

        // Step constraint.
        self.accumulate_step_constraint(trace_domain, &trace_eval, evaluation_accumulator);

        // Boundary constraint.
        self.accumulate_boundary_constraint(trace_domain, &trace_eval, evaluation_accumulator);
    }

    fn mask(&self) -> Mask {
        Mask(vec![vec![0, 1, 2]])
    }

    fn evaluate_quotients_by_mask(
        &self,
        point: CirclePoint<SecureField>,
        mask: &ColumnVec<Vec<SecureField>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    ) {
        let res =
            self.step_constraint_eval_quotient_by_mask(point, &mask[0][..].try_into().unwrap());
        let constraint_log_degree_bound = self.log_size + 1;
        evaluation_accumulator.accumulate(constraint_log_degree_bound, res);
        let res = self
            .boundary_constraint_eval_quotient_by_mask(point, &mask[0][..1].try_into().unwrap());
        let constraint_log_degree_bound = self.log_size;
        evaluation_accumulator.accumulate(constraint_log_degree_bound, res);
    }
}

impl FibonacciComponent
where
    FibonacciComponent: Component<CPUBackend>,
{
    fn accumulate_step_constraint(
        &self,
        trace_domain: CanonicCoset,
        trace_eval: &CircleEvaluation<CPUBackend, BaseField>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CPUBackend>,
    ) {
        let constraint_log_degree_bound = trace_domain.log_size() + 1;
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
            let mul = trace_domain.step_size().div(point_coset.step_size);

            // if the job is smaller than the chunk size, fallback to unoptimized version.
            if point_coset.size() < CPU_CHUNK_SIZE {
                for (i, point) in point_coset.iter().enumerate() {
                    let mask = [eval[i], eval[i as isize + mul], eval[i as isize + 2 * mul]];
                    let res = self.step_constraint_eval_quotient_by_mask(point, &mask);
                    accum.accumulate(bit_reverse_index(i + off, constraint_log_degree_bound), res);
                }
            } else {
                for chunk in point_coset
                    .iter()
                    .enumerate()
                    .array_chunks::<CPU_CHUNK_SIZE>()
                {
                    // Collect denominators and inverse.
                    let mut buff1: [BaseField; CPU_CHUNK_SIZE] =
                        std::array::from_fn(|i| self.step_constraint_denominator(chunk[i].1));
                    let mut inverses_buff = [BaseField::zero(); CPU_CHUNK_SIZE];
                    BaseField::batch_inverse(&buff1, &mut inverses_buff);
                    // Collect nominators.
                    buff1 = std::array::from_fn(|i| {
                        self.step_constraint_nominator(
                            chunk[i].1,
                            &[
                                eval[chunk[i].0],
                                eval[chunk[i].0 as isize + mul],
                                eval[chunk[i].0 as isize + 2 * mul],
                            ],
                        )
                    });

                    for (i, (num, inv_denom)) in buff1.iter().zip(inverses_buff.iter()).enumerate()
                    {
                        accum.accumulate(
                            bit_reverse_index(chunk[i].0 + off, constraint_log_degree_bound),
                            *num * *inv_denom,
                        );
                    }
                }
            }
        }
    }

    fn accumulate_boundary_constraint(
        &self,
        trace_domain: CanonicCoset,
        trace_eval: &CircleEvaluation<CPUBackend, BaseField>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CPUBackend>,
    ) {
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
            // if the job is smaller than the chunk size, fallback to unoptimized version.
            // TODO(Ohad): turn the entire thing to a generic function and call f::<1>() in that
            // case.
            if point_coset.size() < CPU_CHUNK_SIZE {
                for (i, point) in point_coset.iter().enumerate() {
                    let mask = [eval[i]];
                    let res = self.boundary_constraint_eval_quotient_by_mask(point, &mask);
                    accum.accumulate(bit_reverse_index(i + off, constraint_log_degree_bound), res);
                }
            } else {
                for chunk in point_coset
                    .iter()
                    .enumerate()
                    .array_chunks::<CPU_CHUNK_SIZE>()
                {
                    // Collect denominators and inverse.
                    let mut buff1: [BaseField; CPU_CHUNK_SIZE] = std::array::from_fn(|i| {
                        Self::boundary_constraint_denominator(self, chunk[i].1)
                    });
                    let mut buff2 = [BaseField::zero(); CPU_CHUNK_SIZE];
                    BaseField::batch_inverse(&buff1, &mut buff2);
                    // Collect nominators.
                    buff1 = std::array::from_fn(|i| {
                        Self::boundary_constraint_nominator(self, chunk[i].1, &[eval[chunk[i].0]])
                    });

                    for (i, (num, inv_denom)) in buff1.iter().zip(buff2.iter()).enumerate() {
                        accum.accumulate(
                            bit_reverse_index(chunk[i].0 + off, constraint_log_degree_bound),
                            *num * *inv_denom,
                        );
                    }
                }
            }
        }
    }
}
