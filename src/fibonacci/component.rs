use std::ops::Div;

use itertools::Itertools;
use num_traits::One;

use crate::core::air::evaluation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentTrace, Mask};
use crate::core::backend::CPUBackend;
use crate::core::circle::{CirclePoint, Coset};
use crate::core::constraints::{coset_vanishing, pair_vanishing};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{ExtensionOf, FieldExpOps, FieldOps};
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::core::utils::bit_reverse_index;
use crate::core::ColumnVec;

const CPU_CHUNK_SIZE: usize = 1 << 8;

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
    fn step_constraint_eval_quotient_by_mask<F: ExtensionOf<BaseField>, const CHUNK_SIZE: usize>(
        &self,
        point: &[CirclePoint<F>; CHUNK_SIZE],
        mask: &[[F; 3]; CHUNK_SIZE],
    ) -> [F; CHUNK_SIZE] {
        let constraint_zero_domain = Coset::subgroup(self.log_size);
        let nums = point
            .iter()
            .zip(mask)
            .map(|(&point, mask)| {
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
            })
            .collect_vec();

        let denoms = point
            .iter()
            .map(|&point| coset_vanishing(constraint_zero_domain, point))
            .collect_vec();
        let mut inv_denoms = vec![F::zero(); CHUNK_SIZE];
        CPUBackend::batch_inverse(&denoms, &mut inv_denoms);

        nums.iter()
            .zip(inv_denoms)
            .map(|(&num, inv_denom)| num * inv_denom)
            .collect_vec()
            .try_into()
            .unwrap()
    }

    /// Evaluates the boundary constraint quotient polynomial on a single point.
    fn boundary_constraint_eval_quotient_by_mask<
        F: ExtensionOf<BaseField>,
        const CHUNK_SIZE: usize,
    >(
        &self,
        point: &[CirclePoint<F>; CHUNK_SIZE],
        mask: &[[F; 1]; CHUNK_SIZE],
    ) -> [F; CHUNK_SIZE] {
        let constraint_zero_domain = Coset::subgroup(self.log_size);
        let p = constraint_zero_domain.at(constraint_zero_domain.size() - 1);
        // On (1,0), we should get 1.
        // On p, we should get self.claim.
        // 1 + y * (self.claim - 1) * p.y^-1
        // TODO(spapini): Cache the constant.
        let nums = mask
            .iter()
            .zip(point)
            .map(|(&mask, point)| {
                let linear = F::one() + point.y * (self.claim - BaseField::one()) * p.y.inverse();
                mask[0] - linear
            })
            .collect_vec();
        let denoms = point
            .iter()
            .map(|&point| pair_vanishing(p.into_ef(), CirclePoint::zero(), point))
            .collect_vec();
        let mut inv_denoms = vec![F::zero(); CHUNK_SIZE];
        CPUBackend::batch_inverse(&denoms, &mut inv_denoms);

        nums.iter()
            .zip(inv_denoms)
            .map(|(&num, inv_denom)| num * inv_denom)
            .collect_vec()
            .try_into()
            .unwrap()
    }

    fn evaluate_constraint_quotients_on_domain_chunks<const CHUNK_SIZE: usize>(
        &self,
        trace: &ComponentTrace<'_, CPUBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CPUBackend>,
    ) {
        let poly = &trace.columns[0];
        let trace_domain = CanonicCoset::new(self.log_size);
        let trace_eval_domain = trace_domain.evaluation_domain(self.log_size + 1);
        let trace_eval = poly.evaluate(trace_eval_domain).bit_reverse();

        // Step constraint.
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
            debug_assert!(CHUNK_SIZE <= point_coset.size());
            for (i, chunk) in point_coset
                .iter()
                .chunks(CHUNK_SIZE)
                .into_iter()
                .enumerate()
            {
                let masks: [[BaseField; 3]; CHUNK_SIZE] = (0..CHUNK_SIZE)
                    .map(|j| {
                        [
                            eval[i * CHUNK_SIZE + j],
                            eval[(i * CHUNK_SIZE + j) as isize + mul],
                            eval[(i * CHUNK_SIZE + j) as isize + 2 * mul],
                        ]
                    })
                    .collect_vec()
                    .try_into()
                    .unwrap();
                let res = self.step_constraint_eval_quotient_by_mask(
                    &chunk.into_iter().collect_vec().try_into().unwrap(),
                    &masks,
                );
                for (j, r) in res.into_iter().enumerate() {
                    accum.accumulate(
                        bit_reverse_index(i * CHUNK_SIZE + j + off, constraint_log_degree_bound),
                        r,
                    );
                }
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
            debug_assert!(CHUNK_SIZE <= point_coset.size());
            let eval = trace_eval.fetch_eval_on_coset(point_coset.shift(trace_domain.index_at(0)));
            for (i, chunk) in point_coset
                .iter()
                .chunks(CHUNK_SIZE)
                .into_iter()
                .enumerate()
            {
                let masks: [[BaseField; 1]; CHUNK_SIZE] = (0..CHUNK_SIZE)
                    .map(|j| [eval[i * CHUNK_SIZE + j]])
                    .collect_vec()
                    .try_into()
                    .unwrap();
                let res = self.boundary_constraint_eval_quotient_by_mask(
                    &chunk.into_iter().collect_vec().try_into().unwrap(),
                    &masks,
                );
                for (j, r) in res.into_iter().enumerate() {
                    accum.accumulate(
                        bit_reverse_index(i * CHUNK_SIZE + j + off, constraint_log_degree_bound),
                        r,
                    );
                }
            }
        }
    }
}

impl Component<CPUBackend> for FibonacciComponent {
    fn max_constraint_log_degree_bound(&self) -> u32 {
        // Step constraint is of degree 2.
        self.log_size + 1
    }

    fn trace_log_degree_bounds(&self) -> Vec<u32> {
        vec![self.log_size]
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
        let res = self
            .step_constraint_eval_quotient_by_mask(&[point], &[mask[0][..].try_into().unwrap()]);
        let constraint_log_degree_bound = self.log_size + 1;
        evaluation_accumulator.accumulate(constraint_log_degree_bound, res[0]);
        let res = self.boundary_constraint_eval_quotient_by_mask(
            &[point],
            &[mask[0][..1].try_into().unwrap()],
        );
        let constraint_log_degree_bound = self.log_size;
        evaluation_accumulator.accumulate(constraint_log_degree_bound, res[0]);
    }

    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, CPUBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CPUBackend>,
    ) {
        let max_chunk_size = (self.log_size / 2) as usize;
        if CPU_CHUNK_SIZE > max_chunk_size {
            self.evaluate_constraint_quotients_on_domain_chunks::<1>(trace, evaluation_accumulator);
            return;
        }
        self.evaluate_constraint_quotients_on_domain_chunks::<CPU_CHUNK_SIZE>(
            trace,
            evaluation_accumulator,
        );
    }
}
