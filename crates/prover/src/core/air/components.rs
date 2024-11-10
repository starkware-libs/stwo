use itertools::Itertools;

use super::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use super::{Component, ComponentProver, Trace};
use crate::core::backend::Backend;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::SecureCirclePoly;
use crate::core::ColumnVec;

pub struct Components<'a> {
    pub components: Vec<&'a dyn Component>,
    pub n_preprocessed_columns: usize,
}

impl<'a> Components<'a> {
    pub fn composition_log_degree_bound(&self) -> u32 {
        self.components
            .iter()
            .map(|component| component.max_constraint_log_degree_bound())
            .max()
            .unwrap()
    }

    pub fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec::concat_cols(
            self.components
                .iter()
                .map(|component| component.mask_points(point)),
        )
    }

    pub fn eval_composition_polynomial_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask_values: &TreeVec<Vec<Vec<SecureField>>>,
        random_coeff: SecureField,
    ) -> SecureField {
        let mut evaluation_accumulator = PointEvaluationAccumulator::new(random_coeff);
        for component in &self.components {
            component.evaluate_constraint_quotients_at_point(
                point,
                mask_values,
                &mut evaluation_accumulator,
            )
        }
        evaluation_accumulator.finalize()
    }

    pub fn column_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::concat_cols(
            self.components
                .iter()
                .map(|component| component.trace_log_degree_bounds()),
        )
    }
}

pub struct ComponentProvers<'a, B: Backend> {
    pub components: Vec<&'a dyn ComponentProver<B>>,
    pub n_preprocessed_columns: usize,
}

impl<'a, B: Backend> ComponentProvers<'a, B> {
    pub fn components(&self) -> Components<'_> {
        Components {
            components: self
                .components
                .iter()
                .map(|c| *c as &dyn Component)
                .collect_vec(),
            n_preprocessed_columns: self.n_preprocessed_columns,
        }
    }
    pub fn compute_composition_polynomial(
        &self,
        random_coeff: SecureField,
        trace: &Trace<'_, B>,
    ) -> SecureCirclePoly<B> {
        let total_constraints: usize = self.components.iter().map(|c| c.n_constraints()).sum();
        let mut accumulator = DomainEvaluationAccumulator::new(
            random_coeff,
            self.components().composition_log_degree_bound(),
            total_constraints,
        );
        for component in &self.components {
            component.evaluate_constraint_quotients_on_domain(trace, &mut accumulator)
        }
        accumulator.finalize()
    }
}
