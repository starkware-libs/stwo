use dashmap::DashMap;
use itertools::Itertools;

use crate::core::air::{Component, Components};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::CircleDomain;
use crate::core::verifier::PREPROCESSED_TRACE_IDX;
use crate::core::ColumnVec;
use crate::prover::air::accumulation::{DomainEvaluationAccumulator, EvaluationMode};
use crate::prover::backend::{Backend, Col};
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, SecureCirclePoly};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::{CirclePoint, ProvingError};

/// Type alias for the weights hash map used in barycentric eval_at_point.
pub type WeightsHashMap<B> = DashMap<(u32, CirclePoint<SecureField>), Col<B, SecureField>>;

pub trait ComponentProver<B: Backend>: Component {
    /// Evaluates the constraint quotients of the component on the evaluation domain.
    /// Accumulates quotients in `evaluation_accumulator`.
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, B>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<B>,
    );

    /// Evaluates constraint quotients using an explicit constraint degree bound.
    ///
    /// Implementations that can evaluate on verifier-selected ZK domains should
    /// override this method. The default fails closed so unsupported component
    /// provers cannot silently ignore the explicit degree bound.
    fn evaluate_constraint_quotients_on_domain_with_log_degree_bound(
        &self,
        trace: &Trace<'_, B>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<B>,
        max_constraint_log_degree_bound: u32,
    ) -> Result<(), ProvingError> {
        let _ = (
            trace,
            evaluation_accumulator,
            max_constraint_log_degree_bound,
        );
        Err(ProvingError::InvalidZkDegreeGeometry)
    }
}

/// The set of polynomials that make up the trace.
pub struct Trace<'a, B: Backend> {
    /// Polynomials for each column.
    pub polys: TreeVec<ColumnVec<&'a Poly<B>>>,
}

/// A struct for representing a polynomial corresponding to a trace column.
/// A polynomial is defined by it's evaluations on a circle domain of size at least it's degree,
/// and optionally its coefficients in the FFT basis.
pub struct Poly<B: Backend> {
    pub coeffs: Option<CircleCoefficients<B>>,
    pub evals: CircleEvaluation<B, BaseField, BitReversedOrder>,
}

impl<B: Backend> Poly<B> {
    pub const fn new(
        coeffs: Option<CircleCoefficients<B>>,
        evals: CircleEvaluation<B, BaseField, BitReversedOrder>,
    ) -> Self {
        Self { coeffs, evals }
    }

    pub fn eval_at_point(
        &self,
        point: CirclePoint<SecureField>,
        weights_hash_map: Option<&WeightsHashMap<B>>,
    ) -> SecureField {
        if let Some(coeffs) = &self.coeffs {
            coeffs.eval_at_point(point)
        } else {
            self.evals.barycentric_eval_at_point(
                &weights_hash_map
                    .unwrap()
                    .get(&(self.evals.domain.log_size(), point))
                    .expect("weights should exist for all sampled points"),
            )
        }
    }

    pub fn get_evaluation_on_domain(
        &self,
        domain: CircleDomain,
        twiddles: &TwiddleTree<B>,
    ) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
        if let Some(coeffs) = &self.coeffs {
            coeffs.evaluate_with_twiddles(domain, twiddles)
        } else {
            panic!("The polynomial's coefficients are not stored");
        }
    }
}

pub struct ComponentProvers<'a, B: Backend> {
    pub components: Vec<&'a dyn ComponentProver<B>>,
    pub n_preprocessed_columns: usize,
}

impl<B: Backend> ComponentProvers<'_, B> {
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

    fn component_trace_log_degree_bounds_with_zk_bounds(
        &self,
        column_log_degree_bounds: &TreeVec<ColumnVec<u32>>,
    ) -> Result<Vec<u32>, ProvingError> {
        if column_log_degree_bounds
            .get(PREPROCESSED_TRACE_IDX)
            .map(|bounds| bounds.len())
            != Some(self.n_preprocessed_columns)
        {
            return Err(ProvingError::InvalidZkDegreeGeometry);
        }

        let mut next_tree_offsets = vec![0usize; column_log_degree_bounds.len()];
        let mut component_trace_log_degree_bounds = Vec::with_capacity(self.components.len());

        for component in &self.components {
            let component_bounds = component.trace_log_degree_bounds();
            if component_bounds.len() > column_log_degree_bounds.len() {
                return Err(ProvingError::InvalidZkDegreeGeometry);
            }

            let preprocessed_column_indices = component.preprocessed_column_indices();
            let mut component_trace_log_degree_bound = 0;
            for (tree_index, local_tree_bounds) in component_bounds.iter().enumerate() {
                let Some(global_tree_bounds) = column_log_degree_bounds.get(tree_index) else {
                    return Err(ProvingError::InvalidZkDegreeGeometry);
                };

                if tree_index == PREPROCESSED_TRACE_IDX {
                    if preprocessed_column_indices.len() != local_tree_bounds.len() {
                        return Err(ProvingError::InvalidZkDegreeGeometry);
                    }
                    for &column_index in &preprocessed_column_indices {
                        let Some(&log_degree_bound) = global_tree_bounds.get(column_index) else {
                            return Err(ProvingError::InvalidZkDegreeGeometry);
                        };
                        component_trace_log_degree_bound =
                            component_trace_log_degree_bound.max(log_degree_bound);
                    }
                } else {
                    let start = next_tree_offsets[tree_index];
                    let end = start
                        .checked_add(local_tree_bounds.len())
                        .ok_or(ProvingError::InvalidZkDegreeGeometry)?;
                    if end > global_tree_bounds.len() {
                        return Err(ProvingError::InvalidZkDegreeGeometry);
                    }
                    for &log_degree_bound in &global_tree_bounds[start..end] {
                        component_trace_log_degree_bound =
                            component_trace_log_degree_bound.max(log_degree_bound);
                    }
                    next_tree_offsets[tree_index] = end;
                }
            }
            component_trace_log_degree_bounds.push(component_trace_log_degree_bound);
        }

        for (tree_index, (&used_columns, global_tree_bounds)) in next_tree_offsets
            .iter()
            .zip(column_log_degree_bounds.iter())
            .enumerate()
        {
            if tree_index != PREPROCESSED_TRACE_IDX && used_columns != global_tree_bounds.len() {
                return Err(ProvingError::InvalidZkDegreeGeometry);
            }
        }

        Ok(component_trace_log_degree_bounds)
    }

    pub fn compute_composition_polynomial(
        &self,
        random_coeff: SecureField,
        trace: &Trace<'_, B>,
        twiddles: &TwiddleTree<B>,
        log_blowup_factor: u32,
    ) -> SecureCirclePoly<B> {
        let total_constraints: usize = self.components.iter().map(|c| c.n_constraints()).sum();
        let components: Vec<&dyn Component> = self
            .components
            .iter()
            .map(|c| *c as &dyn Component)
            .collect();
        let evaluation_mode = EvaluationMode::infer(&components, log_blowup_factor);
        let mut accumulator = DomainEvaluationAccumulator::new(
            random_coeff,
            self.components().composition_log_degree_bound(),
            total_constraints,
            evaluation_mode,
        );
        for component in &self.components {
            component.evaluate_constraint_quotients_on_domain(trace, &mut accumulator)
        }
        accumulator.finalize(twiddles)
    }

    /// Computes the composition polynomial using an explicit accumulator degree
    /// bound.
    ///
    /// This is used by the explicit ZK STARK path to bind composition
    /// generation to the verifier-owned ZK degree profile. It does not by
    /// itself activate private-witness STARK ZK: component evaluators must still
    /// expose reviewed ZK-aware constraint evaluation domains before private
    /// activation can be unblocked.
    pub(crate) fn compute_composition_polynomial_with_log_degree_bound(
        &self,
        random_coeff: SecureField,
        trace: &Trace<'_, B>,
        twiddles: &TwiddleTree<B>,
        log_blowup_factor: u32,
        trace_log_degree_bound: u32,
        column_log_degree_bounds: &TreeVec<ColumnVec<u32>>,
        composition_log_degree_bound: u32,
    ) -> Result<SecureCirclePoly<B>, ProvingError> {
        let total_constraints: usize = self.components.iter().map(|c| c.n_constraints()).sum();
        let component_trace_log_degree_bounds =
            self.component_trace_log_degree_bounds_with_zk_bounds(column_log_degree_bounds)?;
        let max_component_trace_log_degree_bound = component_trace_log_degree_bounds
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        if trace_log_degree_bound < max_component_trace_log_degree_bound
            || composition_log_degree_bound < trace_log_degree_bound
        {
            return Err(ProvingError::InvalidZkDegreeGeometry);
        }

        let component_derived_composition_log_degree_bound =
            self.components().composition_log_degree_bound();
        let composition_log_degree_delta = composition_log_degree_bound
            .checked_sub(component_derived_composition_log_degree_bound)
            .ok_or(ProvingError::InvalidZkDegreeGeometry)?;
        let mut explicit_component_constraint_log_degree_bounds =
            Vec::with_capacity(self.components.len());
        let mut constrained_trace_log_degree_bounds = Vec::new();
        let mut constrained_constraint_log_degree_bounds = Vec::new();
        for (component, &component_trace_log_degree_bound) in self
            .components
            .iter()
            .zip(&component_trace_log_degree_bounds)
        {
            let component_constraint_log_degree_bound = component
                .max_constraint_log_degree_bound()
                .checked_add(composition_log_degree_delta)
                .ok_or(ProvingError::InvalidZkDegreeGeometry)?;
            if component_constraint_log_degree_bound > composition_log_degree_bound {
                return Err(ProvingError::InvalidZkDegreeGeometry);
            }
            if component.n_constraints() != 0 {
                if component_constraint_log_degree_bound < component_trace_log_degree_bound {
                    return Err(ProvingError::InvalidZkDegreeGeometry);
                }
                constrained_trace_log_degree_bounds.push(component_trace_log_degree_bound);
                constrained_constraint_log_degree_bounds
                    .push(component_constraint_log_degree_bound);
            }
            explicit_component_constraint_log_degree_bounds
                .push(component_constraint_log_degree_bound);
        }

        let evaluation_mode = EvaluationMode::infer_from_component_bounds(
            &constrained_trace_log_degree_bounds,
            &constrained_constraint_log_degree_bounds,
            log_blowup_factor,
        )
        .ok_or(ProvingError::InvalidZkDegreeGeometry)?;
        let mut accumulator = DomainEvaluationAccumulator::new(
            random_coeff,
            composition_log_degree_bound,
            total_constraints,
            evaluation_mode,
        );
        for (component, &component_constraint_log_degree_bound) in self
            .components
            .iter()
            .zip(&explicit_component_constraint_log_degree_bounds)
        {
            component.evaluate_constraint_quotients_on_domain_with_log_degree_bound(
                trace,
                &mut accumulator,
                component_constraint_log_degree_bound,
            )?;
        }
        Ok(accumulator.finalize(twiddles))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::air::accumulation::PointEvaluationAccumulator;
    use crate::prover::backend::CpuBackend;

    struct FakeComponent {
        trace_log_degree_bounds: TreeVec<ColumnVec<u32>>,
        preprocessed_column_indices: ColumnVec<usize>,
    }

    impl Component for FakeComponent {
        fn n_constraints(&self) -> usize {
            1
        }

        fn max_constraint_log_degree_bound(&self) -> u32 {
            8
        }

        fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
            self.trace_log_degree_bounds.clone()
        }

        fn mask_points(
            &self,
            _point: CirclePoint<SecureField>,
            _max_log_degree_bound: u32,
        ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
            unimplemented!("not needed by component trace-bound mapping tests")
        }

        fn preprocessed_column_indices(&self) -> ColumnVec<usize> {
            self.preprocessed_column_indices.clone()
        }

        fn evaluate_constraint_quotients_at_point(
            &self,
            _point: CirclePoint<SecureField>,
            _mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
            _evaluation_accumulator: &mut PointEvaluationAccumulator,
            _max_log_degree_bound: u32,
        ) {
            unimplemented!("not needed by component trace-bound mapping tests")
        }
    }

    impl ComponentProver<CpuBackend> for FakeComponent {
        fn evaluate_constraint_quotients_on_domain(
            &self,
            _trace: &Trace<'_, CpuBackend>,
            _evaluation_accumulator: &mut DomainEvaluationAccumulator<CpuBackend>,
        ) {
            unimplemented!("not needed by component trace-bound mapping tests")
        }
    }

    #[test]
    fn component_trace_bounds_use_zk_global_column_bounds() {
        let component0 = FakeComponent {
            trace_log_degree_bounds: TreeVec::new(vec![vec![3], vec![3, 3]]),
            preprocessed_column_indices: vec![1],
        };
        let component1 = FakeComponent {
            trace_log_degree_bounds: TreeVec::new(vec![vec![2], vec![2]]),
            preprocessed_column_indices: vec![0],
        };
        let component_provers = ComponentProvers::<CpuBackend> {
            components: vec![&component0, &component1],
            n_preprocessed_columns: 2,
        };
        let zk_column_log_degree_bounds = TreeVec::new(vec![vec![5, 4], vec![6, 3, 7]]);

        assert_eq!(
            component_provers
                .component_trace_log_degree_bounds_with_zk_bounds(&zk_column_log_degree_bounds)
                .unwrap(),
            vec![6, 7]
        );
    }

    #[test]
    fn component_trace_bounds_reject_unconsumed_zk_columns() {
        let component = FakeComponent {
            trace_log_degree_bounds: TreeVec::new(vec![vec![], vec![3]]),
            preprocessed_column_indices: vec![],
        };
        let component_provers = ComponentProvers::<CpuBackend> {
            components: vec![&component],
            n_preprocessed_columns: 0,
        };
        let zk_column_log_degree_bounds = TreeVec::new(vec![vec![], vec![3, 4]]);

        assert!(matches!(
            component_provers
                .component_trace_log_degree_bounds_with_zk_bounds(&zk_column_log_degree_bounds),
            Err(ProvingError::InvalidZkDegreeGeometry)
        ));
    }

    #[test]
    fn component_trace_bounds_reject_wrong_preprocessed_column_count() {
        let component = FakeComponent {
            trace_log_degree_bounds: TreeVec::new(vec![vec![3]]),
            preprocessed_column_indices: vec![0],
        };
        let component_provers = ComponentProvers::<CpuBackend> {
            components: vec![&component],
            n_preprocessed_columns: 2,
        };
        let zk_column_log_degree_bounds = TreeVec::new(vec![vec![3]]);

        assert!(matches!(
            component_provers
                .component_trace_log_degree_bounds_with_zk_bounds(&zk_column_log_degree_bounds),
            Err(ProvingError::InvalidZkDegreeGeometry)
        ));
    }
}
