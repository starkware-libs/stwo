use dashmap::DashMap;
use itertools::Itertools;

use crate::core::air::{Component, Components};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::CircleDomain;
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
        composition_log_degree_bound: u32,
    ) -> Result<SecureCirclePoly<B>, ProvingError> {
        let total_constraints: usize = self.components.iter().map(|c| c.n_constraints()).sum();
        let component_trace_log_degree_bounds = self
            .components
            .iter()
            .map(|component| {
                component
                    .trace_log_degree_bounds()
                    .iter()
                    .flatten()
                    .copied()
                    .max()
                    .unwrap_or(0)
            })
            .collect_vec();
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
