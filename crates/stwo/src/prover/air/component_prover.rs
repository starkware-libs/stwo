use itertools::Itertools;

use crate::core::air::{Component, Components};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::CircleDomain;
use crate::core::ColumnVec;
use crate::prover::air::accumulation::DomainEvaluationAccumulator;
use crate::prover::backend::Backend;
use crate::prover::poly::circle::{CircleEvaluation, CirclePoly, SecureCirclePoly};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::CirclePoint;

pub trait ComponentProver<B: Backend>: Component {
    /// Evaluates the constraint quotients of the component on the evaluation domain.
    /// Accumulates quotients in `evaluation_accumulator`.
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, B>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<B>,
    );
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
    pub coeffs: Option<CirclePoly<B>>,
    pub evals: CircleEvaluation<B, BaseField, BitReversedOrder>,
}

impl<B: Backend> Poly<B> {
    pub const fn new(
        coeffs: Option<CirclePoly<B>>,
        evals: CircleEvaluation<B, BaseField, BitReversedOrder>,
    ) -> Self {
        Self { coeffs, evals }
    }

    pub fn eval_at_point(
        &self,
        point: CirclePoint<SecureField>,
        twiddles: &TwiddleTree<B>,
    ) -> SecureField {
        if let Some(coeffs) = &self.coeffs {
            coeffs.eval_at_point(point)
        } else {
            self.evals.eval_at_point_by_folding(point, twiddles)
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
