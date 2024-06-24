use std::ops::Div;

use num_traits::One;

use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::mask::shifted_mask_points;
use crate::core::air::{Component, ComponentProver, ComponentTrace};
use crate::core::backend::CpuBackend;
use crate::core::circle::{CirclePoint, Coset};
use crate::core::constraints::{coset_vanishing, pair_vanishing};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{ExtensionOf, FieldExpOps};
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::BASE_TRACE;
use crate::core::utils::bit_reverse_index;
use crate::core::{ColumnVec, InteractionElements, LookupValues};
use crate::trace_generation::registry::ComponentGenerationRegistry;
use crate::trace_generation::{ComponentGen, ComponentTraceGenerator};

#[derive(Clone)]
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
        let selector = pair_vanishing(
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
        // 1 + y * (self.claim - 1) * p.y^-1
        // TODO(spapini): Cache the constant.
        let linear = F::one() + point.y * (self.claim - BaseField::one()) * p.y.inverse();

        let num = mask[0] - linear;
        let denom = pair_vanishing(p.into_ef(), CirclePoint::zero(), point);
        num / denom
    }
}

impl Component for FibonacciComponent {
    fn n_constraints(&self) -> usize {
        2
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        // Step constraint is of degree 2.
        self.log_size + 1
    }

    fn n_interaction_phases(&self) -> u32 {
        1
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::new(vec![vec![self.log_size], vec![]])
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec::new(vec![
            shifted_mask_points(
                &vec![vec![0, 1, 2]],
                &[CanonicCoset::new(self.log_size)],
                point,
            ),
            vec![],
        ])
    }

    fn interaction_element_ids(&self) -> Vec<String> {
        vec![]
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &ColumnVec<Vec<SecureField>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        evaluation_accumulator.accumulate(
            self.step_constraint_eval_quotient_by_mask(point, &mask[0][..].try_into().unwrap()),
        );
        evaluation_accumulator.accumulate(
            self.boundary_constraint_eval_quotient_by_mask(
                point,
                &mask[0][..1].try_into().unwrap(),
            ),
        );
    }
}

pub struct FibonacciInput(pub u32, pub BaseField);

#[derive(Clone)]
pub struct FibonacciTraceGenerator {
    log_size: Option<u32>,
    claim: Option<BaseField>,
}

impl ComponentGen for FibonacciTraceGenerator {}

impl FibonacciTraceGenerator {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            log_size: None,
            claim: None,
        }
    }

    pub fn inputs_set(&self) -> bool {
        self.log_size.is_some() && self.claim.is_some()
    }
}

impl ComponentTraceGenerator<CpuBackend> for FibonacciTraceGenerator {
    type Component = FibonacciComponent;
    type Inputs = FibonacciInput;

    fn add_inputs(&mut self, inputs: &Self::Inputs) {
        assert!(!self.inputs_set(), "Fibonacci input already set.");
        self.log_size = Some(inputs.0);
        self.claim = Some(inputs.1);
    }

    fn write_trace(
        component_id: &str,
        registry: &mut ComponentGenerationRegistry,
    ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        let trace_generator = registry.get_generator_mut::<Self>(component_id);
        assert!(trace_generator.inputs_set(), "Fibonacci input not set.");
        let trace_domain = CanonicCoset::new(trace_generator.log_size.unwrap());
        let mut trace = Vec::with_capacity(trace_domain.size());

        // Fill trace with fibonacci squared.
        let mut a = BaseField::one();
        let mut b = BaseField::one();
        for _ in 0..trace_domain.size() {
            trace.push(a);
            let tmp = a.square() + b.square();
            a = b;
            b = tmp;
        }

        // Returns as a CircleEvaluation.
        vec![CircleEvaluation::new_canonical_ordered(trace_domain, trace)]
    }

    fn write_interaction_trace(
        &self,
        _trace: &ColumnVec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
        _elements: &InteractionElements,
    ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        vec![]
    }

    fn component(&self) -> Self::Component {
        assert!(self.inputs_set(), "Fibonacci input not set.");
        FibonacciComponent::new(self.log_size.unwrap(), self.claim.unwrap())
    }
}

impl ComponentProver<CpuBackend> for FibonacciComponent {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, CpuBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CpuBackend>,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        let poly = &trace.polys[BASE_TRACE][0];
        let trace_domain = CanonicCoset::new(self.log_size);
        let trace_eval_domain = CanonicCoset::new(self.log_size + 1).circle_domain();
        let trace_eval = poly.evaluate(trace_eval_domain).bit_reverse();

        // Step constraint.
        let constraint_log_degree_bound = trace_domain.log_size() + 1;
        let [mut accum] = evaluation_accumulator.columns([(constraint_log_degree_bound, 2)]);
        let constraint_eval_domain = trace_eval_domain;
        for (off, point_coset) in [
            (0, constraint_eval_domain.half_coset),
            (
                constraint_eval_domain.half_coset.size(),
                constraint_eval_domain.half_coset.conjugate(),
            ),
        ] {
            let eval = trace_eval.fetch_eval_on_coset(point_coset.shift(trace_domain.index_at(0)));
            let mul = trace_domain.step_size().div(point_coset.step_size);
            for (i, point) in point_coset.iter().enumerate() {
                let mask = [eval[i], eval[i as isize + mul], eval[i as isize + 2 * mul]];
                let mut res = self.boundary_constraint_eval_quotient_by_mask(point, &[mask[0]])
                    * accum.random_coeff_powers[0];
                res += self.step_constraint_eval_quotient_by_mask(point, &mask)
                    * accum.random_coeff_powers[1];
                accum.accumulate(bit_reverse_index(i + off, constraint_log_degree_bound), res);
            }
        }
    }
}
