use std::ops::Div;

use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::mask::shifted_mask_points;
use crate::core::air::{Component, ComponentProver, ComponentTrace};
use crate::core::backend::CpuBackend;
use crate::core::circle::{CirclePoint, Coset};
use crate::core::constraints::point_vanishing;
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::ExtensionOf;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::bit_reverse_index;
use crate::core::{ColumnVec, InteractionElements, LookupValues};
use crate::trace_generation::registry::ComponentGenerationRegistry;
use crate::trace_generation::{ComponentGen, ComponentTraceGenerator, BASE_TRACE};

#[derive(Clone)]
pub struct RangeCheckComponent {
    pub log_size: u32,
    pub value: BaseField,
}

impl RangeCheckComponent {
    pub fn new(log_size: u32, value: BaseField) -> Self {
        Self { log_size, value }
    }

    /// Evaluates the step constraint quotient polynomial on a single point.
    fn step_constraint_eval_quotient_by_mask<F: ExtensionOf<BaseField>>(
        &self,
        point: CirclePoint<F>,
        mask: &[F; 16],
    ) -> F {
        let two = F::one().double();
        let constraint_zero_domain = Coset::subgroup(self.log_size);
        // Check if value at mask[0] equals value represented by 15 next bits little endian.
        // If value can be represented with 15 bits, it means that it is in range of 0..2^15
        let constraint_value = mask[0]
            - mask[1..]
                .iter()
                .enumerate()
                .map(|(i, &val)| val * two.pow(i as u128))
                .sum::<F>();
        let num = constraint_value;
        // Apply this step constrain on first row
        let denom = point_vanishing(constraint_zero_domain.at(0).into_ef(), point);
        num / denom
    }

    /// Evaluates the boundary constraint quotient polynomial on a single point.
    fn boundary_constraint_eval_quotient_by_mask<F: ExtensionOf<BaseField>>(
        &self,
        point: CirclePoint<F>,
        mask: &[F; 1],
    ) -> F {
        let constraint_zero_domain = Coset::subgroup(self.log_size);
        // Check if mask value is a binary 0 | 1
        let constraint_value = mask[0].square() - mask[0];
        let num = constraint_value;
        // Apply this boundary constrain on 1..16 trace rows
        let denom = (1..16)
            .map(|i| point_vanishing(constraint_zero_domain.at(i), point))
            .product::<F>();
        num / denom
    }
}

impl Component for RangeCheckComponent {
    fn n_constraints(&self) -> usize {
        2
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        // Step constraint is of degree 2.
        self.log_size + 1
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::new(vec![vec![self.log_size]])
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec::new(vec![shifted_mask_points(
            &vec![(0..16).collect::<Vec<_>>()],
            &[CanonicCoset::new(self.log_size)],
            point,
        )])
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        evaluation_accumulator.accumulate(
            self.step_constraint_eval_quotient_by_mask(point, &mask[0][0][..].try_into().unwrap()),
        );
        evaluation_accumulator.accumulate(self.boundary_constraint_eval_quotient_by_mask(
            point,
            &mask[0][0][..1].try_into().unwrap(),
        ));
    }
}

#[derive(Copy, Clone)]
pub struct RangeCheckInput {
    pub log_size: u32,
    pub value: BaseField,
}

#[derive(Clone)]
pub struct RangeCheckTraceGenerator {
    input: Option<RangeCheckInput>,
}

impl ComponentGen for RangeCheckTraceGenerator {}

impl RangeCheckTraceGenerator {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self { input: None }
    }

    pub fn inputs_set(&self) -> bool {
        self.input.is_some()
    }
}

impl ComponentTraceGenerator<CpuBackend> for RangeCheckTraceGenerator {
    type Component = RangeCheckComponent;
    type Inputs = RangeCheckInput;

    fn add_inputs(&mut self, inputs: &Self::Inputs) {
        assert!(!self.inputs_set(), "range_check input already set.");
        self.input = Some(*inputs);
    }

    fn write_trace(
        component_id: &str,
        registry: &mut ComponentGenerationRegistry,
    ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        let trace_generator = registry.get_generator_mut::<Self>(component_id);
        assert!(trace_generator.inputs_set(), "range_check input not set.");
        let trace_domain = CanonicCoset::new(trace_generator.input.unwrap().log_size);
        let mut trace = Vec::with_capacity(trace_domain.size());

        if let Some(input) = trace_generator.input {
            // Push the value to the trace.
            trace.push(input.value);

            // Fill trace with binary representation of value.
            let mut value_bits = input.value.0;
            for _ in 0..15 {
                trace.push(M31::from(value_bits & 0x1));
                value_bits >>= 1;
            }
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
        assert!(self.inputs_set(), "range_check input not set.");
        RangeCheckComponent::new(self.input.unwrap().log_size, self.input.unwrap().value)
    }
}

impl ComponentProver<CpuBackend> for RangeCheckComponent {
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
                let mask: [M31; 16] = (0..16)
                    .map(|j| eval[i as isize + j * mul])
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();

                let mut res = self.boundary_constraint_eval_quotient_by_mask(point, &[mask[0]])
                    * accum.random_coeff_powers[0];
                res += self.step_constraint_eval_quotient_by_mask(point, &mask)
                    * accum.random_coeff_powers[1];
                accum.accumulate(bit_reverse_index(i + off, constraint_log_degree_bound), res);
            }
        }
    }

    fn lookup_values(&self, _trace: &ComponentTrace<'_, CpuBackend>) -> LookupValues {
        LookupValues::default()
    }
}
