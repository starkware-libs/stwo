use itertools::Itertools;
use num_traits::Zero;

use super::registry::ComponentRegistry;
use super::{ComponentGen, TraceGenerator};
use crate::core::backend::CpuBackend;
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

type BinaryGateInput = (BaseField, BaseField);

#[derive(Default)]
struct AddGate {
    // The inputs to the add gate, left and right respectively.
    inputs: Vec<BinaryGateInput>,
}

impl ComponentGen for AddGate {}

impl AddGate {
    fn write_trace_row(
        dst: &mut [Vec<BaseField>],
        private_input: BinaryGateInput,
        row_index: usize,
    ) -> BaseField {
        let (left, right) = private_input;
        let result = left + right;
        dst[0][row_index] = left;
        dst[1][row_index] = right;
        dst[2][row_index] = result;
        result
    }

    pub fn deduce_output(input: &BinaryGateInput, _registy: &ComponentRegistry) -> BaseField {
        let (left, right) = input;
        *left + *right
    }
}

impl TraceGenerator<CpuBackend> for AddGate {
    type ComponentInputs = Vec<BinaryGateInput>;

    fn add_inputs(&mut self, inputs: &Self::ComponentInputs) {
        self.inputs.extend_from_slice(inputs);
    }

    fn write_trace(
        component_id: &str,
        registry: &mut ComponentRegistry,
    ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        let add_gate_component = registry.get_component::<AddGate>(component_id);
        let inputs_len = add_gate_component.inputs.len();
        let mut trace = vec![vec![BaseField::zero(); inputs_len]; 3];
        add_gate_component
            .inputs
            .iter()
            .enumerate()
            .for_each(|(i, input)| {
                AddGate::write_trace_row(&mut trace, *input, i);
            });
        let domain = CanonicCoset::new(inputs_len.ilog2()).circle_domain();
        trace
            .into_iter()
            .map(|eval| CircleEvaluation::<CpuBackend, _, BitReversedOrder>::new(domain, eval))
            .collect_vec()
    }
}

#[derive(Default)]
struct MulGate {
    // The inputs to the mul gate, left and right respectively.
    inputs: Vec<BinaryGateInput>,
}

impl ComponentGen for MulGate {}

impl MulGate {
    fn write_trace_row(
        dst: &mut [Vec<BaseField>],
        private_input: BinaryGateInput,
        row_index: usize,
    ) -> BaseField {
        let (left, right) = private_input;
        let result = left * right;
        dst[0][row_index] = left;
        dst[1][row_index] = right;
        dst[2][row_index] = result;
        result
    }

    pub fn deduce_output(input: &BinaryGateInput, _registy: &ComponentRegistry) -> BaseField {
        let (left, right) = input;
        *left * *right
    }
}

impl TraceGenerator<CpuBackend> for MulGate {
    type ComponentInputs = Vec<BinaryGateInput>;

    fn add_inputs(&mut self, inputs: &Self::ComponentInputs) {
        self.inputs.extend_from_slice(inputs);
    }

    fn write_trace(
        component_id: &str,
        registry: &mut ComponentRegistry,
    ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        let mul_gate_component = registry.get_component::<MulGate>(component_id);
        let inputs_len = mul_gate_component.inputs.len();
        let mut trace = vec![vec![BaseField::zero(); inputs_len]; 3];
        mul_gate_component
            .inputs
            .iter()
            .enumerate()
            .for_each(|(i, input)| {
                MulGate::write_trace_row(&mut trace, *input, i);
            });
        let domain = CanonicCoset::new(inputs_len.checked_ilog2().unwrap()).circle_domain();
        trace
            .into_iter()
            .map(|eval| CircleEvaluation::<CpuBackend, _, BitReversedOrder>::new(domain, eval))
            .collect_vec()
    }
}

#[derive(Default)]
struct ButterflyGate {
    // The inputs to the butterfly gate, left, right, and twiddle respectively.
    inputs: Vec<(BaseField, BaseField, BaseField)>,
}

impl ComponentGen for ButterflyGate {}

#[allow(clippy::type_complexity)]
impl ButterflyGate {
    pub fn write_trace_row(
        dst: &mut [Vec<BaseField>],
        registy: &ComponentRegistry,
        private_input: (BaseField, BaseField, BaseField),
        row_index: usize,
    ) -> (
        (BaseField, BaseField), // output.
        [BinaryGateInput; 2],   // add gates.
        [BinaryGateInput; 1],   // mul gates.
    ) {
        let (left, right, twiddle) = private_input;
        let mut add_gates_inputs = Vec::<BinaryGateInput>::with_capacity(2);
        let mut mul_gates_inputs = Vec::<BinaryGateInput>::with_capacity(1);

        dst[0][row_index] = left;
        dst[1][row_index] = right;
        dst[2][row_index] = twiddle;

        let mul_gate_input = (right, twiddle);
        let t0 = MulGate::deduce_output(&mul_gate_input, registy);
        mul_gates_inputs.push(mul_gate_input);

        let add_gate_input = (left, t0);
        let r0 = AddGate::deduce_output(&add_gate_input, registy);
        add_gates_inputs.push(add_gate_input);

        dst[3][row_index] = r0;

        // TODO change to SubGate.
        let add_gate_input = (left, t0);
        let r1 = AddGate::deduce_output(&add_gate_input, registy);
        add_gates_inputs.push(add_gate_input);

        dst[4][row_index] = r1;

        (
            (r0, r1),
            add_gates_inputs.try_into().unwrap(),
            mul_gates_inputs.try_into().unwrap(),
        )
    }

    #[allow(dead_code)]
    pub fn deduce_output(
        input: (BaseField, BaseField, BaseField),
        _registy: &ComponentRegistry,
    ) -> (BaseField, BaseField) {
        let (left, right, twiddle) = input;
        (left + right * twiddle, left - right * twiddle)
    }
}

impl TraceGenerator<CpuBackend> for ButterflyGate {
    type ComponentInputs = Vec<(BaseField, BaseField, BaseField)>;

    fn add_inputs(&mut self, inputs: &Self::ComponentInputs) {
        self.inputs.extend_from_slice(inputs);
    }

    fn write_trace(
        component_id: &str,
        registry: &mut ComponentRegistry,
    ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        let butterfly_component = registry.get_component::<ButterflyGate>(component_id);
        let inputs_len = butterfly_component.inputs.len();
        let mut trace = vec![vec![BaseField::zero(); inputs_len]; 5];
        let res: Vec<_> = butterfly_component
            .inputs
            .iter()
            .enumerate()
            .map(|(i, input)| ButterflyGate::write_trace_row(&mut trace, registry, *input, i))
            .collect();

        let (_, add_gates_inputs, mul_gates_inputs): (Vec<_>, Vec<_>, Vec<_>) =
            res.into_iter().multiunzip();

        let mul_component = registry.get_component_mut::<MulGate>("MulGate");
        mul_component.add_inputs(&mul_gates_inputs.flatten().to_vec());

        let add_component = registry.get_component_mut::<AddGate>("AddGate");
        add_component.add_inputs(&add_gates_inputs.flatten().to_vec());

        let domain = CanonicCoset::new(inputs_len.checked_ilog2().unwrap()).circle_domain();
        trace
            .into_iter()
            .map(|eval| CircleEvaluation::<CpuBackend, _, BitReversedOrder>::new(domain, eval))
            .collect_vec()
    }
}

#[test]
fn test_butterfly_component() {
    let mut registry = ComponentRegistry::default();
    let butterfly_component = Box::new(ButterflyGate {
        inputs: (0..16)
            .map(|i| {
                (
                    BaseField::from_u32_unchecked(i),
                    BaseField::from_u32_unchecked(2 * i),
                    BaseField::from_u32_unchecked(3 * i),
                )
            })
            .collect(),
    });
    registry.register_component("ButterflyGate".to_string(), butterfly_component);
    registry.register_component("MulGate".to_string(), Box::<MulGate>::default());
    registry.register_component("AddGate".to_string(), Box::<AddGate>::default());

    ButterflyGate::write_trace("ButterflyGate", &mut registry);

    assert_eq!(
        registry.get_component::<MulGate>("MulGate").inputs.len(),
        16
    );
    assert_eq!(
        registry.get_component::<AddGate>("AddGate").inputs.len(),
        16 * 2
    );
}
