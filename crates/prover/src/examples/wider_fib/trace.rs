use std::collections::HashMap;

use itertools::Itertools;
use num_traits::One;

use crate::core::fields::m31::BaseField;
use crate::examples::wide_fibonacci::component::Input as NarrowFibInput;
use crate::examples::wide_fibonacci::trace_gen::component_output;

pub struct Input {
    secret: BaseField,
}

impl Input {
    pub fn new(secret: BaseField) -> Self {
        Self { secret }
    }
}

struct WideFibComponent {
    log_n_instances: u32,
    n_columns: usize,
}

impl WideFibComponent {
    pub fn write_trace(
        &self,
        trace: &mut [Vec<BaseField>],
        inpus: &[Input],
    ) -> Vec<BaseField> {
        inpus
            .iter()
            .enumerate()
            .map(|(i, input)| write_trace_row(trace, input, HashMap::new(), i))
            .collect_vec()
    }
}

pub fn write_trace_row(
    dst: &mut [Vec<BaseField>],
    input: &Input,
    registry: HashMap<String, ComponentGen>,
    row_index: usize,
) -> BaseField {
    dst[0][row_index] = BaseField::one();
    dst[1][row_index] = input.secret;

    let narrow_input = NarrowFibInput::new(dst[0][row_index], dst[1][row_index]);
    let fib_call_output = component_output(&narrow_input);
    registry.get("narrow_fib").add_input(narrow_input);

    dst[2][row_index] = fib_call_output.0;
    dst[3][row_index] = fib_call_output.1;

    let narrow_input = NarrowFibInput::new(dst[2][row_index], dst[3][row_index]);
    let fib_call_output = component_output(&narrow_input);
    registry.get("narrow_fib").add_input(narrow_input);

    dst[4][row_index] = fib_call_output.0;
    dst[5][row_index] = fib_call_output.1;

    let narrow_input = NarrowFibInput::new(dst[4][row_index], dst[5][row_index]);
    let fib_call_output = component_output(&narrow_input);
    registry.get("narrow_fib").add_input(narrow_input);

    dst[6][row_index] = fib_call_output.0;
    dst[7][row_index] = fib_call_output.1;

    return  dst[7][row_index];
}
