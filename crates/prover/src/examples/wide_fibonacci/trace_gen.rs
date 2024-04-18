use super::component::Input;
use crate::core::fields::m31::BaseField;
use crate::core::fields::FieldExpOps;

/// Given a private input, write the trace row for the wide Fibonacci example to dst. Returns the
/// last two elements of the row in case the sequence is continued.
pub fn write_trace_row(
    dst: &mut [Vec<BaseField>],
    private_input: &Input,
    row_index: usize,
) -> (BaseField, BaseField) {
    let n_columns = dst.len();
    dst[0][row_index] = private_input.a;
    dst[1][row_index] = private_input.b;
    for i in 2..n_columns {
        dst[i][row_index] = dst[i - 1][row_index].square() + dst[i - 2][row_index].square();
    }

    (dst[n_columns - 2][row_index], dst[n_columns - 1][row_index])
}
