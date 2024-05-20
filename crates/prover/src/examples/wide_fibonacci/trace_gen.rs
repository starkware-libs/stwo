use num_traits::{One, Zero};

use super::component::{Input, N_COLUMNS};
use crate::core::fields::m31::BaseField;
use crate::core::fields::FieldExpOps;
use crate::core::utils::shifted_secure_combination;

pub fn component_output(input: &Input) -> (BaseField, BaseField) {
    let mut trace = vec![vec![BaseField::zero(); 1]; N_COLUMNS];
    let output = write_trace_row(&mut trace, input, 0);
    output
}

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

pub fn write_lookup_column(
    input_trace: &[Vec<BaseField>],
    // TODO(AlonH): Change alpha and z to SecureField.
    alpha: BaseField,
    z: BaseField,
) -> Vec<BaseField> {
    let n_rows = input_trace[0].len();
    let n_columns = input_trace.len();
    let mut prev_value = BaseField::one();
    (0..n_rows)
        .map(|i| {
            let numerator =
                shifted_secure_combination(&[input_trace[0][i], input_trace[1][i]], alpha, z);
            let denominator = shifted_secure_combination(
                &[input_trace[n_columns - 2][i], input_trace[n_columns - 1][i]],
                alpha,
                z,
            );
            // TODO(AlonH): Use batch inversion.
            let cell = (numerator / denominator) * prev_value;
            prev_value = cell;
            cell
        })
        .collect()
}
