use itertools::Itertools;
use num_traits::{One, Zero};

use super::component::Input;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::utils::{
    bit_reverse, circle_domain_order_to_coset_order, shifted_secure_combination,
};

/// Writes the trace row for the wide Fibonacci example to dst, given a private input. Returns the
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

/// Writes and returns the lookup column for the wide Fibonacci example, which is the partial
/// product of the shifted secure combination of the first two elements in each row divided by the
/// the shifted secure combination of the last two elements in each row.
pub fn write_lookup_column(
    input_trace: &[&[BaseField]],
    alpha: SecureField,
    z: SecureField,
) -> Vec<SecureField> {
    let n_rows = input_trace[0].len();
    let n_columns = input_trace.len();
    let mut prev_value = SecureField::one();
    let mut input_trace = input_trace
        .iter()
        .map(|column| column.to_vec())
        .collect_vec();
    let natural_ordered_trace = input_trace
        .iter_mut()
        .map(|column| {
            bit_reverse(column);
            circle_domain_order_to_coset_order(column)
        })
        .collect_vec();

    let denominators = (0..n_rows)
        .map(|i| {
            shifted_secure_combination(
                &[
                    natural_ordered_trace[n_columns - 2][i],
                    natural_ordered_trace[n_columns - 1][i],
                ],
                alpha,
                z,
            )
        })
        .collect_vec();
    let mut denominator_inverses = vec![SecureField::zero(); denominators.len()];
    SecureField::batch_inverse(&denominators, &mut denominator_inverses);

    (0..n_rows)
        .map(|i| {
            let numerator = shifted_secure_combination(
                &[natural_ordered_trace[0][i], natural_ordered_trace[1][i]],
                alpha,
                z,
            );
            let cell = (numerator * denominator_inverses[i]) * prev_value;
            prev_value = cell;
            cell
        })
        .collect_vec()
}
