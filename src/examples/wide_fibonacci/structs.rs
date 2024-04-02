use num_traits::Zero;

use super::trace_gen::write_trace_row;
use crate::core::fields::m31::BaseField;
use crate::core::ColumnVec;

pub const N_COLUMNS: usize = 64;

/// Component that computes fibonacci numbers over [N_COLUMNS] columns.
pub struct WideFibComponent {
    pub log_fibonacci_size: u32,
    pub log_n_instances: u32,
}

impl WideFibComponent {
    pub fn fill_trace(&self, private_input: &[Input]) -> ColumnVec<Vec<BaseField>> {
        let n_instances = 1 << self.log_n_instances;
        assert_eq!(private_input.len(), n_instances);
        let n_rows_per_instance = (1 << self.log_fibonacci_size) / N_COLUMNS;
        let n_rows = n_instances * n_rows_per_instance;
        let zero_vec = vec![BaseField::zero(); n_rows];
        let mut dst = vec![zero_vec; N_COLUMNS];
        for (ith_fib, input) in private_input.iter().enumerate() {
            (0..n_rows_per_instance).fold((input.a, input.b), |(a, b), row_offset| {
                write_trace_row(&mut dst, &Input { a, b }, ith_fib + row_offset)
            });
        }
        dst
    }
}

// Input for the fibonacci claim.
#[derive(Debug, Clone, Copy)]
pub struct Input {
    pub a: BaseField,
    pub b: BaseField,
}
