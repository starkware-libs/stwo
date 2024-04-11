use itertools::Itertools;
use num_traits::Zero;

use super::trace_gen::{write_lookup_column, write_trace_row};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::ColumnVec;

pub const LOG_N_COLUMNS: usize = 6;
pub const N_COLUMNS: usize = 1 << LOG_N_COLUMNS;
pub const LOG_N_ROWS: usize = 4;

/// Component that computes fibonacci numbers over [N_COLUMNS] columns.
pub struct WideFibComponent {
    pub log_fibonacci_size: u32,
    pub log_n_instances: u32,
}

impl WideFibComponent {
    pub fn fill_initial_trace(&self, private_input: Vec<Input>) -> ColumnVec<Vec<BaseField>> {
        let n_instances = 1 << self.log_n_instances;
        assert_eq!(
            private_input.len(),
            n_instances,
            "The number of inputs must match the number of instances"
        );
        assert!(
            self.log_fibonacci_size >= LOG_N_COLUMNS as u32,
            "The fibonacci size must be at least equal to the length of a row"
        );
        let n_rows_per_instance = (1 << self.log_fibonacci_size) / N_COLUMNS;
        let n_rows = n_instances * n_rows_per_instance;
        let zero_vec = vec![BaseField::zero(); n_rows];
        let mut dst = vec![zero_vec; N_COLUMNS];
        (0..n_rows_per_instance).fold(private_input, |input, row| {
            (0..n_instances)
                .map(|instance| {
                    let (a, b) =
                        write_trace_row(&mut dst, &input[instance], row * n_instances + instance);
                    Input { a, b }
                })
                .collect_vec()
        });
        dst
    }

    pub fn lookup_columns(
        &self,
        trace: &[Vec<BaseField>],
        alpha: SecureField,
        z: SecureField,
    ) -> ColumnVec<Vec<SecureField>> {
        let n_rows = trace[0].len();
        let zero_vec = vec![SecureField::zero(); n_rows];
        let mut dst = vec![zero_vec; LOG_N_ROWS];
        write_lookup_column(&mut dst[0], trace, 0, alpha, z);
        write_lookup_column(&mut dst[1], trace, N_COLUMNS - 2, alpha, z);
        dst
    }

    pub fn log_column_size(&self) -> u32 {
        self.log_n_instances + self.log_fibonacci_size - LOG_N_COLUMNS as u32
    }
}

pub struct WideFibAir {
    pub component: WideFibComponent,
}

// Input for the fibonacci claim.
#[derive(Debug, Clone, Copy)]
pub struct Input {
    pub a: BaseField,
    pub b: BaseField,
}
