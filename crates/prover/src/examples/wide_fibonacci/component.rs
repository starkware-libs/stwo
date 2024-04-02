use itertools::Itertools;
use num_traits::Zero;

use super::trace_gen::write_trace_row;
use crate::core::air::{Air, Component};
use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;
use crate::core::ColumnVec;

pub const LOG_N_COLUMNS: usize = 8;
pub const N_COLUMNS: usize = 1 << LOG_N_COLUMNS;

/// Component that computes 2^`self.log_n_instances` instances of fibonacci sequences of size
/// 2^`self.log_fibonacci_size`. The numbers are computes over [N_COLUMNS] trace columns. The
/// number of rows (i.e the size of the columns) is determined by the parameters above (see
/// [WideFibComponent::log_column_size()]).
pub struct WideFibComponent {
    pub log_fibonacci_size: u32,
    pub log_n_instances: u32,
}

impl WideFibComponent {
    /// Returns the log of the size of the columns in the trace (which could also be looked at as
    /// the log number of rows).
    pub fn log_column_size(&self) -> u32 {
        // log(n_instances * fibonacci_size / N_COLUMNS)
        self.log_n_instances + self.log_fibonacci_size - LOG_N_COLUMNS as u32
    }

    pub fn log_n_columns(&self) -> usize {
        LOG_N_COLUMNS
    }

    pub fn n_columns(&self) -> usize {
        N_COLUMNS
    }
}

pub struct WideFibAir {
    pub component: WideFibComponent,
}

impl Air<CPUBackend> for WideFibAir {
    fn components(&self) -> Vec<&dyn Component<CPUBackend>> {
        vec![&self.component]
    }
}

pub fn fill_initial_trace(
    wide_fib: &WideFibComponent,
    private_input: Vec<Input>,
) -> ColumnVec<Vec<BaseField>> {
    let n_instances = 1 << wide_fib.log_n_instances;
    assert_eq!(private_input.len(), n_instances);
    let n_rows_per_instance = 1 << (wide_fib.log_fibonacci_size - wide_fib.log_n_columns() as u32);
    let n_rows = n_instances * n_rows_per_instance;
    let zero_vec = vec![BaseField::zero(); n_rows];
    let mut dst = vec![zero_vec; wide_fib.n_columns()];
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

// Input for the fibonacci claim.
#[derive(Debug, Clone, Copy)]
pub struct Input {
    pub a: BaseField,
    pub b: BaseField,
}
