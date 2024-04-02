use itertools::Itertools;
use num_traits::Zero;

use super::trace_gen::write_trace_row;
use crate::core::air::{Air, Component};
use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;
use crate::core::ColumnVec;

pub const LOG_N_COLUMNS: usize = 8;
pub const N_COLUMNS: usize = 1 << LOG_N_COLUMNS;

/// Component that computes fibonacci numbers over [N_COLUMNS] columns.
pub struct WideFibComponent {
    pub log_fibonacci_size: u32,
    pub log_n_instances: u32,
}

impl WideFibComponent {
    pub fn fill_initial_trace(&self, private_input: Vec<Input>) -> ColumnVec<Vec<BaseField>> {
        let n_instances = 1 << self.log_n_instances;
        assert_eq!(private_input.len(), n_instances);
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

    pub fn log_column_size(&self) -> u32 {
        self.log_n_instances + self.log_fibonacci_size - LOG_N_COLUMNS as u32
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

// Input for the fibonacci claim.
#[derive(Debug, Clone, Copy)]
pub struct Input {
    pub a: BaseField,
    pub b: BaseField,
}
