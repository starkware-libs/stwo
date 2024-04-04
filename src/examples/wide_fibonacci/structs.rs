use crate::core::air::{Air, Component};
use crate::core::backend::avx512::AVX512Backend;
use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;

/// Component that computes fibonacci numbers over 64 columns.
pub struct WideFibComponent {
    pub log_size: u32,
}

pub struct WideFibAir {
    pub component: WideFibComponent,
}
impl Air<AVX512Backend> for WideFibAir {
    fn components(&self) -> Vec<&dyn Component<AVX512Backend>> {
        vec![&self.component]
    }
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
