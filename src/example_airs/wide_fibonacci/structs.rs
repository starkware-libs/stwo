use crate::core::fields::m31::BaseField;

/// Component that computes fibonacci numbers over 64 columns.
pub struct WideFibComponent {
    pub log_size: u32,
}

// Input for the fibonacci claim.
#[derive(Debug, Clone, Copy)]
pub struct Input {
    pub a: BaseField,
    pub b: BaseField,
}
