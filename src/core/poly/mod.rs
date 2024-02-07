pub mod circle;
pub mod line;
mod utils;

/// Bit-reversed evaluation ordering.
#[derive(Copy, Clone, Debug)]
pub struct BitReversedOrder;

/// Natural evaluation ordering (same order as domain).
#[derive(Copy, Clone, Debug)]
pub struct NaturalOrder;
