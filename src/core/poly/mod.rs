pub mod circle;
pub mod commitment;
pub mod line;
mod utils;

/// Bit-reversed evaluation ordering.
pub struct BitReversedOrder;

/// Natural evaluation ordering (same order as domain).
pub struct NaturalOrder;
