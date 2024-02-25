pub mod circle;
pub mod line;
// TODO(spapini): Remove pub, when LinePoly moved to the backend as well, and we can move the fold
// function there.
pub mod twiddles;
pub mod utils;

/// Bit-reversed evaluation ordering.
#[derive(Copy, Clone, Debug)]
pub struct BitReversedOrder;

/// Natural evaluation ordering (same order as domain).
#[derive(Copy, Clone, Debug)]
pub struct NaturalOrder;
