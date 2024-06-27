use super::{Backend, MultilinearEvalAtPointIopOps};

pub mod accumulation;
pub mod bit_reverse;
pub mod blake2s;
pub mod circle;
pub mod cm31;
pub mod column;
pub mod fft;
pub mod fri;
pub mod lookups;
pub mod m31;
pub mod qm31;
pub mod quotients;
mod utils;

#[derive(Copy, Clone, Debug)]
pub struct SimdBackend;

impl Backend for SimdBackend {}

// TODO: Remove.
impl MultilinearEvalAtPointIopOps for SimdBackend {}
