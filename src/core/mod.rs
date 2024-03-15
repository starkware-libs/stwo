pub mod air;
pub mod backend;
pub mod channel;
pub mod circle;
pub mod commitment_scheme;
pub mod constraints;
pub mod fft;
pub mod fields;
pub mod fri;
pub mod poly;
pub mod proof_of_work;
pub mod prover;
pub mod queries;
pub mod utils;

/// A vector in which each element relates (by index) to a column in the trace.
pub type ColumnVec<T> = Vec<T>;
/// A vector of [ColumnVec]s. Each [ColumnVec] relates (by index) to a component in the air.
#[derive(Debug, Clone)]
pub struct ComponentVec<T>(pub Vec<ColumnVec<T>>);
impl<T> Default for ComponentVec<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}
