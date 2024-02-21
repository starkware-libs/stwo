pub mod air;
pub mod backend;
pub mod channel;
pub mod circle;
pub mod commitment_scheme;
pub mod constraints;
pub mod fft;
pub mod fields;
pub mod fri;
pub mod oods;
pub mod poly;
pub mod proof_of_work;
pub mod queries;
pub mod utils;

/// A vector in which each element relates (by index) to a column in the trace.
pub type ColumnVec<T> = Vec<T>;
/// A vector of [ColumnVec]s. Each [ColumnVec] relates (by index) to a component in the air.
pub type ComponentVec<T> = Vec<ColumnVec<T>>;
