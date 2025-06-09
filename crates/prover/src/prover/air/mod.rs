// use crate::core::air::Air;
// use crate::core::backend::Backend;
// use crate::prover::air::component_prover::ComponentProver;

mod accumulation;
pub use accumulation::{AccumulationOps, DomainEvaluationAccumulator};
pub mod component_prover;

// pub trait AirProver<B: Backend>: Air {
//     fn component_provers(&self) -> Vec<&dyn ComponentProver<B>>;
// }
