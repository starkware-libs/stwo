mod accumulation;
pub mod backend;
mod blake2s;
mod column;
mod field;
mod fri;
mod lookups;
pub mod poly;
pub mod poseidon252;
mod quotient;
mod secure_column;

pub use backend::CudaBackend;

// pub mod examples;

// #[cfg(test)]
// mod tests {
//     use itertools::Itertools;
//     use crate::core::{
//         backend::ColumnOps,
//         circle::SECURE_FIELD_CIRCLE_GEN,
//         fields::{m31::BaseField, FieldOps},
//         poly::circle::{CanonicCoset, PolyOps},
//     };
//     use tracing::{span, Level};

//     use crate::prover::backend::cuda::CudaBackend;
//     use crate::stwo_cuda::base_field_vec::BaseFieldVec;
//     use rand::rngs::SmallRng;
//     use rand::{Rng, SeedableRng};

//     #[test_log::test]
//     fn test_with_log() {
//         let log_size = 28;

//         let size = 1 << log_size;

//         let mut rng = SmallRng::seed_from_u64(0);
//         let data = BaseFieldVec::from_vec((0..size).map(|_| rng.gen()).collect_vec());
//         let mut inverses = BaseFieldVec::new_uninitialized(size);

//         let span = span!(Level::INFO, "batch inverse").entered();
//         <CudaBackend as FieldOps<BaseField>>::batch_inverse(&data, &mut inverses);
//         span.exit();

//         let mut data_for_bit_reverse = data.clone();
//         let span = span!(Level::INFO, "bit_reverse_column").entered();
//         <CudaBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut data_for_bit_reverse);
//         span.exit();

//         let coset = CanonicCoset::new(log_size);

//         let span = span!(Level::INFO, "new_canonical_ordered").entered();
//         let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, data);
//         span.exit();

//         let span = span!(Level::INFO, "precompute_twiddles").entered();
//         let twiddles = CudaBackend::precompute_twiddles(coset.half_coset());
//         span.exit();

//         let span = span!(Level::INFO, "interpolate").entered();
//         let poly = CudaBackend::interpolate(gpu_evaluations, &twiddles);
//         span.exit();

//         let span = span!(Level::INFO, "evaluate").entered();
//         let _ = CudaBackend::evaluate(&poly, coset.circle_domain(), &twiddles);
//         span.exit();

//         let span = span!(Level::INFO, "eval_at_point").entered();
//         let _ = CudaBackend::eval_at_point(&poly, SECURE_FIELD_CIRCLE_GEN);
//         span.exit();
//     }
// }
