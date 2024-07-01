use itertools::Itertools;
use thiserror::Error;
use tracing::{span, Level};

use super::air::AirProver;
use super::backend::Backend;
use super::fields::secure_column::SECURE_EXTENSION_DEGREE;
use super::fri::FriVerificationError;
use super::pcs::{CommitmentSchemeProof, TreeVec};
use super::poly::circle::{SecureCirclePoly, MAX_CIRCLE_DOMAIN_LOG_SIZE};
use super::poly::twiddles::TwiddleTree;
use super::proof_of_work::ProofOfWorkVerificationError;
use super::ColumnVec;
use crate::core::air::{Air, AirExt, AirProverExt};
use crate::core::backend::CpuBackend;
use crate::core::channel::{Blake2sChannel, Channel as ChannelTrait};
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier};
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::vcs::blake2_hash::Blake2sHasher;
use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use crate::core::vcs::hasher::Hasher;
use crate::core::vcs::ops::MerkleOps;
use crate::core::vcs::verifier::MerkleVerificationError;

type Channel = Blake2sChannel;
type ChannelHasher = Blake2sHasher;
type MerkleHasher = Blake2sMerkleHasher;

pub const LOG_BLOWUP_FACTOR: u32 = 1;
pub const LOG_LAST_LAYER_DEGREE_BOUND: u32 = 0;
pub const PROOF_OF_WORK_BITS: u32 = 12;
pub const N_QUERIES: usize = 3;

#[derive(Debug)]
pub struct StarkProof {
    pub commitments: TreeVec<<ChannelHasher as Hasher>::Hash>,
    pub commitment_scheme_proof: CommitmentSchemeProof,
}

#[derive(Debug)]
pub struct AdditionalProofData {
    pub composition_polynomial_oods_value: SecureField,
    pub composition_polynomial_random_coeff: SecureField,
    pub oods_point: CirclePoint<SecureField>,
    pub oods_quotients: Vec<CircleEvaluation<CpuBackend, SecureField, BitReversedOrder>>,
}

pub fn prove<B: Backend + MerkleOps<MerkleHasher>>(
    air: &impl AirProver<B>,
    commitment_scheme: &mut CommitmentSchemeProver<B>,
    channel: &mut Channel,
    twiddles: &TwiddleTree<B>,
) -> Result<StarkProof, ProvingError> {
    // Check that the composition polynomial is not too big.
    // TODO(AlonH): Get traces log degree bounds from trace writer.
    let composition_polynomial_log_degree_bound = air.composition_log_degree_bound();
    if composition_polynomial_log_degree_bound + LOG_BLOWUP_FACTOR > MAX_CIRCLE_DOMAIN_LOG_SIZE {
        return Err(ProvingError::MaxCompositionDegreeExceeded {
            degree: composition_polynomial_log_degree_bound,
        });
    }

    // Evaluate and commit on composition polynomial.
    let composition_random_coeff = channel.draw_felt();

    let span = span!(Level::INFO, "Composition generation").entered();
    let composition_polynomial_poly = air.compute_composition_polynomial(
        composition_random_coeff,
        &air.component_traces(&commitment_scheme.trees),
    );
    span.exit();

    let span = span!(Level::INFO, "Composition commitment").entered();
    commitment_scheme.commit(composition_polynomial_poly.to_vec(), channel, twiddles);
    span.exit();

    // Draw OODS point.
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);

    // Get mask sample points relative to oods point.
    let mut sample_points = air.mask_points(oods_point);
    sample_points.push(vec![vec![oods_point]; SECURE_EXTENSION_DEGREE]);

    // Prove the trace and composition OODS values, and retrieve them.
    let commitment_scheme_proof = commitment_scheme.prove_values(sample_points, channel, twiddles);

    // Evaluate composition polynomial at OODS point and check that it matches the trace OODS
    // values. This is a sanity check.
    // TODO(spapini): Save clone.
    let (trace_oods_values, composition_oods_value) =
        sampled_values_to_mask(commitment_scheme_proof.sampled_values.clone()).unwrap();

    if composition_oods_value
        != air.eval_composition_polynomial_at_point(
            oods_point,
            trace_oods_values,
            composition_random_coeff,
        )
    {
        return Err(ProvingError::ConstraintsNotSatisfied);
    }

    Ok(StarkProof {
        commitments: commitment_scheme.roots(),
        commitment_scheme_proof,
    })
}

pub fn verify(
    proof: StarkProof,
    air: &impl Air,
    commitment_scheme: &mut CommitmentSchemeVerifier,
    channel: &mut Channel,
) -> Result<(), VerificationError> {
    let composition_random_coeff = channel.draw_felt();

    // Read composition polynomial commitment.
    commitment_scheme.commit(
        *proof.commitments.last().unwrap(),
        &[air.composition_log_degree_bound(); SECURE_EXTENSION_DEGREE],
        channel,
    );

    // Draw OODS point.
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);

    // Get mask sample points relative to oods point.
    let mut sample_points = air.mask_points(oods_point);
    sample_points.push(vec![vec![oods_point]; SECURE_EXTENSION_DEGREE]);

    // TODO(spapini): Save clone.
    let (trace_oods_values, composition_oods_value) = sampled_values_to_mask(
        proof.commitment_scheme_proof.sampled_values.clone(),
    )
    .map_err(|_| {
        VerificationError::InvalidStructure("Unexpected sampled_values structure".to_string())
    })?;

    if composition_oods_value
        != air.eval_composition_polynomial_at_point(
            oods_point,
            trace_oods_values,
            composition_random_coeff,
        )
    {
        return Err(VerificationError::OodsNotMatching);
    }

    commitment_scheme.verify_values(sample_points, proof.commitment_scheme_proof, channel)
}

/// Structures the tree-wise sampled values into component-wise OODS values and a composition
/// polynomial OODS value.
#[allow(clippy::type_complexity)]
fn sampled_values_to_mask(
    mut sampled_values: TreeVec<ColumnVec<Vec<SecureField>>>,
) -> Result<(TreeVec<ColumnVec<Vec<SecureField>>>, SecureField), InvalidOodsSampleStructure> {
    let composition_partial_sampled_values =
        sampled_values.pop().ok_or(InvalidOodsSampleStructure)?;
    let composition_oods_value = SecureCirclePoly::<CpuBackend>::eval_from_partial_evals(
        composition_partial_sampled_values
            .iter()
            .flatten()
            .cloned()
            .collect_vec()
            .try_into()
            .map_err(|_| InvalidOodsSampleStructure)?,
    );

    Ok((sampled_values, composition_oods_value))
}

/// Error when the sampled values have an invalid structure.
#[derive(Clone, Copy, Debug)]
pub struct InvalidOodsSampleStructure;

#[derive(Clone, Copy, Debug, Error)]
pub enum ProvingError {
    #[error(
        "Trace column {trace_index} log degree bound ({degree}) exceeded max log degree ({}).",
        MAX_CIRCLE_DOMAIN_LOG_SIZE - LOG_BLOWUP_FACTOR
    )]
    MaxTraceDegreeExceeded { trace_index: usize, degree: u32 },
    #[error(
        "Composition polynomial log degree bound ({degree}) exceeded max log degree ({}).",
        MAX_CIRCLE_DOMAIN_LOG_SIZE - LOG_BLOWUP_FACTOR
    )]
    MaxCompositionDegreeExceeded { degree: u32 },
    #[error("Constraints not satisfied.")]
    ConstraintsNotSatisfied,
}

#[derive(Clone, Debug, Error)]
pub enum VerificationError {
    #[error("Proof has invalid structure: {0}.")]
    InvalidStructure(String),
    #[error(transparent)]
    Merkle(#[from] MerkleVerificationError),
    #[error(
        "The composition polynomial OODS value does not match the trace OODS values
    (DEEP-ALI failure)."
    )]
    OodsNotMatching,
    #[error(transparent)]
    Fri(#[from] FriVerificationError),
    #[error(transparent)]
    ProofOfWork(#[from] ProofOfWorkVerificationError),
}

// #[cfg(test)]
// mod tests {
//     use num_traits::Zero;

//     use crate::core::air::accumulation::{DomainEvaluationAccumulator,
// PointEvaluationAccumulator};     use crate::core::air::{
//         Air, AirProver, AirTraceVerifier, AirTraceWriter, Component, ComponentProver,
//         ComponentTrace, ComponentTraceWriter,
//     };
//     use crate::core::backend::cpu::CpuCircleEvaluation;
//     use crate::core::backend::CpuBackend;
//     use crate::core::channel::Blake2sChannel;
//     use crate::core::circle::{CirclePoint, CirclePointIndex, Coset};
//     use crate::core::fields::m31::BaseField;
//     use crate::core::fields::qm31::SecureField;
//     use crate::core::pcs::{ChunkLocation, TreeVec};
//     use crate::core::poly::circle::{
//         CanonicCoset, CircleDomain, CircleEvaluation, PolyOps, MAX_CIRCLE_DOMAIN_LOG_SIZE,
//     };
//     use crate::core::poly::BitReversedOrder;
//     use crate::core::prover::{prove, ProvingError};
//     use crate::core::{ColumnVec, InteractionElements};
//     use crate::qm31;

//     struct TestAir<C: ComponentProver<CpuBackend>> {
//         component: C,
//     }

//     impl Air for TestAir<TestComponent> {
//         fn components(&self) -> Vec<&dyn Component> {
//             vec![&self.component]
//         }
//     }

//     impl AirTraceVerifier for TestAir<TestComponent> {
//         fn interaction_elements(&self, _channel: &mut Blake2sChannel) -> InteractionElements {
//             InteractionElements::default()
//         }
//     }

//     impl AirTraceWriter<CpuBackend> for TestAir<TestComponent> {
//         fn interact(
//             &self,
//             _trace: &ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
//             _elements: &InteractionElements,
//         ) -> Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> { vec![]
//         }

//         fn to_air_prover(&self) -> &impl AirProver<CpuBackend> {
//             self
//         }
//     }

//     impl AirProver<CpuBackend> for TestAir<TestComponent> {
//         fn prover_components(&self) -> Vec<&dyn ComponentProver<CpuBackend>> {
//             vec![&self.component]
//         }
//     }

//     struct TestComponent {
//         log_size: u32,
//         max_constraint_log_degree_bound: u32,
//     }

//     impl Component for TestComponent {
//         fn n_constraints(&self) -> usize {
//             0
//         }

//         fn max_constraint_log_degree_bound(&self) -> u32 {
//             self.max_constraint_log_degree_bound
//         }

//         fn mask_points(
//             &self,
//             point: CirclePoint<SecureField>,
//         ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
//           TreeVec::new(vec![vec![vec![point]], vec![]])
//         }

//         fn interaction_element_ids(&self) -> Vec<String> {
//             vec![]
//         }

//         fn evaluate_constraint_quotients_at_point(
//             &self,
//             _point: CirclePoint<SecureField>,
//             _mask: &crate::core::ColumnVec<Vec<SecureField>>,
//             evaluation_accumulator: &mut PointEvaluationAccumulator,
//             _interaction_elements: &InteractionElements,
//         ) { evaluation_accumulator.accumulate(qm31!(0, 0, 0, 1))
//         }

//         fn chunk_locations(&self) -> Vec<ChunkLocation> {
//             vec![ChunkLocation {
//                 tree_index: 0,
//                 col_start: 0,
//                 col_end: 1,
//             }]
//         }
//     }

//     // #[test]
//     // fn test_composition_polynomial_too_big() {
//     //     const COMPOSITION_POLYNOMIAL_DEGREE: u32 = MAX_CIRCLE_DOMAIN_LOG_SIZE;
//     //     const LOG_DOMAIN_SIZE: u32 = 5;
//     //     let air = TestAir {
//     //         component: TestComponent {
//     //             log_size: LOG_DOMAIN_SIZE,
//     //             max_constraint_log_degree_bound: COMPOSITION_POLYNOMIAL_DEGREE,
//     //         },
//     //     };
//     //     let domain = CircleDomain::new(Coset::new(
//     //         CirclePointIndex::generator(),
//     //         LOG_DOMAIN_SIZE - 1,
//     //     ));
//     //     let values = vec![BaseField::zero(); 1 << LOG_DOMAIN_SIZE];
//     //     let trace = vec![CpuCircleEvaluation::new(domain, values)];

//     //     let proof_error = prove(&air, &mut test_channel(), trace).unwrap_err();
//     //     assert!(matches!(
//     //         proof_error,
//     //         ProvingError::MaxCompositionDegreeExceeded {
//     //             degree: COMPOSITION_POLYNOMIAL_DEGREE
//     //         }
//     //     ));
//     // }

//     // #[test]
//     // fn test_constraints_not_satisfied() {
//     //     const LOG_DOMAIN_SIZE: u32 = 5;
//     //     let air = TestAir {
//     //         component: TestComponent {
//     //             log_size: LOG_DOMAIN_SIZE,
//     //             max_constraint_log_degree_bound: LOG_DOMAIN_SIZE + 1,
//     //         },
//     //     };
//     //     let domain = CanonicCoset::new(LOG_DOMAIN_SIZE).circle_domain();
//     //     let values = vec![BaseField::zero(); 1 << LOG_DOMAIN_SIZE];
//     //     let trace = vec![CpuCircleEvaluation::new(domain, values)];

//     //     let proof = prove(&air, &mut test_channel(), trace).unwrap_err();
//     //     assert!(matches!(proof, ProvingError::ConstraintsNotSatisfied));
//     // }
// }
