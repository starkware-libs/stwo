use itertools::Itertools;
use thiserror::Error;
use tracing::{span, Level};

use super::air::AirProver;
use super::backend::Backend;
use super::fields::secure_column::SECURE_EXTENSION_DEGREE;
use super::fri::FriVerificationError;
use super::pcs::{CommitmentSchemeProof, TreeVec};
use super::poly::circle::MAX_CIRCLE_DOMAIN_LOG_SIZE;
use super::proof_of_work::ProofOfWorkVerificationError;
use super::{ColumnVec, InteractionElements, LookupValues};
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
    pub lookup_values: LookupValues,
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
    channel: &mut Channel,
    interaction_elements: &InteractionElements,
    commitment_scheme: &mut CommitmentSchemeProver<'_, B>,
) -> Result<StarkProof, ProvingError> {
    let component_traces = air.component_traces(&commitment_scheme.trees);
    let lookup_values = air.lookup_values(&component_traces);

    // Evaluate and commit on composition polynomial.
    let random_coeff = channel.draw_felt();

    let span = span!(Level::INFO, "Composition").entered();
    let span1 = span!(Level::INFO, "Generation").entered();
    let composition_polynomial_poly = air.compute_composition_polynomial(
        random_coeff,
        &component_traces,
        interaction_elements,
        &lookup_values,
    );
    span1.exit();

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_polys(composition_polynomial_poly.to_vec());
    tree_builder.commit(channel);
    span.exit();

    // Draw OODS point.
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);

    // Get mask sample points relative to oods point.
    let sample_points = air.mask_points(oods_point);

    // Prove the trace and composition OODS values, and retrieve them.
    let commitment_scheme_proof = commitment_scheme.prove_values(sample_points, channel);

    // Evaluate composition polynomial at OODS point and check that it matches the trace OODS
    // values. This is a sanity check.
    // TODO(spapini): Save clone.
    let (trace_oods_values, composition_oods_value) =
        sampled_values_to_mask(air, &commitment_scheme_proof.sampled_values).unwrap();

    if composition_oods_value
        != air.eval_composition_polynomial_at_point(
            oods_point,
            &trace_oods_values,
            random_coeff,
            interaction_elements,
            &lookup_values,
        )
    {
        return Err(ProvingError::ConstraintsNotSatisfied);
    }

    Ok(StarkProof {
        commitments: commitment_scheme.roots(),
        lookup_values,
        commitment_scheme_proof,
    })
}

pub fn verify(
    air: &impl Air,
    channel: &mut Blake2sChannel,
    interaction_elements: &InteractionElements,
    commitment_scheme: &mut CommitmentSchemeVerifier,
    proof: StarkProof,
) -> Result<(), VerificationError> {
    let random_coeff = channel.draw_felt();

    // Read composition polynomial commitment.
    commitment_scheme.commit(
        *proof.commitments.last().unwrap(),
        &[air.composition_log_degree_bound(); SECURE_EXTENSION_DEGREE],
        channel,
    );

    // Draw OODS point.
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);

    // Get mask sample points relative to oods point.
    let sample_points = air.mask_points(oods_point);

    // TODO(spapini): Save clone.
    let (trace_oods_values, composition_oods_value) = sampled_values_to_mask(
        air,
        &proof.commitment_scheme_proof.sampled_values,
    )
    .map_err(|_| {
        VerificationError::InvalidStructure("Unexpected sampled_values structure".to_string())
    })?;

    if composition_oods_value
        != air.eval_composition_polynomial_at_point(
            oods_point,
            &trace_oods_values,
            random_coeff,
            interaction_elements,
            &proof.lookup_values,
        )
    {
        return Err(VerificationError::OodsNotMatching);
    }

    commitment_scheme.verify_values(sample_points, proof.commitment_scheme_proof, channel)
}

#[allow(clippy::type_complexity)]
/// Structures the tree-wise sampled values into component-wise OODS values and a composition
/// polynomial OODS value.
fn sampled_values_to_mask(
    air: &impl Air,
    sampled_values: &TreeVec<ColumnVec<Vec<SecureField>>>,
) -> Result<(Vec<TreeVec<Vec<Vec<SecureField>>>>, SecureField), InvalidOodsSampleStructure> {
    let mut sampled_values = sampled_values.as_ref();
    let composition_values = sampled_values.pop().ok_or(InvalidOodsSampleStructure)?;

    let mut sample_iters = sampled_values.map(|tree_value| tree_value.iter());
    let trace_oods_values = air
        .components()
        .iter()
        .map(|component| {
            component
                .mask_points(CirclePoint::zero())
                .zip(sample_iters.as_mut())
                .map(|(mask_per_tree, tree_iter)| {
                    tree_iter.take(mask_per_tree.len()).cloned().collect_vec()
                })
        })
        .collect_vec();

    let composition_oods_value = SecureField::from_partial_evals(
        composition_values
            .iter()
            .flatten()
            .cloned()
            .collect_vec()
            .try_into()
            .map_err(|_| InvalidOodsSampleStructure)?,
    );

    Ok((trace_oods_values, composition_oods_value))
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
    #[error("{0} lookup values do not match.")]
    InvalidLookup(String),
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
