use std::ops::Deref;
use std::{array, mem};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info, instrument, span, Level};

use super::air::{Component, ComponentProver, ComponentProvers, Components};
use super::backend::BackendForChannel;
use super::channel::MerkleChannel;
use super::fields::secure_column::SECURE_EXTENSION_DEGREE;
use super::fri::FriVerificationError;
use super::pcs::CommitmentSchemeProof;
use super::vcs::ops::MerkleHasher;
use crate::constraint_framework::PREPROCESSED_TRACE_IDX;
use crate::core::channel::Channel;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fri::{FriLayerProof, FriProof};
use crate::core::pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier};
use crate::core::vcs::hash::Hash;
use crate::core::vcs::prover::MerkleDecommitment;
use crate::core::vcs::verifier::MerkleVerificationError;

#[instrument(skip_all)]
pub fn prove<B: BackendForChannel<MC>, MC: MerkleChannel>(
    components: &[&dyn ComponentProver<B>],
    channel: &mut MC::C,
    mut commitment_scheme: CommitmentSchemeProver<'_, B, MC>,
) -> Result<StarkProof<MC::H>, ProvingError> {
    let n_preprocessed_columns = commitment_scheme.trees[PREPROCESSED_TRACE_IDX]
        .polynomials
        .len();
    let component_provers = ComponentProvers {
        components: components.to_vec(),
        n_preprocessed_columns,
    };
    let trace = commitment_scheme.trace();

    // Evaluate and commit on composition polynomial.
    let random_coeff = channel.draw_felt();

    let span = span!(Level::INFO, "Composition").entered();
    let span1 = span!(Level::INFO, "Generation").entered();
    let composition_poly = component_provers.compute_composition_polynomial(random_coeff, &trace);
    span1.exit();

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_polys(composition_poly.into_coordinate_polys());
    tree_builder.commit(channel);
    span.exit();

    // Draw OODS point.
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);

    // Get mask sample points relative to oods point.
    let mut sample_points = component_provers.components().mask_points(oods_point);

    // Add the composition polynomial mask points.
    sample_points.push(vec![vec![oods_point]; SECURE_EXTENSION_DEGREE]);

    // Prove the trace and composition OODS values, and retrieve them.
    let commitment_scheme_proof = commitment_scheme.prove_values(sample_points, channel);
    let proof = StarkProof(commitment_scheme_proof);
    info!(proof_size_estimate = proof.size_estimate());

    // Evaluate composition polynomial at OODS point and check that it matches the trace OODS
    // values. This is a sanity check.
    if proof.extract_composition_oods_eval().unwrap()
        != component_provers
            .components()
            .eval_composition_polynomial_at_point(oods_point, &proof.sampled_values, random_coeff)
    {
        return Err(ProvingError::ConstraintsNotSatisfied);
    }

    Ok(proof)
}

pub fn verify<MC: MerkleChannel>(
    components: &[&dyn Component],
    channel: &mut MC::C,
    commitment_scheme: &mut CommitmentSchemeVerifier<MC>,
    proof: StarkProof<MC::H>,
) -> Result<(), VerificationError> {
    let n_preprocessed_columns = commitment_scheme.trees[PREPROCESSED_TRACE_IDX]
        .column_log_sizes
        .len();

    let components = Components {
        components: components.to_vec(),
        n_preprocessed_columns,
    };
    let random_coeff = channel.draw_felt();

    // Read composition polynomial commitment.
    commitment_scheme.commit(
        *proof.commitments.last().unwrap(),
        &[components.composition_log_degree_bound(); SECURE_EXTENSION_DEGREE],
        channel,
    );

    // Draw OODS point.
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);

    // Get mask sample points relative to oods point.
    let mut sample_points = components.mask_points(oods_point);
    // Add the composition polynomial mask points.
    sample_points.push(vec![vec![oods_point]; SECURE_EXTENSION_DEGREE]);

    let composition_oods_eval = proof.extract_composition_oods_eval().map_err(|_| {
        VerificationError::InvalidStructure("Unexpected sampled_values structure".to_string())
    })?;

    if composition_oods_eval
        != components.eval_composition_polynomial_at_point(
            oods_point,
            &proof.sampled_values,
            random_coeff,
        )
    {
        return Err(VerificationError::OodsNotMatching);
    }

    commitment_scheme.verify_values(sample_points, proof.0, channel)
}

/// Error when the sampled values have an invalid structure.
#[derive(Clone, Copy, Debug)]
pub struct InvalidOodsSampleStructure;

#[derive(Clone, Copy, Debug, Error)]
pub enum ProvingError {
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
    #[error("Proof of work verification failed.")]
    ProofOfWork,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StarkProof<H: MerkleHasher>(pub CommitmentSchemeProof<H>);

impl<H: MerkleHasher> StarkProof<H> {
    /// Extracts the composition trace Out-Of-Domain-Sample evaluation from the mask.
    fn extract_composition_oods_eval(&self) -> Result<SecureField, InvalidOodsSampleStructure> {
        // TODO(andrew): `[.., composition_mask, _quotients_mask]` when add quotients commitment.
        let [.., composition_mask] = &**self.sampled_values else {
            return Err(InvalidOodsSampleStructure);
        };

        let mut composition_cols = composition_mask.iter();

        let coordinate_evals = array::try_from_fn(|_| {
            let col = &**composition_cols.next().ok_or(InvalidOodsSampleStructure)?;
            let [eval] = col.try_into().map_err(|_| InvalidOodsSampleStructure)?;
            Ok(eval)
        })?;

        // Too many columns.
        if composition_cols.next().is_some() {
            return Err(InvalidOodsSampleStructure);
        }

        Ok(SecureField::from_partial_evals(coordinate_evals))
    }

    /// Returns the estimate size (in bytes) of the proof.
    pub fn size_estimate(&self) -> usize {
        SizeEstimate::size_estimate(self)
    }

    /// Returns size estimates (in bytes) for different parts of the proof.
    pub fn size_breakdown_estimate(&self) -> StarkProofSizeBreakdown {
        let Self(commitment_scheme_proof) = self;

        let CommitmentSchemeProof {
            commitments,
            sampled_values,
            decommitments,
            queried_values,
            proof_of_work: _,
            fri_proof,
            config: _,
        } = commitment_scheme_proof;

        let FriProof {
            first_layer,
            inner_layers,
            last_layer_poly,
        } = fri_proof;

        let mut inner_layers_samples_size = 0;
        let mut inner_layers_hashes_size = 0;

        for FriLayerProof {
            fri_witness,
            decommitment,
            commitment,
        } in inner_layers
        {
            inner_layers_samples_size += fri_witness.size_estimate();
            inner_layers_hashes_size += decommitment.size_estimate() + commitment.size_estimate();
        }

        StarkProofSizeBreakdown {
            oods_samples: sampled_values.size_estimate(),
            queries_values: queried_values.size_estimate(),
            fri_samples: last_layer_poly.size_estimate()
                + inner_layers_samples_size
                + first_layer.fri_witness.size_estimate(),
            fri_decommitments: inner_layers_hashes_size
                + first_layer.decommitment.size_estimate()
                + first_layer.commitment.size_estimate(),
            trace_decommitments: commitments.size_estimate() + decommitments.size_estimate(),
        }
    }
}

impl<H: MerkleHasher> Deref for StarkProof<H> {
    type Target = CommitmentSchemeProof<H>;

    fn deref(&self) -> &CommitmentSchemeProof<H> {
        &self.0
    }
}

/// Size estimate (in bytes) for different parts of the proof.
#[derive(Debug)]
pub struct StarkProofSizeBreakdown {
    pub oods_samples: usize,
    pub queries_values: usize,
    pub fri_samples: usize,
    pub fri_decommitments: usize,
    pub trace_decommitments: usize,
}

trait SizeEstimate {
    fn size_estimate(&self) -> usize;
}

impl<T: SizeEstimate> SizeEstimate for [T] {
    fn size_estimate(&self) -> usize {
        self.iter().map(|v| v.size_estimate()).sum()
    }
}

impl<T: SizeEstimate> SizeEstimate for Vec<T> {
    fn size_estimate(&self) -> usize {
        self.iter().map(|v| v.size_estimate()).sum()
    }
}

impl<H: Hash> SizeEstimate for H {
    fn size_estimate(&self) -> usize {
        mem::size_of::<Self>()
    }
}

impl SizeEstimate for BaseField {
    fn size_estimate(&self) -> usize {
        mem::size_of::<Self>()
    }
}

impl SizeEstimate for SecureField {
    fn size_estimate(&self) -> usize {
        mem::size_of::<Self>()
    }
}

impl<H: MerkleHasher> SizeEstimate for MerkleDecommitment<H> {
    fn size_estimate(&self) -> usize {
        let Self {
            hash_witness,
            column_witness,
        } = self;
        hash_witness.size_estimate() + column_witness.size_estimate()
    }
}

impl<H: MerkleHasher> SizeEstimate for FriLayerProof<H> {
    fn size_estimate(&self) -> usize {
        let Self {
            fri_witness,
            decommitment,
            commitment,
        } = self;
        fri_witness.size_estimate() + decommitment.size_estimate() + commitment.size_estimate()
    }
}

impl<H: MerkleHasher> SizeEstimate for FriProof<H> {
    fn size_estimate(&self) -> usize {
        let Self {
            first_layer,
            inner_layers,
            last_layer_poly,
        } = self;
        first_layer.size_estimate() + inner_layers.size_estimate() + last_layer_poly.size_estimate()
    }
}

impl<H: MerkleHasher> SizeEstimate for CommitmentSchemeProof<H> {
    fn size_estimate(&self) -> usize {
        let Self {
            commitments,
            sampled_values,
            decommitments,
            queried_values,
            proof_of_work,
            fri_proof,
            config,
        } = self;
        commitments.size_estimate()
            + sampled_values.size_estimate()
            + decommitments.size_estimate()
            + queried_values.size_estimate()
            + mem::size_of_val(proof_of_work)
            + fri_proof.size_estimate()
            + mem::size_of_val(config)
    }
}

impl<H: MerkleHasher> SizeEstimate for StarkProof<H> {
    fn size_estimate(&self) -> usize {
        let Self(commitment_scheme_proof) = self;
        commitment_scheme_proof.size_estimate()
    }
}

#[cfg(test)]
mod tests {
    use num_traits::One;

    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
    use crate::core::prover::SizeEstimate;

    #[test]
    fn test_base_field_size_estimate() {
        assert_eq!(BaseField::one().size_estimate(), 4);
    }

    #[test]
    fn test_secure_field_size_estimate() {
        assert_eq!(
            SecureField::one().size_estimate(),
            4 * SECURE_EXTENSION_DEGREE
        );
    }
}
