use core::mem;
use core::ops::Deref;

use serde::{Deserialize, Serialize};
use std_shims::Vec;

use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use crate::core::fri::{FriLayerProof, FriProof};
use crate::core::pcs::quotients::{CommitmentSchemeProof, CommitmentSchemeProofAux};
use crate::core::vcs::hash::Hash;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::MerkleDecommitmentLifted;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StarkProof<H: MerkleHasherLifted>(pub CommitmentSchemeProof<H>);

pub struct ExtendedStarkProof<H: MerkleHasherLifted> {
    pub proof: StarkProof<H>,
    pub aux: CommitmentSchemeProofAux<H>,
}

impl<H: MerkleHasherLifted> StarkProof<H> {
    /// Extracts the composition trace Out-Of-Domain-Sample evaluation from the mask.
    pub(crate) fn extract_composition_oods_eval(
        &self,
        oods_point: CirclePoint<SecureField>,
        composition_log_size: u32,
    ) -> Option<SecureField> {
        // TODO(andrew): `[.., composition_mask, _quotients_mask]` when add quotients
        // commitment.
        let [.., left_and_right_composition_mask] = &**self.sampled_values else {
            return None;
        };
        let left_and_right_coordinate_evals: [SecureField; 2 * SECURE_EXTENSION_DEGREE] =
            left_and_right_composition_mask
                .iter()
                .map(|columns| {
                    let &[eval] = &columns[..] else {
                        return None;
                    };
                    Some(eval)
                })
                .collect::<Option<Vec<_>>>()?
                .try_into()
                .ok()?;

        let (left_coordinate_evals, right_coordinate_evals) =
            left_and_right_coordinate_evals.split_at(SECURE_EXTENSION_DEGREE);

        let left_eval = SecureField::from_partial_evals(left_coordinate_evals.try_into().ok()?);
        let right_eval = SecureField::from_partial_evals(right_coordinate_evals.try_into().ok()?);
        let value = left_eval + oods_point.repeated_double(composition_log_size - 2).x * right_eval;
        Some(value)
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

impl<H: MerkleHasherLifted> Deref for StarkProof<H> {
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

impl<H: MerkleHasherLifted> SizeEstimate for MerkleDecommitmentLifted<H> {
    fn size_estimate(&self) -> usize {
        let Self { hash_witness } = self;
        hash_witness.size_estimate()
    }
}

impl<H: MerkleHasherLifted> SizeEstimate for FriLayerProof<H> {
    fn size_estimate(&self) -> usize {
        let Self {
            fri_witness,
            decommitment,
            commitment,
        } = self;
        fri_witness.size_estimate() + decommitment.size_estimate() + commitment.size_estimate()
    }
}

impl<H: MerkleHasherLifted> SizeEstimate for FriProof<H> {
    fn size_estimate(&self) -> usize {
        let Self {
            first_layer,
            inner_layers,
            last_layer_poly,
        } = self;
        first_layer.size_estimate() + inner_layers.size_estimate() + last_layer_poly.size_estimate()
    }
}

impl<H: MerkleHasherLifted> SizeEstimate for CommitmentSchemeProof<H> {
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

impl<H: MerkleHasherLifted> SizeEstimate for StarkProof<H> {
    fn size_estimate(&self) -> usize {
        let Self(commitment_scheme_proof) = self;
        commitment_scheme_proof.size_estimate()
    }
}

#[cfg(test)]
mod tests {
    use num_traits::One;

    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
    use crate::core::proof::SizeEstimate;

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
