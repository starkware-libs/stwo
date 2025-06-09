use std::ops::Deref;
use std::{array, mem};

use serde::{Deserialize, Serialize};

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fri::{FriLayerProof, FriProof};
use crate::core::pcs::CommitmentSchemeProof;
use crate::core::vcs::hash::Hash;
use crate::core::vcs::ops::MerkleHasher;
use crate::core::vcs::prover::MerkleDecommitment;

/// Error when the sampled values have an invalid structure.
#[derive(Clone, Copy, Debug)]
pub struct InvalidOodsSampleStructure;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StarkProof<H: MerkleHasher>(pub CommitmentSchemeProof<H>);

impl<H: MerkleHasher> StarkProof<H> {
    /// Extracts the composition trace Out-Of-Domain-Sample evaluation from the mask.
    pub(crate) fn extract_composition_oods_eval(
        &self,
    ) -> Result<SecureField, InvalidOodsSampleStructure> {
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
