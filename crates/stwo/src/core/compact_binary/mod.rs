pub use stwo_compact_binary::{
    buf_to_array_ctr, strip_expected_tag, strip_expected_version, CompactBinary,
    CompactDeserializeError, CompactSerializeError, ZippedCompactBinary,
};
pub use stwo_compact_binary_derive::CompactBinary;

use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::{BaseField, P};
use crate::core::fields::qm31::SecureField;
use crate::core::fri::{FriConfig, FriLayerProof, FriProof};
use crate::core::pcs::quotients::CommitmentSchemeProof;
use crate::core::pcs::{PcsConfig, TreeVec};
use crate::core::poly::line::LinePoly;
use crate::core::proof::StarkProof;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs::verifier::MerkleDecommitment;
use crate::core::vcs::MerkleHasher;
use crate::core::ColumnVec;

#[cfg(test)]
mod tests;

impl CompactBinary for BaseField {
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        output.extend_from_slice(&self.0.to_be_bytes());
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let (input, u32_value) = buf_to_array_ctr(input, |v| u32::from_be_bytes(*v))
            .ok_or(CompactDeserializeError::DecodeError)?;

        if u32_value > P {
            Err(CompactDeserializeError::DecodeError)
        } else {
            Ok((input, BaseField::from_u32_unchecked(u32_value)))
        }
    }
}

impl CompactBinary for CM31 {
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        self.0.compact_serialize(output)?;
        self.1.compact_serialize(output)?;
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let (input, u32_value_0) = BaseField::compact_deserialize(input)?;
        let (input, u32_value_1) = BaseField::compact_deserialize(input)?;
        Ok((input, CM31::from_m31(u32_value_0, u32_value_1)))
    }
}

impl CompactBinary for SecureField {
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        self.0.compact_serialize(output)?;
        self.1.compact_serialize(output)?;
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let (input, m31_value_0) = BaseField::compact_deserialize(input)?;
        let (input, m31_value_1) = BaseField::compact_deserialize(input)?;
        let (input, m31_value_2) = BaseField::compact_deserialize(input)?;
        let (input, m31_value_3) = BaseField::compact_deserialize(input)?;
        Ok((
            input,
            SecureField::from_m31(m31_value_0, m31_value_1, m31_value_2, m31_value_3),
        ))
    }
}

impl<H: MerkleHasher> CompactBinary for MerkleDecommitment<H>
where
    H::Hash: CompactBinary,
{
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        let Self {
            hash_witness,
            column_witness,
        } = self;
        let version = 0;
        let to_serialize: Vec<(usize, &dyn CompactBinary)> =
            vec![(0, hash_witness), (1, column_witness)];
        u32::compact_serialize(&version, output)?;
        for (tag, value) in to_serialize {
            usize::compact_serialize(&tag, output)?;
            value.compact_serialize(output)?;
        }
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let input = strip_expected_version(input, 0)?;
        let input = strip_expected_tag(input, 0)?;
        let (input, hash_witness) = Vec::<H::Hash>::compact_deserialize(input)?;
        let input = strip_expected_tag(input, 1)?;
        let (input, column_witness) = Vec::<BaseField>::compact_deserialize(input)?;
        Ok((
            input,
            MerkleDecommitment {
                hash_witness,
                column_witness,
            },
        ))
    }
}

impl CompactBinary for LinePoly {
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        let coeffs = self.clone().into_ordered_coefficients();
        coeffs.len().compact_serialize(output)?;
        for coeff in &coeffs {
            coeff.compact_serialize(output)?;
        }
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let (mut input, len) = usize::compact_deserialize(input)?;
        let mut coeffs = Vec::with_capacity(len);
        for _ in 0..len {
            let (updated_input, coeff) = SecureField::compact_deserialize(input)?;
            input = updated_input;
            coeffs.push(coeff);
        }
        Ok((input, LinePoly::from_ordered_coefficients(coeffs)))
    }
}

impl<H: MerkleHasher> CompactBinary for FriLayerProof<H>
where
    H::Hash: CompactBinary,
{
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        let Self {
            fri_witness,
            decommitment,
            commitment,
        } = self;
        let version = 0;
        let to_serialize: Vec<(usize, &dyn CompactBinary)> =
            vec![(0, fri_witness), (1, decommitment), (2, commitment)];
        u32::compact_serialize(&version, output)?;
        for (tag, value) in to_serialize {
            usize::compact_serialize(&tag, output)?;
            value.compact_serialize(output)?;
        }
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let input = strip_expected_version(input, 0)?;
        let input = strip_expected_tag(input, 0)?;
        let (input, fri_witness) = Vec::<SecureField>::compact_deserialize(input)?;
        let input = strip_expected_tag(input, 1)?;
        let (input, decommitment) = MerkleDecommitment::compact_deserialize(input)?;
        let input = strip_expected_tag(input, 2)?;
        let (input, commitment) = H::Hash::compact_deserialize(input)?;
        Ok((
            input,
            FriLayerProof {
                fri_witness,
                decommitment,
                commitment,
            },
        ))
    }
}

impl<H: MerkleHasher> CompactBinary for FriProof<H>
where
    H::Hash: CompactBinary,
{
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        let Self {
            first_layer,
            inner_layers,
            last_layer_poly,
        } = self;
        let version = 0;
        let to_serialize: Vec<(usize, &dyn CompactBinary)> =
            vec![(0, first_layer), (1, inner_layers), (2, last_layer_poly)];
        u32::compact_serialize(&version, output)?;
        for (tag, value) in to_serialize {
            usize::compact_serialize(&tag, output)?;
            value.compact_serialize(output)?;
        }
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let input = strip_expected_version(input, 0)?;
        let input = strip_expected_tag(input, 0)?;
        let (input, first_layer) = FriLayerProof::compact_deserialize(input)?;
        let input = strip_expected_tag(input, 1)?;
        let (input, inner_layers) = Vec::<FriLayerProof<H>>::compact_deserialize(input)?;
        let input = strip_expected_tag(input, 2)?;
        let (input, last_layer_poly) = LinePoly::compact_deserialize(input)?;
        Ok((
            input,
            FriProof {
                first_layer,
                inner_layers,
                last_layer_poly,
            },
        ))
    }
}

impl CompactBinary for FriConfig {
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        let Self {
            log_blowup_factor,
            log_last_layer_degree_bound,
            n_queries,
        } = self;
        let version = 0;
        let to_serialize: Vec<(usize, &dyn CompactBinary)> = vec![
            (0, log_blowup_factor),
            (1, log_last_layer_degree_bound),
            (2, n_queries),
        ];
        u32::compact_serialize(&version, output)?;
        for (tag, value) in to_serialize {
            usize::compact_serialize(&tag, output)?;
            value.compact_serialize(output)?;
        }
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let input = strip_expected_version(input, 0)?;
        let input = strip_expected_tag(input, 0)?;
        let (input, log_blowup_factor) = u32::compact_deserialize(input)?;
        let input = strip_expected_tag(input, 1)?;
        let (input, log_last_layer_degree_bound) = u32::compact_deserialize(input)?;
        let input = strip_expected_tag(input, 2)?;
        let (input, n_queries) = usize::compact_deserialize(input)?;
        Ok((
            input,
            FriConfig {
                log_blowup_factor,
                log_last_layer_degree_bound,
                n_queries,
            },
        ))
    }
}

impl CompactBinary for PcsConfig {
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        let Self {
            pow_bits,
            fri_config,
        } = self;
        let version = 0;
        let to_serialize: Vec<(usize, &dyn CompactBinary)> = vec![(0, pow_bits), (1, fri_config)];
        u32::compact_serialize(&version, output)?;
        for (tag, value) in to_serialize {
            usize::compact_serialize(&tag, output)?;
            value.compact_serialize(output)?;
        }
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let input = strip_expected_version(input, 0)?;
        let input = strip_expected_tag(input, 0)?;
        let (input, pow_bits) = u32::compact_deserialize(input)?;
        let input = strip_expected_tag(input, 1)?;
        let (input, fri_config) = FriConfig::compact_deserialize(input)?;
        Ok((
            input,
            PcsConfig {
                pow_bits,
                fri_config,
            },
        ))
    }
}

impl<H: MerkleHasher> CompactBinary for CommitmentSchemeProof<H>
where
    H::Hash: CompactBinary,
{
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        let Self {
            config,
            commitments,
            sampled_values,
            decommitments,
            queried_values,
            proof_of_work,
            fri_proof,
        } = self;
        let version = 0;
        let to_serialize: Vec<(usize, &dyn CompactBinary)> = vec![
            (0, config),
            (1, &commitments.0),
            (2, &sampled_values.0),
            (3, &decommitments.0),
            (4, &queried_values.0),
            (5, proof_of_work),
            (6, fri_proof),
        ];
        u32::compact_serialize(&version, output)?;
        for (tag, value) in to_serialize {
            usize::compact_serialize(&tag, output)?;
            value.compact_serialize(output)?;
        }
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let input = strip_expected_version(input, 0)?;
        let input = strip_expected_tag(input, 0)?;
        let (input, config) = PcsConfig::compact_deserialize(input)?;
        let input = strip_expected_tag(input, 1)?;
        let (input, commitments) = Vec::<H::Hash>::compact_deserialize(input)?;
        let input = strip_expected_tag(input, 2)?;
        let (input, sampled_values) =
            Vec::<ColumnVec<Vec<SecureField>>>::compact_deserialize(input)?;
        let input = strip_expected_tag(input, 3)?;
        let (input, decommitments) = Vec::<MerkleDecommitment<H>>::compact_deserialize(input)?;
        let input = strip_expected_tag(input, 4)?;
        let (input, queried_values) = Vec::<Vec<BaseField>>::compact_deserialize(input)?;
        let input = strip_expected_tag(input, 5)?;
        let (input, proof_of_work) = u64::compact_deserialize(input)?;
        let input = strip_expected_tag(input, 6)?;
        let (input, fri_proof) = FriProof::compact_deserialize(input)?;
        Ok((
            input,
            CommitmentSchemeProof {
                config,
                commitments: TreeVec::new(commitments),
                sampled_values: TreeVec::new(sampled_values),
                decommitments: TreeVec::new(decommitments),
                queried_values: TreeVec::new(queried_values),
                proof_of_work,
                fri_proof,
            },
        ))
    }
}

impl<H: MerkleHasher> CompactBinary for StarkProof<H>
where
    H::Hash: CompactBinary,
{
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        let Self(commitment_scheme_proof) = self;
        let version = 0;
        let to_serialize: Vec<(usize, &dyn CompactBinary)> = vec![(0, commitment_scheme_proof)];
        u32::compact_serialize(&version, output)?;
        for (tag, value) in to_serialize {
            usize::compact_serialize(&tag, output)?;
            value.compact_serialize(output)?;
        }
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let input = strip_expected_version(input, 0)?;
        let input = strip_expected_tag(input, 0)?;
        let (input, commitment_scheme_proof) = CommitmentSchemeProof::compact_deserialize(input)?;
        Ok((input, StarkProof(commitment_scheme_proof)))
    }
}

impl CompactBinary for Blake2sHash {
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        output.extend_from_slice(&self.0);
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let (input, hash) = buf_to_array_ctr(input, |v| Blake2sHash(*v))
            .ok_or(CompactDeserializeError::DecodeError)?;
        Ok((input, hash))
    }
}
