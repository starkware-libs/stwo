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

#[cfg(test)]
mod tests {
    use num_traits::One;
    use stwo_compact_binary::CompactBinary;

    use super::*;
    use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;

    #[test]
    fn test_base_field_serialization() {
        let m1 = BaseField::from_u32_unchecked(42);
        let m2 = BaseField::from_u32_unchecked(100);
        let m3 = BaseField::from_u32_unchecked(0);
        let m4 = BaseField::from_u32_unchecked(31);
        let field = SecureField::from_m31(m1, m2, m3, m4);
        let mut output = Vec::new();
        field.compact_serialize(&mut output).unwrap();
        let (remaining, deserialized) = SecureField::compact_deserialize(&output).unwrap();
        assert!(remaining.is_empty());
        assert_eq!(deserialized, field);
    }

    #[test]
    fn test_pcs_config_serialization() {
        let pcs_config = PcsConfig {
            pow_bits: 5,
            fri_config: FriConfig::new(0, 1, 3),
        };

        let mut output = Vec::new();
        pcs_config.compact_serialize(&mut output).unwrap();
        let (remaining, deserialized) = PcsConfig::compact_deserialize(&output).unwrap();
        assert!(remaining.is_empty());
        assert_eq_pcs_config(&pcs_config, &deserialized);
    }

    #[test]
    fn test_proof_serialization() {
        let stark_proof: StarkProof<Blake2sMerkleHasher> = StarkProof(CommitmentSchemeProof {
            config: PcsConfig::default(),
            commitments: TreeVec::new(vec![Blake2sHash([0; 32])]),
            sampled_values: TreeVec::new(vec![]),
            decommitments: TreeVec::new(vec![MerkleDecommitment {
                hash_witness: vec![Blake2sHash([0; 32])],
                column_witness: vec![BaseField::one()],
            }]),
            queried_values: TreeVec::new(vec![vec![BaseField::one()]]),
            proof_of_work: 42,
            fri_proof: FriProof {
                first_layer: FriLayerProof {
                    fri_witness: vec![SecureField::one()],
                    decommitment: MerkleDecommitment {
                        hash_witness: vec![Blake2sHash([0; 32])],
                        column_witness: vec![BaseField::one()],
                    },
                    commitment: Blake2sHash([0; 32]),
                },
                inner_layers: vec![],
                last_layer_poly: LinePoly::from_ordered_coefficients(vec![SecureField::one()]),
            },
        });

        let mut output = Vec::new();
        stark_proof.compact_serialize(&mut output).unwrap();
        let (remaining, deserialized) =
            StarkProof::<Blake2sMerkleHasher>::compact_deserialize(&output).unwrap();
        assert!(remaining.is_empty());

        assert_eq_pcs_config(&deserialized.0.config, &stark_proof.0.config);

        assert_eq!(deserialized.0.commitments.0, stark_proof.0.commitments.0);
        assert_eq!(
            deserialized.0.sampled_values.0,
            stark_proof.0.sampled_values.0
        );
        assert_eq!(
            deserialized.0.decommitments.0,
            stark_proof.0.decommitments.0
        );
        assert_eq!(
            deserialized.0.queried_values.0,
            stark_proof.0.queried_values.0
        );
        assert_eq!(deserialized.0.proof_of_work, stark_proof.0.proof_of_work);

        assert_eq_fri_proof(&deserialized.0.fri_proof, &stark_proof.0.fri_proof);
    }

    fn assert_eq_pcs_config(pcs_config1: &PcsConfig, pcs_config2: &PcsConfig) {
        assert_eq!(pcs_config1.pow_bits, pcs_config2.pow_bits);
        assert_eq!(
            pcs_config1.fri_config.log_blowup_factor,
            pcs_config2.fri_config.log_blowup_factor
        );
        assert_eq!(
            pcs_config1.fri_config.log_last_layer_degree_bound,
            pcs_config2.fri_config.log_last_layer_degree_bound
        );
        assert_eq!(
            pcs_config1.fri_config.n_queries,
            pcs_config2.fri_config.n_queries
        );
    }

    fn assert_eq_fri_proof(
        fri_proof1: &FriProof<Blake2sMerkleHasher>,
        fri_proof2: &FriProof<Blake2sMerkleHasher>,
    ) {
        assert_eq_fri_layer(&fri_proof1.first_layer, &fri_proof2.first_layer);

        assert_eq!(fri_proof1.inner_layers.len(), fri_proof2.inner_layers.len());
        for (layer1, layer2) in fri_proof1
            .inner_layers
            .iter()
            .zip(fri_proof2.inner_layers.iter())
        {
            assert_eq_fri_layer(layer1, layer2);
        }
        assert_eq!(fri_proof1.last_layer_poly, fri_proof2.last_layer_poly);
    }

    fn assert_eq_fri_layer(
        fri_layer1: &FriLayerProof<Blake2sMerkleHasher>,
        fri_layer2: &FriLayerProof<Blake2sMerkleHasher>,
    ) {
        assert_eq!(fri_layer1.fri_witness, fri_layer2.fri_witness);
        assert_eq!(fri_layer1.decommitment, fri_layer2.decommitment);
        assert_eq!(fri_layer1.commitment, fri_layer2.commitment);
    }

    // The tests in this module are for the `CompactBinary` derive macro:
    // - a base test
    // - a test with a zipped field
    // - a test with a generic type `H: MerkleHasher` that requires
    // the bound `H:Hash: CompactBinary` to be implemented
    mod tests_derive {
        use crate::core::compact_binary::{
            CompactBinary, CompactDeserializeError, CompactSerializeError,
        };
        use crate::core::fields::m31::BaseField;
        use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
        use crate::core::vcs::MerkleHasher;
        use crate::{self as stwo};

        // We first define three structs `TestStruct`, `TestStructZipped` and `TestStructGeneric` to
        // test the derive macro
        #[derive(CompactBinary, Debug, PartialEq, Eq)]
        struct TestStruct {
            base: [BaseField; 64],
        }

        #[derive(CompactBinary, Debug, PartialEq, Eq)]
        struct TestStructZipped {
            #[zipped]
            base: [BaseField; 64],
        }

        #[derive(CompactBinary, Debug, PartialEq, Eq)]
        struct TestStructGeneric<H: MerkleHasher> {
            hashed_base: HashedBaseField<H>,
        }

        #[derive(Debug, PartialEq, Eq)]
        struct HashedBaseField<H: MerkleHasher> {
            base: BaseField,
            hash: H::Hash,
        }

        impl<H: MerkleHasher> HashedBaseField<H> {
            fn new(base: BaseField) -> Self {
                let hash = H::hash_node(None, &[base]);
                Self { base, hash }
            }
        }

        impl<H: MerkleHasher> CompactBinary for HashedBaseField<H>
        where
            H::Hash: CompactBinary,
        {
            fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
                self.base.compact_serialize(output)?;
                self.hash.compact_serialize(output)?;
                Ok(())
            }

            fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
                let (input, base) = BaseField::compact_deserialize(input)?;
                let (input, hash) = H::Hash::compact_deserialize(input)?;
                Ok((input, HashedBaseField { base, hash }))
            }
        }

        #[test]
        fn test_proc_macro() {
            let test_instance = TestStruct {
                base: [BaseField::from_u32_unchecked(1654); 64],
            };

            let mut output = Vec::new();
            test_instance.compact_serialize(&mut output).unwrap();
            let (remaining, deserialized) = TestStruct::compact_deserialize(&output).unwrap();
            assert!(remaining.is_empty());
            assert_eq!(deserialized, test_instance);
        }

        #[test]
        fn test_proc_macro_zipped_field() {
            let test_instance_unzipped = TestStruct {
                base: [BaseField::from_u32_unchecked(1654); 64],
            };
            let mut output_unzipped = Vec::new();
            test_instance_unzipped
                .compact_serialize(&mut output_unzipped)
                .unwrap();

            let test_instance_zipped = TestStructZipped {
                base: [BaseField::from_u32_unchecked(1654); 64],
            };
            let mut output_zipped = Vec::new();
            test_instance_zipped
                .compact_serialize(&mut output_zipped)
                .unwrap();

            assert!(
                output_zipped.len() < output_unzipped.len(),
                "Zipped output should be smaller (on redundant data)"
            );

            let (remaining, deserialized) =
                TestStructZipped::compact_deserialize(&output_zipped).unwrap();
            assert!(remaining.is_empty());
            assert_eq!(deserialized, test_instance_zipped);
        }

        #[test]
        fn test_proc_macro_generic() {
            let test_instance_generic: TestStructGeneric<Blake2sMerkleHasher> = TestStructGeneric {
                hashed_base: HashedBaseField::new(BaseField::from_u32_unchecked(1)),
            };

            let mut output = Vec::new();
            test_instance_generic
                .compact_serialize(&mut output)
                .unwrap();
            let (remaining, deserialized) =
                TestStructGeneric::<Blake2sMerkleHasher>::compact_deserialize(&output).unwrap();
            assert!(remaining.is_empty());
            assert_eq!(deserialized, test_instance_generic);
        }
    }
}
