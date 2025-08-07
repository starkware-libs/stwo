use std::array;
use std::io::{Cursor, Read, Write};

use starknet_ff::FieldElement;
// Re-export the derive macro for use in other crates.
pub use stwo_compact_binary_derive::CompactBinary;
use unsigned_varint::encode::{u32_buffer, u64_buffer, usize_buffer};
use unsigned_varint::{decode, encode};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

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

/// Trait for types that can be serialized and deserialized in a compact binary format.
///
/// ## Format guidelines
/// - Integers (`u32`, `u64`, and `usize`) should be handled as VarInts.
/// - Relevant `FieldElement` fields should be compactified if possible
///  - Structured data should have:
///    - version numbers, to be able to update the structure
///    - tags for each field, to be able to add new fields
///
/// ## Struct Versioning
/// If we want to add or change a field of a struct `StructA`, while still being able to deserialize
/// previous versions of this struct, we should:
/// - Update `compact_serialize()` to serialize a new version number, and serialize the new struct
/// - Update `compact_deserialize()` to:
///   - Get the version of the deserialized struct
///   - Match on it and dispatch to the deserialization logic corresponding to this version
pub trait CompactBinary {
    /// Serializes the object into a compact binary format.
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError>;

    /// Deserializes the object from a compact binary format.
    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError>
    where
        Self: Sized;
}

/// Error enum for CompactBinary deserialization.
pub enum CompactDeserializeError {
    /// UnexpectedVersion(got, expected),
    UnexpectedVersion(u32, u32),
    /// UnexpectedTag(got, expected),
    UnexpectedTag(usize, usize),
    /// Generic decode error, e.g. when the input is malformed.
    DecodeError,
}

/// Error struct for CompactBinary serialization.
pub struct CompactSerializeError;

/// Helper function to convert a byte slice into an array of a specific size from a closure if
/// possible.
pub fn buf_to_array_ctr<F: Fn(&[u8; N]) -> V, V, const N: usize>(
    buf: &[u8],
    ctr: F,
) -> Option<(&[u8], V)> {
    Some((&buf[N..], ctr(&buf.get(..N)?.try_into().ok()?)))
}

/// Helper function to deserialize a struct's version and check it against an expected value.
pub fn strip_expected_version(
    input: &[u8],
    expected_version: u32,
) -> Result<&[u8], CompactDeserializeError> {
    let (input, version) = u32::compact_deserialize(input)?;
    if version != expected_version {
        return Err(CompactDeserializeError::UnexpectedVersion(
            version,
            expected_version,
        ));
    }
    Ok(input)
}

/// Helper function to deserialize a field's tag and check it against an expected value.
pub fn strip_expected_tag(
    input: &[u8],
    expected_tag: usize,
) -> Result<&[u8], CompactDeserializeError> {
    let (input, tag) = usize::compact_deserialize(input)?;
    if tag != expected_tag {
        return Err(CompactDeserializeError::UnexpectedTag(tag, expected_tag));
    }
    Ok(input)
}

/// A wrapper type for zipping and unzipping data during serialization and deserialization.
pub struct ZippedCompactBinary<T>(pub T);

impl<T: CompactBinary> ZippedCompactBinary<&T> {
    pub fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        let mut unzipped_data = Vec::new();
        T::compact_serialize(self.0, &mut unzipped_data)?;
        let zipped_data = zip_bytes(&unzipped_data)?;
        usize::compact_serialize(&zipped_data.len(), output)?;
        output.extend_from_slice(&zipped_data);
        Ok(())
    }

    pub fn compact_deserialize(input: &[u8]) -> Result<(&[u8], T), CompactDeserializeError> {
        let (input, len) = usize::compact_deserialize(input)?;
        let (zipped_data, input) = input.split_at(len);
        let unzipped_data = unzip_bytes(zipped_data)?;
        let (_, data) = T::compact_deserialize(&unzipped_data)?;
        Ok((input, data))
    }
}

/// Helper function for zipping bytes with Bzip2 compression.
fn zip_bytes(input: &[u8]) -> Result<Vec<u8>, CompactSerializeError> {
    let mut buf = Vec::new();
    let cursor = Cursor::new(&mut buf);
    let mut zip = ZipWriter::new(cursor);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Bzip2);
    zip.start_file("", options)
        .map_err(|_| CompactSerializeError)?;
    zip.write_all(input).map_err(|_| CompactSerializeError)?;
    let mut cursor = zip.finish().map_err(|_| CompactSerializeError)?;
    cursor.set_position(0);
    let mut out = Vec::new();
    Read::read_to_end(&mut cursor, &mut out).map_err(|_| CompactSerializeError)?;
    Ok(out)
}

/// Helper function for unzipping bytes.
fn unzip_bytes(input: &[u8]) -> Result<Vec<u8>, CompactDeserializeError> {
    let cursor = Cursor::new(input);
    let mut archive = ZipArchive::new(cursor).map_err(|_| CompactDeserializeError::DecodeError)?;
    let mut file = archive
        .by_index(0)
        .map_err(|_| CompactDeserializeError::DecodeError)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .map_err(|_| CompactDeserializeError::DecodeError)?;
    Ok(buf)
}

impl CompactBinary for u32 {
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        output.extend_from_slice(encode::u32(*self, &mut u32_buffer()));
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let (value, input) =
            decode::u32(input).map_err(|_| CompactDeserializeError::DecodeError)?;
        Ok((input, value))
    }
}

impl CompactBinary for u64 {
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        output.extend_from_slice(encode::u64(*self, &mut u64_buffer()));
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let (value, input) =
            decode::u64(input).map_err(|_| CompactDeserializeError::DecodeError)?;
        Ok((input, value))
    }
}

impl CompactBinary for usize {
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        output.extend_from_slice(encode::usize(*self, &mut usize_buffer()));
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let (value, input) =
            decode::usize(input).map_err(|_| CompactDeserializeError::DecodeError)?;
        Ok((input, value))
    }
}

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

impl CompactBinary for FieldElement {
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        output.extend_from_slice(&self.to_bytes_be());
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let (input, field_elem) = buf_to_array_ctr(input, FieldElement::from_bytes_be)
            .ok_or(CompactDeserializeError::DecodeError)?;
        let field_elem = field_elem.map_err(|_| CompactDeserializeError::DecodeError)?;
        Ok((input, field_elem))
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

impl<T: CompactBinary> CompactBinary for Option<T> {
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        if let Some(value) = self {
            output.push(b'1');
            value.compact_serialize(output)?;
        } else {
            output.push(b'0');
        }
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let (first, input) = if input.is_empty() {
            Err(CompactDeserializeError::DecodeError)
        } else {
            Ok((input[0], &input[1..]))
        }?;
        if first == b'1' {
            let (input, value) = T::compact_deserialize(input)?;
            Ok((input, Some(value)))
        } else {
            Ok((input, None))
        }
    }
}

impl<T: CompactBinary + Clone, const N: usize> CompactBinary for [T; N] {
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        for v in self {
            v.compact_serialize(output)?;
        }
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let mut input = input;
        let mut values = Vec::with_capacity(N);
        for _ in 0..N {
            let (updated_input, value) = T::compact_deserialize(input)?;
            input = updated_input;
            values.push(value);
        }
        Ok((input, array::from_fn(|i| values[i].clone())))
    }
}

impl<T: CompactBinary> CompactBinary for Vec<T> {
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        self.len().compact_serialize(output)?;
        for v in self {
            v.compact_serialize(output)?;
        }
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let (mut input, len) = usize::compact_deserialize(input)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            let (updated_input, value) = T::compact_deserialize(input)?;
            input = updated_input;
            values.push(value);
        }
        Ok((input, values))
    }
}

impl<T0: CompactBinary, T1: CompactBinary> CompactBinary for (T0, T1) {
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        let (v0, v1) = self;
        v0.compact_serialize(output)?;
        v1.compact_serialize(output)?;
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let (input, v0) = T0::compact_deserialize(input)?;
        let (input, v1) = T1::compact_deserialize(input)?;
        Ok((input, (v0, v1)))
    }
}

impl<T0: CompactBinary, T1: CompactBinary, T2: CompactBinary> CompactBinary for (T0, T1, T2) {
    fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
        let (v0, v1, v2) = self;
        v0.compact_serialize(output)?;
        v1.compact_serialize(output)?;
        v2.compact_serialize(output)?;
        Ok(())
    }

    fn compact_deserialize(input: &[u8]) -> Result<(&[u8], Self), CompactDeserializeError> {
        let (input, v0) = T0::compact_deserialize(input)?;
        let (input, v1) = T1::compact_deserialize(input)?;
        let (input, v2) = T2::compact_deserialize(input)?;
        Ok((input, (v0, v1, v2)))
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
