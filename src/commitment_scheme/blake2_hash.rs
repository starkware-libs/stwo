use std::fmt;
use std::slice::Iter;

use blake2::digest::{Update, VariableOutput};
use blake2::{Blake2s256, Blake2sVar, Digest};

use super::merkle_hasher::MerkleHasher;
use crate::core::fields::{Field, IntoSlice};

// Wrapper for the blake2s hash type.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Blake2sHash([u8; 32]);

impl From<Blake2sHash> for Vec<u8> {
    fn from(value: Blake2sHash) -> Self {
        Vec::from(value.0)
    }
}

impl From<Vec<u8>> for Blake2sHash {
    fn from(value: Vec<u8>) -> Self {
        Self(
            value
                .try_into()
                .expect("Failed converting Vec<u8> to Blake2Hash type"),
        )
    }
}

impl From<&[u8]> for Blake2sHash {
    fn from(value: &[u8]) -> Self {
        Self(
            value
                .try_into()
                .expect("Failed converting &[u8] to Blake2sHash Type!"),
        )
    }
}

impl AsRef<[u8]> for Blake2sHash {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl From<Blake2sHash> for [u8; 32] {
    fn from(val: Blake2sHash) -> Self {
        val.0
    }
}

impl fmt::Display for Blake2sHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex::encode(self.0))
    }
}

impl super::hasher::Name for Blake2sHash {
    const NAME: std::borrow::Cow<'static, str> = std::borrow::Cow::Borrowed("BLAKE2");
}

// Wrapper for the blake2s Hashing functionalities.
#[derive(Clone)]
pub struct Blake2sHasher {}

impl super::hasher::Hasher for Blake2sHasher {
    type Hash = Blake2sHash;
    const BLOCK_SIZE: usize = 64;
    const OUTPUT_SIZE: usize = 32;
    type NativeType = u8;

    fn hash(val: &[u8]) -> Self::Hash {
        let mut hasher = Blake2s256::new();
        blake2::Digest::update(&mut hasher, val);
        Blake2sHash(hasher.finalize().into())
    }

    fn concat_and_hash(v1: &Self::Hash, v2: &Self::Hash) -> Blake2sHash {
        let mut hasher = Blake2s256::new();
        blake2::Digest::update(&mut hasher, v1.0);
        blake2::Digest::update(&mut hasher, v2.0);

        Blake2sHash(hasher.finalize().into())
    }

    fn hash_many(data: &[Vec<u8>]) -> Vec<Self::Hash> {
        data.iter().map(|x| Self::hash(x)).collect()
    }

    fn hash_one_in_place(data: &[u8], dst: &mut [u8]) {
        let mut hasher = Blake2sVar::new(Self::OUTPUT_SIZE).unwrap();
        hasher.update(data);
        hasher.finalize_variable(dst).unwrap();
    }

    unsafe fn hash_many_in_place(
        data: &[*const u8],
        single_input_length_bytes: usize,
        dst: &[*mut u8],
    ) {
        data.iter()
            .map(|p| std::slice::from_raw_parts(*p, single_input_length_bytes))
            .zip(
                dst.iter()
                    .map(|p| std::slice::from_raw_parts_mut(*p, Self::OUTPUT_SIZE)),
            )
            .for_each(|(input, out)| Self::hash_one_in_place(input, out))
    }

    // TODO(Ohad): Consider allocating manually and using the in_place function.
    fn hash_many_multi_src(data: &[Vec<&[u8]>]) -> Vec<Self::Hash> {
        let mut hasher = Blake2s256::new();
        data.iter()
            .map(|input_group| {
                input_group.iter().for_each(|d| {
                    blake2::Digest::update(&mut hasher, d);
                });
                Blake2sHash(hasher.finalize_reset().into())
            })
            .collect()
    }

    fn hash_many_multi_src_in_place(data: &[Vec<&[Self::NativeType]>], dst: &mut [Self::Hash]) {
        assert!(
            data.len() == dst.len(),
            "Attempt to hash many multi src with different input and output lengths!"
        );
        let mut hasher = Blake2s256::new();
        data.iter()
            .zip(dst.iter_mut())
            .for_each(|(input_group, out)| {
                input_group.iter().for_each(|d| {
                    blake2::Digest::update(&mut hasher, d);
                });
                *out = Blake2sHash(hasher.finalize_reset().into());
            })
    }
}

impl<F: Field + Sized> MerkleHasher<F> for Blake2sHasher {
    /// Assumes prev_hashes is twice the size of dst.
    // TODO(Ohad): Implement SIMD blake2s.
    fn inject_and_compress_layer_in_place(
        prev_hashes: Option<&[Self::Hash]>,
        dst: &mut [Self::Hash],
        col_iter: &Iter<'_, &[F]>,
    ) where
        F: IntoSlice<Self::NativeType>,
    {
        let produced_layer_length = dst.len();
        let mut hasher = blake2::Blake2s256::new();
        let dst_iter = dst.iter_mut();
        let col_iter = col_iter
            .clone()
            .zip(col_iter.clone().map(|c| c.len() / produced_layer_length));
        dst_iter.enumerate().for_each(|(i, dst)| {
            if let Some(hashes) = prev_hashes {
                blake2::Digest::update(&mut hasher, hashes[i * 2].0.as_ref());
                blake2::Digest::update(&mut hasher, hashes[i * 2 + 1].0.as_ref());
            }
            for (column, n_elements_in_chunk) in col_iter.clone() {
                let chunk = &column[i * n_elements_in_chunk..(i + 1) * n_elements_in_chunk];
                blake2::Digest::update(&mut hasher, F::into_slice(chunk));
            }
            *dst = Blake2sHash(hasher.finalize_reset().into());
        });
    }
}

#[cfg(test)]
mod tests {
    use blake2::{Blake2s256, Digest};

    use super::Blake2sHasher;
    use crate::commitment_scheme::blake2_hash::{self, Blake2sHash};
    use crate::commitment_scheme::hasher::Hasher;
    use crate::commitment_scheme::merkle_hasher::MerkleHasher;
    use crate::core::fields::m31::M31;

    #[test]
    fn single_hash_test() {
        let hash_a = blake2_hash::Blake2sHasher::hash(b"a");
        assert_eq!(
            hash_a.to_string(),
            "4a0d129873403037c2cd9b9048203687f6233fb6738956e0349bd4320fec3e90"
        );
    }

    #[test]
    fn hash_many_test() {
        let input: Vec<Vec<u8>> = std::iter::repeat(b"a".to_vec()).take(3).collect();
        let hash_result = blake2_hash::Blake2sHasher::hash_many(&input);

        for h in hash_result {
            assert_eq!(
                h.to_string(),
                "4a0d129873403037c2cd9b9048203687f6233fb6738956e0349bd4320fec3e90"
            );
        }
    }

    #[test]
    fn hash_xof_test() {
        let input = b"a";
        let mut out = [0_u8; 32];

        Blake2sHasher::hash_one_in_place(input, &mut out[..]);
        assert_eq!(
            "4a0d129873403037c2cd9b9048203687f6233fb6738956e0349bd4320fec3e90",
            hex::encode(out)
        )
    }

    #[test]
    fn hash_many_xof_test() {
        let input1 = "a";
        let input2 = "b";
        let input_arr = [input1.as_ptr(), input2.as_ptr()];

        let mut out = [0_u8; 96];
        let out_ptrs = [out.as_mut_ptr(), unsafe { out.as_mut_ptr().add(42) }];
        unsafe { Blake2sHasher::hash_many_in_place(&input_arr, 1, &out_ptrs) };

        assert_eq!("4a0d129873403037c2cd9b9048203687f6233fb6738956e0349bd4320fec3e900000000000000000000004449e92c9a7657ef2d677b8ef9da46c088f13575ea887e4818fc455a2bca50000000000000000000000000000000000000000000000", hex::encode(out));
    }

    #[test]
    fn hash_many_multi_src_test() {
        let input1 = b"a";
        let input2 = b"bb";
        let input3 = b"ccc";
        let input4 = b"dddd";
        let input_group_1 = [&input1[..], &input2[..]].to_vec();
        let input_group_2 = [&input3[..], &input4[..]].to_vec();
        let input_arr = [input_group_1, input_group_2];
        let mut hash_in_place_results = Vec::new();
        hash_in_place_results.resize(2, Default::default());
        let expected_result0 = Blake2sHasher::hash(b"abb");
        let expected_result1 = Blake2sHasher::hash(b"cccdddd");

        let hash_results = Blake2sHasher::hash_many_multi_src(&input_arr);
        Blake2sHasher::hash_many_multi_src_in_place(&input_arr, &mut hash_in_place_results);

        assert!(hash_results.len() == 2);
        assert_eq!(hash_results[0], expected_result0);
        assert_eq!(hash_results[1], expected_result1);
        assert_eq!(hash_in_place_results[0], expected_result0);
        assert_eq!(hash_in_place_results[1], expected_result1);
    }

    #[test]
    fn inject_and_compress_test() {
        let prev_hashes = vec![
            Blake2sHasher::hash(b"a"),
            Blake2sHasher::hash(b"b"),
            Blake2sHasher::hash(b"a"),
            Blake2sHasher::hash(b"b"),
        ];
        let mut dst = vec![Blake2sHash::default(); 2];
        let col1: Vec<M31> = (0..2).map(M31::from_u32_unchecked).collect();
        let col2: Vec<M31> = (2..4).map(M31::from_u32_unchecked).collect();
        let columns = [&col1[..], &col2[..]];
        let mut hasher = Blake2s256::new();
        hasher.update(Blake2sHasher::hash(b"a").as_ref());
        hasher.update(Blake2sHasher::hash(b"b").as_ref());
        hasher.update(0_u32.to_le_bytes());
        hasher.update(2_u32.to_le_bytes());
        let expected_result0 = Blake2sHash(hasher.finalize_reset().into());
        hasher.update(Blake2sHasher::hash(b"a").as_ref());
        hasher.update(Blake2sHasher::hash(b"b").as_ref());
        hasher.update(1_u32.to_le_bytes());
        hasher.update(3_u32.to_le_bytes());
        let expected_result1 = Blake2sHash(hasher.finalize().into());

        Blake2sHasher::inject_and_compress_layer_in_place(
            Some(&prev_hashes),
            &mut dst,
            &columns.iter(),
        );

        assert_eq!(dst[0], expected_result0);
        assert_eq!(dst[1], expected_result1);
    }
}
