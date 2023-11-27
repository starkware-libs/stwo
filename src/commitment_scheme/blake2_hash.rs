use std::fmt;

use blake2::digest::{Update, VariableOutput};
use blake2::{Blake2s256, Blake2sVar, Digest};

use crate::core::fields::m31::N_BYTES_FELT;

pub const FELTS_PER_HASH: usize = 8;

// Wrapper for the blake2s hash type.
#[derive(Clone, Copy, PartialEq, Debug)]
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

impl From<Blake2sHash> for [u32; FELTS_PER_HASH] {
    fn from(val: Blake2sHash) -> Self {
        val.0
            .chunks_exact(N_BYTES_FELT) // 4 bytes per u32.
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
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
    const BLOCK_SIZE_IN_BYTES: usize = 64;
    const OUTPUT_SIZE_IN_BYTES: usize = 32;

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
        let mut hasher = Blake2sVar::new(Self::OUTPUT_SIZE_IN_BYTES).unwrap();
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
                    .map(|p| std::slice::from_raw_parts_mut(*p, Self::OUTPUT_SIZE_IN_BYTES)),
            )
            .for_each(|(input, out)| Self::hash_one_in_place(input, out))
    }

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
}

#[cfg(test)]
mod tests {
    use super::Blake2sHasher;
    use crate::commitment_scheme::blake2_hash;
    use crate::commitment_scheme::hasher::Hasher;

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

        let hash_results = Blake2sHasher::hash_many_multi_src(&input_arr);

        assert!(hash_results.len() == 2);
        assert_eq!(hash_results[0], Blake2sHasher::hash(b"abb"));
        assert_eq!(hash_results[1], Blake2sHasher::hash(b"cccdddd"));
    }
}
