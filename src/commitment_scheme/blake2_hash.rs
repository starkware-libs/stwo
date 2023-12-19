use std::fmt;

use blake2::digest::{Update, VariableOutput};
use blake2::{Blake2s256, Blake2sVar, Digest};

use super::hasher::IncrementalHasher;

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

impl super::hasher::Hash<u8> for Blake2sHash {}

// Wrapper for the blake2s Hashing functionalities.
#[derive(Clone)]
pub struct Blake2sHasher {
    state: Blake2s256,
}

impl super::hasher::BasicHasher for Blake2sHasher {
    type Hash = Blake2sHash;
    const BLOCK_SIZE: usize = 64;
    const OUTPUT_SIZE: usize = 32;
    type NativeType = u8;

    fn hash(val: &[u8]) -> Self::Hash {
        let mut hasher = Blake2sHasher::new();
        hasher.update(val);
        hasher.finalize()
    }

    fn concat_and_hash(v1: &Self::Hash, v2: &Self::Hash) -> Blake2sHash {
        let mut hasher = Blake2sHasher::new();
        hasher.update(&v1.0);
        hasher.update(&v2.0);
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
        let mut hasher = Blake2sHasher::new();
        data.iter()
            .map(|input_group| {
                input_group.iter().for_each(|d| {
                    hasher.update(d);
                });
                hasher.finalize_reset()
            })
            .collect()
    }

    fn hash_many_multi_src_in_place(data: &[Vec<&[Self::NativeType]>], dst: &mut [Self::Hash]) {
        assert!(
            data.len() == dst.len(),
            "Attempt to hash many multi src with different input and output lengths!"
        );
        let mut hasher = Blake2sHasher::new();
        data.iter()
            .zip(dst.iter_mut())
            .for_each(|(input_group, out)| {
                input_group.iter().for_each(|d| {
                    hasher.update(d);
                });
                *out = hasher.finalize_reset();
            })
    }
}

impl IncrementalHasher<u8, Blake2sHash> for Blake2sHasher {
    fn new() -> Self {
        Self {
            state: Blake2s256::new(),
        }
    }

    fn reset(&mut self) {
        blake2::Digest::reset(&mut self.state);
    }

    fn update(&mut self, data: &[u8]) {
        blake2::Digest::update(&mut self.state, data);
    }

    fn finalize(self) -> Blake2sHash {
        Blake2sHash(self.state.finalize().into())
    }

    fn finalize_reset(&mut self) -> Blake2sHash {
        Blake2sHash(self.state.finalize_reset().into())
    }
}

#[cfg(test)]
mod tests {
    use super::Blake2sHasher;
    use crate::commitment_scheme::blake2_hash;
    use crate::commitment_scheme::hasher::{BasicHasher, IncrementalHasher};

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
    fn hash_state_test() {
        let mut state = Blake2sHasher::new();
        state.update(b"a");
        state.update(b"b");
        let hash = state.finalize_reset();
        let hash_empty = state.finalize();

        assert_eq!(hash.to_string(), Blake2sHasher::hash(b"ab").to_string());
        assert_eq!(hash_empty.to_string(), Blake2sHasher::hash(b"").to_string());
    }
}
