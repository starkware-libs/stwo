use std::fmt;
use std::slice::Iter;

use super::hasher::Name;
use super::merkle_hasher::MerkleHasher;
use crate::core::fields::{Field, IntoSlice};

// Wrapper for the blake3 hash type.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Blake3Hash([u8; 32]);

impl From<Blake3Hash> for Vec<u8> {
    fn from(value: Blake3Hash) -> Self {
        Vec::from(value.0)
    }
}

impl From<Vec<u8>> for Blake3Hash {
    fn from(value: Vec<u8>) -> Self {
        Self(
            value
                .try_into()
                .expect("Failed converting Vec<u8> to Blake3Hash Type!"),
        )
    }
}

impl From<&[u8]> for Blake3Hash {
    fn from(value: &[u8]) -> Self {
        Self(
            value
                .try_into()
                .expect("Failed converting &[u8] to Blake3Hash Type!"),
        )
    }
}

impl AsRef<[u8]> for Blake3Hash {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Display for Blake3Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex::encode(self.0))
    }
}

impl Name for Blake3Hash {
    const NAME: std::borrow::Cow<'static, str> = std::borrow::Cow::Borrowed("BLAKE3");
}

// Wrapper for the blake3 Hashing functionalities.
#[derive(Clone)]
pub struct Blake3Hasher {}

impl super::hasher::Hasher for Blake3Hasher {
    type Hash = Blake3Hash;
    const BLOCK_SIZE: usize = 64;
    const OUTPUT_SIZE: usize = 32;
    type NativeType = u8;

    fn hash(val: &[u8]) -> Blake3Hash {
        Blake3Hash(*blake3::hash(val).as_bytes())
    }

    fn concat_and_hash(v1: &Self::Hash, v2: &Self::Hash) -> Blake3Hash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&v1.0);
        hasher.update(&v2.0);

        Blake3Hash(*hasher.finalize().as_bytes())
    }

    fn hash_one_in_place(data: &[u8], dst: &mut [u8]) {
        assert_eq!(
            dst.len(),
            Self::OUTPUT_SIZE,
            "Attempt to Generate blake3 hash of size different than 32 bytes!"
        );
        let mut hasher = blake3::Hasher::new();
        hasher.update(data);
        let mut output_reader = hasher.finalize_xof();
        output_reader.fill(&mut dst[..Self::OUTPUT_SIZE])
    }

    fn hash_many(data: &[Vec<u8>]) -> Vec<Self::Hash> {
        data.iter().map(|x| Self::hash(x)).collect()
    }

    // TODO(Ohad): Implement better blake3 module (SIMD & Memory optimizations)
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
        let mut hasher = blake3::Hasher::new();
        data.iter()
            .map(|input_group| {
                hasher.reset();
                input_group.iter().for_each(|d| {
                    hasher.update(d);
                });
                Blake3Hash(hasher.finalize().into())
            })
            .collect()
    }

    fn hash_many_multi_src_in_place(data: &[Vec<&[Self::NativeType]>], dst: &mut [Self::Hash]) {
        assert!(
            data.len() == dst.len(),
            "Attempt to hash many multi src with different input and output lengths!"
        );
        let mut hasher = blake3::Hasher::new();
        data.iter()
            .zip(dst.iter_mut())
            .for_each(|(input_group, out)| {
                hasher.reset();
                input_group.iter().for_each(|d| {
                    hasher.update(d);
                });
                *out = Blake3Hash(hasher.finalize().into());
            })
    }
}

impl<F: Field> MerkleHasher<F> for Blake3Hasher {
    /// Assumes prev_hashes is twice the size of dst.
    // TODO(Ohad): Implement SIMD blake3.
    fn inject_and_compress_layer_in_place(
        prev_hashes: Option<&[Self::Hash]>,
        dst: &mut [Self::Hash],
        col_iter: &Iter<'_, &[F]>,
    ) where
        F: IntoSlice<Self::NativeType>,
    {
        let produced_layer_length = dst.len();
        let mut hasher = blake3::Hasher::new();
        let col_iter = col_iter
            .clone()
            .zip(col_iter.clone().map(|c| c.len() / produced_layer_length));
        dst.iter_mut().enumerate().for_each(|(i, dst)| {
            hasher.reset();
            if let Some(hashes) = prev_hashes {
                hasher.update(&hashes[i * 2].0);
                hasher.update(&hashes[i * 2 + 1].0);
            }
            for (column, n_elements_in_chunk) in col_iter.clone() {
                let chunk = &column[i * n_elements_in_chunk..(i + 1) * n_elements_in_chunk];
                hasher.update(F::into_slice(chunk));
            }
            *dst = Blake3Hash(hasher.finalize().into());
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::blake3_hash::{Blake3Hash, Blake3Hasher};
    use crate::commitment_scheme::hasher::Hasher;
    use crate::commitment_scheme::merkle_hasher::MerkleHasher;
    use crate::core::fields::m31::M31;

    #[test]
    fn single_hash_test() {
        let hash_a = Blake3Hasher::hash(b"a");
        assert_eq!(
            hash_a.to_string(),
            "17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f"
        );
    }

    #[test]
    fn hash_many_test() {
        let input: Vec<Vec<u8>> = std::iter::repeat(b"a".to_vec()).take(3).collect();
        let hash_result = Blake3Hasher::hash_many(&input);

        for h in hash_result {
            assert_eq!(
                h.to_string(),
                "17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f"
            );
        }
    }

    #[test]
    fn hash_xof_test() {
        let input = b"a";
        let mut out = [0_u8; 32];

        Blake3Hasher::hash_one_in_place(input, &mut out[..]);
        assert_eq!(
            "17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f",
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
        unsafe { Blake3Hasher::hash_many_in_place(&input_arr, 1, &out_ptrs) };

        assert_eq!("17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f0000000000000000000010e5cf3d3c8a4f9f3468c8cc58eea84892a22fdadbc1acb22410190044c1d55300000000000000000000000000000000000000000000", hex::encode(out));
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
        let expected_result0 = Blake3Hasher::hash(b"abb");
        let expected_result1 = Blake3Hasher::hash(b"cccdddd");

        let hash_results = Blake3Hasher::hash_many_multi_src(&input_arr);
        Blake3Hasher::hash_many_multi_src_in_place(&input_arr, &mut hash_in_place_results);

        assert!(hash_results.len() == 2);
        assert_eq!(hash_results[0], expected_result0);
        assert_eq!(hash_results[1], expected_result1);
        assert_eq!(hash_in_place_results[0], expected_result0);
        assert_eq!(hash_in_place_results[1], expected_result1);
    }

    #[test]
    fn inject_and_compress_test() {
        let prev_hashes = vec![
            Blake3Hasher::hash(b"a"),
            Blake3Hasher::hash(b"b"),
            Blake3Hasher::hash(b"a"),
            Blake3Hasher::hash(b"b"),
        ];
        let mut dst = vec![Blake3Hash::default(); 2];
        let col1: Vec<M31> = (0..2).map(M31::from_u32_unchecked).collect();
        let col2: Vec<M31> = (2..4).map(M31::from_u32_unchecked).collect();
        let columns = [&col1[..], &col2[..]];
        let mut hasher = blake3::Hasher::new();
        hasher.update(blake3::hash(b"a").as_bytes());
        hasher.update(blake3::hash(b"b").as_bytes());
        hasher.update(&0_u32.to_le_bytes());
        hasher.update(&2_u32.to_le_bytes());
        let expected_result0 = Blake3Hash(hasher.finalize().into());
        hasher.reset();
        hasher.update(blake3::hash(b"a").as_bytes());
        hasher.update(blake3::hash(b"b").as_bytes());
        hasher.update(&1_u32.to_le_bytes());
        hasher.update(&3_u32.to_le_bytes());
        let expected_result1 = Blake3Hash(hasher.finalize().into());

        Blake3Hasher::inject_and_compress_layer_in_place(
            Some(&prev_hashes),
            &mut dst,
            &columns.iter(),
        );

        assert_eq!(dst[0], expected_result0);
        assert_eq!(dst[1], expected_result1);
    }
}
