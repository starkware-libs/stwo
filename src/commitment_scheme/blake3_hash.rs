use std::fmt;

use super::hasher::Name;

// Wrapper for the blake3 hash type.
#[derive(Clone, Copy, PartialEq, Debug, Default, Eq)]
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

impl super::hasher::Hash<u8> for Blake3Hash {}

// Wrapper for the blake3 Hashing functionalities.
#[derive(Clone)]
pub struct Blake3Hasher {
    state: blake3::Hasher,
}

impl super::hasher::Hasher for Blake3Hasher {
    type Hash = Blake3Hash;
    const BLOCK_SIZE: usize = 64;
    const OUTPUT_SIZE: usize = 32;
    type NativeType = u8;

    fn new() -> Self {
        Self {
            state: blake3::Hasher::new(),
        }
    }

    fn reset(&mut self) {
        self.state.reset();
    }

    fn update(&mut self, data: &[u8]) {
        self.state.update(data);
    }

    fn finalize(self) -> Blake3Hash {
        Blake3Hash(self.state.finalize().into())
    }

    fn finalize_reset(&mut self) -> Blake3Hash {
        let res = Blake3Hash(self.state.finalize().into());
        self.state.reset();
        res
    }

    unsafe fn hash_many_in_place(
        data: &[*const u8],
        single_input_length_bytes: usize,
        dst: &[*mut u8],
    ) {
        let mut hasher = blake3::Hasher::new();
        data.iter()
            .map(|p| std::slice::from_raw_parts(*p, single_input_length_bytes))
            .zip(
                dst.iter()
                    .map(|p| std::slice::from_raw_parts_mut(*p, Self::OUTPUT_SIZE)),
            )
            .for_each(|(input, out)| {
                hasher.update(input);
                let mut output_reader = hasher.finalize_xof();
                output_reader.fill(&mut out[..Self::OUTPUT_SIZE]);
                hasher.reset();
            })
    }
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::blake3_hash::Blake3Hasher;
    use crate::commitment_scheme::hasher::Hasher;

    #[test]
    fn single_hash_test() {
        let hash_a = Blake3Hasher::hash(b"a");
        assert_eq!(
            hash_a.to_string(),
            "17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f"
        );
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
    fn hash_state_test() {
        let mut state = Blake3Hasher::new();
        state.update(b"a");
        state.update(b"b");
        let hash = state.finalize_reset();
        let hash_empty = state.finalize();

        assert_eq!(hash.to_string(), Blake3Hasher::hash(b"ab").to_string());
        assert_eq!(hash_empty.to_string(), Blake3Hasher::hash(b"").to_string())
    }
}
