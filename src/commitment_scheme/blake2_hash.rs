use std::fmt;

use super::blake2s_ref;
use super::hasher::Hasher;

// Wrapper for the blake2s hash type.
#[derive(Clone, Copy, PartialEq, Default, Eq)]
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

impl fmt::Debug for Blake2sHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Blake2sHash as fmt::Display>::fmt(self, f)
    }
}

impl super::hasher::Name for Blake2sHash {
    const NAME: std::borrow::Cow<'static, str> = std::borrow::Cow::Borrowed("BLAKE2");
}

impl super::hasher::Hash<u8> for Blake2sHash {}

/// Wrapper for the blake2s Hashing functionalities.
#[derive(Clone, Debug)]
pub struct Blake2sHasher {
    state: [u32; 8],
    pending_buffer: [u8; 64],
    pending_len: usize,
}

impl Default for Blake2sHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher for Blake2sHasher {
    type Hash = Blake2sHash;
    const BLOCK_SIZE: usize = 64;
    const OUTPUT_SIZE: usize = 32;
    type NativeType = u8;

    fn new() -> Self {
        Self {
            state: [0; 8],
            pending_buffer: [0; 64],
            pending_len: 0,
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    fn update(&mut self, mut data: &[u8]) {
        while self.pending_len + data.len() >= 64 {
            // Fill the buffer and compress.
            let (prefix, rest) = data.split_at(64 - self.pending_len);
            self.pending_buffer[self.pending_len..].copy_from_slice(prefix);
            data = rest;
            let words =
                unsafe { std::mem::transmute::<&[u8; 64], &[u32; 16]>(&self.pending_buffer) };
            self.state = blake2s_ref::compress(self.state, *words, 0, 0, 0, 0);
            self.pending_len = 0;
        }
        // Copy the remaining data into the buffer.
        self.pending_buffer[self.pending_len..self.pending_len + data.len()].copy_from_slice(data);
        self.pending_len += data.len();
    }

    fn finalize(mut self) -> Blake2sHash {
        if self.pending_len != 0 {
            self.update(&[0; 64]);
        }
        Blake2sHash(unsafe { std::mem::transmute(self.state) })
    }

    fn finalize_reset(&mut self) -> Blake2sHash {
        let hash = self.clone().finalize();
        self.reset();
        hash
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
            "f2ab64ae6530f3a5d19369752cd30eadf455153c29dbf2cb70f00f73d5b41c50"
        );
    }

    #[test]
    fn hash_many_vs_single() {
        let hash_a = blake2_hash::Blake2sHasher::hash(b"a");
        let mut out = [0_u8; 32];
        let out_ptrs = [out.as_mut_ptr()];
        unsafe { Blake2sHasher::hash_many_in_place(&[b"a".as_ptr()], 1, &out_ptrs) };
        assert_eq!(hash_a.to_string(), hex::encode(out));
    }

    #[test]
    fn hash_many_xof_test() {
        let input1 = "a";
        let input2 = "b";
        let input_arr = [input1.as_ptr(), input2.as_ptr()];

        let mut out = [0_u8; 96];
        let out_ptrs = [out.as_mut_ptr(), unsafe { out.as_mut_ptr().add(32) }];
        unsafe { Blake2sHasher::hash_many_in_place(&input_arr, 1, &out_ptrs) };

        assert_eq!("f2ab64ae6530f3a5d19369752cd30eadf455153c29dbf2cb70f00f73d5b41c504383e3109201cbbb59233961bb55f5fe0c49f444751dc782080fe1d9780bfded0000000000000000000000000000000000000000000000000000000000000000", hex::encode(out));
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
