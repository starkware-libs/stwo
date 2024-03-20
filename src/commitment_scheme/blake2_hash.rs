use std::arch::x86_64::{__m512i, _mm512_loadu_si512};
use std::fmt;

use super::blake2s_avx::{compress16_transposed, set1, transpose_msgs, transpose_states};
use super::blake2s_ref;

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

impl super::hasher::Hasher for Blake2sHasher {
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
            blake2s_ref::compress(&mut self.state, words, 0, 0, 0, 0);
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

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe fn hash_many_in_place(
        data: &[*const u8],
        single_input_length_bytes: usize,
        dst: &[*mut u8],
    ) {
        // TODO(spapini): this implementation assumes that dst are consecutive.
        //   Match that in the trait.
        let mut dst = dst[0];

        // Compress 16 instances at a time.
        let mut data_iter = data.array_chunks::<16>();
        for inputs in &mut data_iter {
            let bytes = compress16(inputs, single_input_length_bytes);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
            dst = dst.add(16 * 32);
        }
        // Handle the remainder.
        let inputs = data_iter.remainder();
        if inputs.is_empty() {
            return;
        }
        // Pad inputs with the same input address.
        let remainder = inputs.len();
        let inputs = inputs
            .iter()
            .copied()
            .chain(std::iter::repeat(inputs[0]))
            .take(16)
            .collect::<Vec<_>>();
        let bytes = compress16(&inputs.try_into().unwrap(), single_input_length_bytes);
        // Store only the relevant part of the output.
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, remainder * 32);
    }
}

/// Compress 16 blake2s instances.
/// # Safety
/// Inputs must be of the size `single_input_length_bytes`.
unsafe fn compress16(inputs: &[*const u8; 16], single_input_length_bytes: usize) -> [u8; 16 * 32] {
    let mut states = [set1(0); 8];
    let mut inputs = inputs.map(|input| input);

    // Compress in chunks of 64 bytes.
    for _j in (0..single_input_length_bytes).step_by(64) {
        // Load next 64 bytes of each of the 16 inputs, unaligned.
        // TODO(spapini): Figure out if aligned loading is possible.
        let chunk_per_input: [__m512i; 16] =
            inputs.map(|input| _mm512_loadu_si512(input as *const i32));
        inputs = inputs.map(|input| input.add(64));

        // Transpose the input chunks (16 chunks of 16 u32s each), to get 16 __m512i, each
        // representing 16 packed instances of a message word.
        let vectorized_chunk = transpose_msgs(chunk_per_input);

        // Compress the 16 instances.
        compress16_transposed(
            &mut states,
            &vectorized_chunk,
            set1(0),
            set1(0),
            set1(0),
            set1(0),
        );
    }

    // Handle the remainder.
    let remainder = single_input_length_bytes % 64;
    if remainder != 0 {
        // Load the remainder of each input, padded with 0s.
        let mut words = [set1(0); 16];
        for (i, input) in inputs.into_iter().enumerate() {
            let mut word = [0; 64];
            word[..remainder].copy_from_slice(std::slice::from_raw_parts(input, remainder));
            words[i] = _mm512_loadu_si512(word.as_ptr() as *const i32);
        }
        // Compress the 16 instances.
        compress16_transposed(
            &mut states,
            &transpose_msgs(words),
            set1(single_input_length_bytes as i32),
            set1(0),
            set1(0),
            set1(0),
        );
    }

    // Transpose the states, from 8 packed words, to get 16 results, each of size 32B.
    transpose_states(states)
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
    fn hash_many_xof_test() {
        let input1 = "a";
        let input2 = "b";
        let input_arr = [input1.as_ptr(), input2.as_ptr()];

        let mut out = [0_u8; 96];
        let out_ptrs = [out.as_mut_ptr(), unsafe { out.as_mut_ptr().add(42) }];
        unsafe { Blake2sHasher::hash_many_in_place(&input_arr, 1, &out_ptrs) };

        assert_eq!("8e7b8823fa9ad8fb8b6e992849c2bbfa0bb1809c1b0666996d6c622ac1df197d85230cd8a7f7d2cd23e24497ac432193e8efa81ac6688f0b64efad1c53acaccf0000000000000000000000000000000000000000000000000000000000000000", hex::encode(out));
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
