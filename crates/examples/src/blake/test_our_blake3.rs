//! Test our Blake3 implementation with permuted message

#[cfg(test)]
mod tests {
    use std::simd::u32x16;
    use crate::blake::blake3;

    #[test]
    fn test_our_blake3_with_123() {
        let input = [1u8, 2u8, 3u8];

        // Padding do 64 bajtów
        let mut padded = [0u8; 64];
        padded[..3].copy_from_slice(&input);

        // Konwersja do u32
        let message: [u32; 16] = std::array::from_fn(|i| {
            u32::from_le_bytes([
                padded[i * 4],
                padded[i * 4 + 1],
                padded[i * 4 + 2],
                padded[i * 4 + 3],
            ])
        });

        // Blake3 IV
        const IV: [u32; 8] = [
            0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
            0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
        ];

        // Initialize state for Blake3 compression
        // Ref: https://github.com/BLAKE3-team/BLAKE3/blob/master/reference_impl/reference_impl.rs
        let mut v = [0u32; 16];
        // First 8 words: chaining value (IV for first chunk)
        v[0..8].copy_from_slice(&IV);
        // Next 4 words: IV[0..4]
        v[8..12].copy_from_slice(&IV[0..4]);
        // Last 4 words: counter_low, counter_high, block_len, flags
        v[12] = 0;  // counter_low
        v[13] = 0;  // counter_high
        v[14] = 3;  // block_len in bytes (we have 3 bytes input)
        v[15] = 0b1011;  // flags: CHUNK_START | CHUNK_END | ROOT

        // Convert to SIMD
        let mut v_simd: [u32x16; 16] = v.map(u32x16::splat);
        let m_simd: [u32x16; 16] = message.map(u32x16::splat);
        let mut m_current = m_simd;

        // Save original chaining value for finalization
        let chaining_value: [u32; 8] = v[0..8].try_into().unwrap();

        // Run 7 rounds (Blake3) - permute message BETWEEN each round
        for r in 0..7 {
            blake3::round(&mut v_simd, m_current, r);
            // Permute message for next round
            m_current = blake3::MSG_SCHEDULE.map(|i| m_current[i as usize]);
        }

        // Extract state from SIMD
        let mut state: [u32; 16] = std::array::from_fn(|i| v_simd[i].as_array()[0]);

        // Finalize: XOR with chaining value
        // Ref: https://github.com/BLAKE3-team/BLAKE3/blob/master/reference_impl/reference_impl.rs
        for i in 0..8 {
            state[i] ^= state[i + 8];
            state[i + 8] ^= chaining_value[i];
        }

        // Output is first 8 words
        let hash: [u32; 8] = state[0..8].try_into().unwrap();

        let hash_bytes: Vec<u8> = hash.iter().flat_map(|&x| x.to_le_bytes()).collect();
        let hash_hex: String = hash_bytes.iter().map(|b| format!("{:02x}", b)).collect();

        // Expected hash for input [1,2,3] with Blake3
        let expected = "b177ec1bf26dfb3b7010d473e6d44713b29b765b99c6e60ecbfae742de496543";
        assert_eq!(hash_hex, expected, "Blake3 hash mismatch!");
    }
}
