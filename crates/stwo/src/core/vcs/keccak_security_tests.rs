#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    use crate::core::channel::{Channel, Keccak256Channel};
    use crate::core::fields::m31::P;
    use crate::core::vcs::keccak_hash::{Keccak256Hash, Keccak256Hasher};
    use crate::core::vcs::keccak_merkle::Keccak256MerkleHasher;
    use crate::core::vcs::MerkleHasher;

    // Helper to count set bits
    fn count_diff_bits(h1: Keccak256Hash, h2: Keccak256Hash) -> u32 {
        let b1: [u8; 32] = h1.into();
        let b2: [u8; 32] = h2.into();
        b1.iter().zip(b2.iter()).map(|(a, b)| (a ^ b).count_ones()).sum()
    }

    #[test]
    fn test_avalanche_effect() {
        // Test that flipping 1 bit in input changes ~50% of bits in output
        let mut rng = StdRng::seed_from_u64(12345);
        let input_len = 64;
        let mut input: Vec<u8> = (0..input_len).map(|_| rng.gen()).collect();
        
        let h1 = Keccak256Hasher::hash(&input);
        
        // Flip one bit
        let byte_idx = rng.gen_range(0..input_len);
        let bit_idx = rng.gen_range(0..8);
        input[byte_idx] ^= 1 << bit_idx;
        
        let h2 = Keccak256Hasher::hash(&input);
        
        let diff = count_diff_bits(h1, h2);
        
        // Expected diff is 128 bits (50% of 256). 
        // 5 sigma range is approx [88, 168]
        assert!(diff > 90 && diff < 166, "Avalanche failed: changed {} bits (expected ~128)", diff);
    }

    #[test]
    fn test_merkle_domain_separation() {
        // Test that the prefix separation between LEAF and NODE works.
        // We compare:
        // 1. A Leaf node with empty values (simulated)
        // 2. A Internal Node with simulated children
        // Even if the input binary data *after* prefix was identical, the prefix should separate them.
        
        let dummy_hash = Keccak256Hash::default();
        
        // hash_node for NODE uses NODE_PREFIX + left + right
        // If we tried to simulate this using a LEAF, we would need to put the children hashes as column values?
        // But hash_node implementation for LEAF takes column values.
        // It's hard to create a collision because the input structure is different (Option<(Hash, Hash)> vs &[BaseField]).
        // But we can verify that the output of a node hash is NOT what we would get if we just hashed the children without prefix.
        
        let node_hash = Keccak256MerkleHasher::hash_node(Some((dummy_hash, dummy_hash)), &[]);
        
        // Manually hash without prefix (simulating "no domain separation")
        let mut raw_hasher = Keccak256Hasher::new();
        // Skip prefix
        raw_hasher.update(dummy_hash.as_ref());
        raw_hasher.update(dummy_hash.as_ref());
        let raw_hash = raw_hasher.finalize();
        
        assert_ne!(node_hash, raw_hash, "Node hash should include domain separation prefix");
        
        // Also compare Leaf vs Node just in case
        let leaf_hash = Keccak256MerkleHasher::hash_node(None, &[]);
        assert_ne!(node_hash, leaf_hash, "Node and Leaf (empty) must be different");
    }

    #[test]
    fn test_channel_uniformity() {
        // Chi-squared test for uniformity of draw_u32s
        // This validates that the Keccak256 output is being properly used as a PRNG source.
        let mut channel = Keccak256Channel::default();
        channel.mix_u64(0xDEADBEEF);
        
        let n_buckets = 16;
        let n_samples = 10_000;
        let mut buckets = vec![0u32; n_buckets];
        
        // We'll draw u32s and bin them by their highest 4 bits
        let mut drawn_count = 0;
        while drawn_count < n_samples {
            let random_u32s = channel.draw_u32s();
            for val in random_u32s {
                if drawn_count >= n_samples { break; }
                
                // Use top 4 bits for bucket index [0, 15]
                let bucket = (val >> 28) as usize;
                buckets[bucket] += 1;
                drawn_count += 1;
            }
        }
        
        let expected = n_samples as f64 / n_buckets as f64;
        let chi_squared: f64 = buckets.iter()
            .map(|&o| {
                let diff = o as f64 - expected;
                diff * diff / expected
            })
            .sum();
            
        // Degrees of freedom = 15
        // Critical value p=0.01 is 30.58
        // Critical value p=0.001 is 37.7
        println!("Chi-squared statistic (u32s): {}", chi_squared);
        assert!(chi_squared < 40.0, "Distribution of draw_u32s is likely not uniform (Chi2 = {})", chi_squared);
    }
    
    #[test]
    fn test_felt_uniformity() {
        // Chi-squared test for draw_secure_felt
        let mut channel = Keccak256Channel::default();
        channel.mix_u64(0xCAFEBABE);
        
        let n_buckets = 20;
        let n_samples = 5_000;
        let mut buckets = vec![0u32; n_buckets];
        
        for _ in 0..n_samples {
            let felt = channel.draw_secure_felt();
            // SecureField is extension. We check the first base field component.
            // Assuming .0 or .to_m31_array() exposes M31.
            // Using a hack if accessor not available: M31 is u32 wrapper.
            // Let's use debug formatting or assumption about structure if public API is restrictive.
            // core::fields::qm31::SecureField::to_m31_array is public.
            let val_m31 = felt.to_m31_array()[0]; 
            let val_u32 = val_m31.0;
            
            // Bucket based on range [0, P)
            let bucket = (val_u32 as u64 * n_buckets as u64 / P as u64) as usize;
            if bucket < n_buckets {
                buckets[bucket] += 1;
            }
        }
        
        let expected = n_samples as f64 / n_buckets as f64;
        let chi_squared: f64 = buckets.iter()
            .map(|&o| {
                let diff = o as f64 - expected;
                diff * diff / expected
            })
            .sum();
            
        // DoF = 19. Critical p=0.01 ~ 36.
        println!("Chi-squared statistic (felts): {}", chi_squared);
        assert!(chi_squared < 45.0, "Distribution of felts is likely not uniform (Chi2 = {})", chi_squared);
    }
}
