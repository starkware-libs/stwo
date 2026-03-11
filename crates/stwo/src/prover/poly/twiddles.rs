use super::circle::PolyOps;
use super::BitReversedOrder;
use crate::core::circle::Coset;

/// Precomputed twiddles for a specific coset tower.
///
/// A coset tower is every repeated doubling of a `root_coset`.
/// The largest CircleDomain that can be ffted using these twiddles is one with `root_coset` as
/// its `half_coset`.
pub struct TwiddleTree<B: PolyOps> {
    pub root_coset: Coset,
    // TODO(shahars): Represent a slice, and grabbing, in a generic way
    pub twiddles: B::Twiddles,
    pub itwiddles: B::Twiddles,
}

unsafe impl<B: PolyOps> Sync for TwiddleTree<B> {}

/// Trait for twiddle buffers that support subdomain extraction.
pub trait TwiddleBuffer<Order> {
    /// Returns an empty twiddle buffer.
    ///
    /// Can be used as a placeholder when the TwiddleBuffer is not needed.
    fn empty() -> Self;

    /// Extracts twiddles for the subdomain `G_{n+1} * <G_{subdomain_log_size}>` of the
    /// canonic coset `G_{n+1} * <G_n>` (where `n = domain_log_size`).
    ///
    /// The buffer may contain twiddles for a domain larger than `domain_log_size`.
    fn extract_subdomain_twiddles(&self, domain_log_size: u32, subdomain_log_size: u32) -> Self;
}

/// In bit-reversed order the subdomain is a prefix, so at each FFT layer we take the
/// first portion of the corresponding layer.
impl<T: Copy> TwiddleBuffer<BitReversedOrder> for Vec<T> {
    fn empty() -> Self {
        Vec::new()
    }

    fn extract_subdomain_twiddles(&self, domain_log_size: u32, subdomain_log_size: u32) -> Self {
        let domain_half_log_size = domain_log_size - 1;
        let subdomain_half_log_size = subdomain_log_size - 1;
        let buf_half_log_size = self.len().ilog2();
        assert!(
            subdomain_half_log_size <= domain_half_log_size
                && domain_half_log_size <= buf_half_log_size,
            "Invalid sizes: subdomain_half={subdomain_half_log_size}, \
             domain_half={domain_half_log_size}, buf_half={buf_half_log_size}"
        );

        // A twiddle buffer of size 2^K stores K FFT layers concatenated from outermost
        // (largest) to innermost (size 1): layer i has 2^{K-1-i} elements starting at
        // offset 2^K - 2^{K-i}. The K layers total 2^K - 1 elements; one unused element
        // pads the buffer to a power of two.
        //
        // The source domain's layers start at root layer index (K - D), where
        // K = buf_half_log_size and D = domain_half_log_size.
        //
        // In bit-reversed order the subdomain is a prefix, so for each layer we take the
        // first `subdomain_layer_size` elements.
        let skip_layers = buf_half_log_size - domain_half_log_size;
        let buf_size = 1usize << buf_half_log_size;
        let out_size = 1usize << subdomain_half_log_size;
        let mut result = Vec::with_capacity(out_size);

        for layer in 0..subdomain_half_log_size as usize {
            let root_layer = skip_layers as usize + layer;
            let layer_start = buf_size - (buf_size >> root_layer);
            let subdomain_layer_size = 1usize << (subdomain_half_log_size as usize - 1 - layer);
            result.extend_from_slice(&self[layer_start..layer_start + subdomain_layer_size]);
        }
        // Padding to round the output buffer to a power of two.
        result.push(self[self.len() - 1]);
        debug_assert_eq!(result.len(), out_size);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::poly::circle::CanonicCoset;
    use crate::prover::backend::cpu::circle::slow_precompute_twiddles;

    #[test]
    fn test_extract_subdomain_twiddles() {
        let committed_log_size = 6;
        let subdomain_log_size = 4;

        let committed_domain = CanonicCoset::new(committed_log_size).circle_domain();
        let subdomain = committed_domain
            .split(committed_log_size - subdomain_log_size)
            .0;

        // Precompute twiddles for the committed domain and extract subdomain twiddles.
        let committed_twiddles = slow_precompute_twiddles(committed_domain.half_coset);
        let extracted: Vec<_> = TwiddleBuffer::<BitReversedOrder>::extract_subdomain_twiddles(
            &committed_twiddles,
            committed_log_size,
            subdomain_log_size,
        );

        // Precompute twiddles directly for the subdomain.
        let expected = slow_precompute_twiddles(subdomain.half_coset);

        // Compare all layers (excluding the padding element).
        assert_eq!(extracted.len(), expected.len());
        assert_eq!(
            extracted[..extracted.len() - 1],
            expected[..expected.len() - 1]
        );
    }

    /// Tests extraction when the twiddle buffer is larger than the committed domain.
    #[test]
    fn test_extract_subdomain_twiddles_from_larger_buffer() {
        let root_log_size = 8;
        let committed_log_size = 6;
        let subdomain_log_size = 4;

        let root_domain = CanonicCoset::new(root_log_size).circle_domain();
        let committed_domain = CanonicCoset::new(committed_log_size).circle_domain();
        let subdomain = committed_domain
            .split(committed_log_size - subdomain_log_size)
            .0;

        // Precompute twiddles for the root (larger) domain and extract subdomain twiddles.
        let root_twiddles = slow_precompute_twiddles(root_domain.half_coset);
        let extracted: Vec<_> = TwiddleBuffer::<BitReversedOrder>::extract_subdomain_twiddles(
            &root_twiddles,
            committed_log_size,
            subdomain_log_size,
        );

        // Precompute twiddles directly for the subdomain.
        let expected = slow_precompute_twiddles(subdomain.half_coset);

        // Compare all layers (excluding the padding element).
        assert_eq!(extracted.len(), expected.len());
        assert_eq!(
            extracted[..extracted.len() - 1],
            expected[..expected.len() - 1]
        );
    }
}
