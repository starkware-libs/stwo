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

    /// Given twiddles for the canonic coset `G_{n+1} * <G_n>`, returns twiddles for the
    /// subdomain `G_{n+1} * <G_{subdomain_log_size}>`.
    fn extract_subdomain_twiddles(&self, subdomain_log_size: u32) -> Self;
}

impl<T: Copy> TwiddleBuffer<BitReversedOrder> for Vec<T> {
    fn empty() -> Self {
        Vec::new()
    }

    fn extract_subdomain_twiddles(&self, subdomain_log_size: u32) -> Self {
        let subdomain_half_log_size = subdomain_log_size - 1;
        let half_log_size = self.len().ilog2();
        assert!(
            subdomain_half_log_size <= half_log_size,
            "subdomain_half_log_size={subdomain_half_log_size} > half_log_size={half_log_size}"
        );

        // A twiddle buffer of size 2^K stores K FFT layers concatenated from outermost (largest)
        // to innermost (size 1): layer i has 2^{K-1-i} elements starting at offset
        // 2^K - 2^{K-i}. The K layers total 2^K - 1 elements; one unused element pads the
        // buffer to a power of two.
        //
        // In bit-reversed order the subdomain is a prefix, so for each layer we take the
        // first `subdomain_layer_size` elements.
        let buf_size = 1usize << half_log_size;
        let out_size = 1usize << subdomain_half_log_size;
        let mut result = Vec::with_capacity(out_size);

        for layer in 0..subdomain_half_log_size as usize {
            let layer_start = buf_size - (buf_size >> layer);
            let subdomain_layer_size = 1usize << (subdomain_half_log_size as usize - 1 - layer);
            result.extend_from_slice(&self[layer_start..layer_start + subdomain_layer_size]);
        }
        // Padding to round the output buffer to a power of two.
        result.push(self[self.len() - 1]);
        debug_assert_eq!(result.len(), out_size);
        result
    }
}
