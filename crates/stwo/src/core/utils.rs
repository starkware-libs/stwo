use core::iter::Peekable;
use core::mem::{self, MaybeUninit};
use core::ops::Deref;
use core::ptr;

use std_shims::Vec;

use super::fields::Field;

/// An enum that either borrows or owns a value.
/// Useful when a struct can optionally receive an external `& T` but also needs a fallback owned
/// instance.
pub enum MaybeOwned<'a, T> {
    Borrowed(&'a T),
    Owned(T),
}

impl<T> Deref for MaybeOwned<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        match self {
            MaybeOwned::Borrowed(r) => r,
            MaybeOwned::Owned(v) => v,
        }
    }
}

pub trait IteratorMutExt<'a, T: 'a>: Iterator<Item = &'a mut T> {
    fn assign(self, other: impl IntoIterator<Item = T>)
    where
        Self: Sized,
    {
        self.zip(other).for_each(|(a, b)| *a = b);
    }
}

impl<'a, T: 'a, I: Iterator<Item = &'a mut T>> IteratorMutExt<'a, T> for I {}

/// An iterator that takes elements from the underlying [Peekable] while the predicate is true.
/// Used to implement [PeekableExt::peek_take_while].
pub struct PeekTakeWhile<'a, I: Iterator, P: FnMut(&I::Item) -> bool> {
    iter: &'a mut Peekable<I>,
    predicate: P,
}
impl<I: Iterator, P: FnMut(&I::Item) -> bool> Iterator for PeekTakeWhile<'_, I, P> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next_if(&mut self.predicate)
    }
}
pub trait PeekableExt<'a, I: Iterator> {
    /// Returns an iterator that takes elements from the underlying [Peekable] while the predicate
    /// is true.
    /// Unlike [Iterator::take_while], this iterator does not consume the first element that does
    /// not satisfy the predicate.
    fn peek_take_while<P: FnMut(&I::Item) -> bool>(
        &'a mut self,
        predicate: P,
    ) -> PeekTakeWhile<'a, I, P>;
}
impl<'a, I: Iterator> PeekableExt<'a, I> for Peekable<I> {
    fn peek_take_while<P: FnMut(&I::Item) -> bool>(
        &'a mut self,
        predicate: P,
    ) -> PeekTakeWhile<'a, I, P> {
        PeekTakeWhile {
            iter: self,
            predicate,
        }
    }
}

/// Extension trait providing `checked_as_chunks` / `checked_as_chunks_mut` on slices.
/// Wraps `as_chunks`, asserting that no remainder is left over.
pub trait SliceExt {
    type Item;

    /// Splits the slice into chunks of exactly `N` elements and returns them as a slice of arrays.
    /// Panics if the slice length is not a multiple of `N`.
    fn checked_as_chunks<const N: usize>(&self) -> &[[Self::Item; N]];

    /// Splits the slice into mutable chunks of exactly `N` elements.
    /// Panics if the slice length is not a multiple of `N`.
    fn checked_as_chunks_mut<const N: usize>(&mut self) -> &mut [[Self::Item; N]];
}

impl<T> SliceExt for [T] {
    type Item = T;

    fn checked_as_chunks<const N: usize>(&self) -> &[[T; N]] {
        let (chunks, remainder) = self.as_chunks::<N>();
        assert!(remainder.is_empty());
        chunks
    }

    fn checked_as_chunks_mut<const N: usize>(&mut self) -> &mut [[T; N]] {
        let (chunks, remainder) = self.as_chunks_mut::<N>();
        assert!(remainder.is_empty());
        chunks
    }
}

pub fn all_unique<T: Eq + core::hash::Hash>(iter: impl IntoIterator<Item = T>) -> bool {
    let mut used = hashbrown::HashSet::new();
    iter.into_iter().all(|elt| used.insert(elt))
}

/// Returns the bit reversed index of `i` which is represented by `log_size` bits.
pub const fn bit_reverse_index(i: usize, log_size: u32) -> usize {
    if log_size == 0 {
        return i;
    }
    i.reverse_bits() >> (usize::BITS - log_size)
}

/// Conservative L1 data cache size estimate (bytes).
const L1_CACHE_SIZE: usize = 32 * 1024;

/// Performs an in-place bit-reversal permutation.
///
/// Uses a cache-blocking algorithm (Carter & Gatlin, 1998) for arrays larger than L1 cache,
/// falling back to the naive swap-based approach for small arrays.
///
/// # Panics
///
/// Panics if the length of the slice is not a power of two.
pub fn bit_reverse<T>(v: &mut [T]) {
    let n = v.len();
    assert!(n.is_power_of_two());
    if n <= 1 {
        return;
    }

    let elem_size = mem::size_of::<T>().max(1);
    let log_n = n.ilog2();

    // Use cache-blocking when the array exceeds half the L1 cache and is large enough for
    // tiling (need at least 2 * log_tile bits in log_n).
    let log_tile = log_tile_size::<T>(log_n);
    if n * elem_size > L1_CACHE_SIZE / 2 && 2 * log_tile <= log_n && log_tile > 0 {
        // SAFETY: `bit_reverse_cache_blocked` only performs element-sized memcpy and swaps
        // through raw pointers, which is valid for any `T`. All source and destination
        // positions are within bounds, and every element ends up in exactly one valid
        // position. No `Drop` glue is skipped because no values are created or destroyed;
        // they are only moved between `v` and a temporary buffer.
        unsafe {
            bit_reverse_cache_blocked(v, log_tile);
        }
    } else {
        bit_reverse_naive(v);
    }
}

/// Naive bit-reversal permutation. O(n) swaps with random access pattern.
fn bit_reverse_naive<T>(v: &mut [T]) {
    let n = v.len();
    let log_n = n.ilog2();
    for i in 0..n {
        let j = bit_reverse_index(i, log_n);
        if j > i {
            v.swap(i, j);
        }
    }
}

/// Computes the log2 of the tile dimension for cache-blocking.
///
/// Tile is `tile_size x tile_size` elements. Chosen so that `tile_size^2 * sizeof(T)` fits
/// in half of L1 cache (leaving room for the main array's active cache lines).
fn log_tile_size<T>(log_n: u32) -> u32 {
    let elem_size = mem::size_of::<T>().max(1);
    let half_l1_elems = (L1_CACHE_SIZE / 2) / elem_size;
    if half_l1_elems < 4 {
        return 0;
    }
    let mut q = half_l1_elems.next_power_of_two().ilog2() / 2;

    // Ensure tile is at least one cache line wide.
    let elems_per_line = (64 / elem_size).max(1);
    let min_q = elems_per_line.next_power_of_two().ilog2();
    q = q.max(min_q);

    // Shrink until 2*q fits in log_n.
    while 2 * q > log_n {
        if q == 0 {
            return 0;
        }
        q -= 1;
    }
    q
}

/// Cache-blocked bit-reversal permutation (Carter & Gatlin, 1998).
///
/// Decomposes index bits as `a (log_tile) | b (log_b) | c (log_tile)` and processes
/// tiles of `tile_size x tile_size` elements that fit in L1 cache.
///
/// # Safety
///
/// Caller must ensure `2 * log_tile <= log_n` and `log_tile > 0`.
unsafe fn bit_reverse_cache_blocked<T>(v: &mut [T], log_tile: u32) {
    let n = v.len();
    let log_n = n.ilog2();
    let tile_size = 1usize << log_tile;
    let log_b = log_n - 2 * log_tile;
    let b_len = 1usize << log_b;
    let shift_a = (log_b + log_tile) as usize;

    let mut temp: Vec<MaybeUninit<T>> = Vec::with_capacity(tile_size * tile_size);
    temp.set_len(tile_size * tile_size);

    let v_ptr = v.as_mut_ptr();

    for b in 0..b_len {
        let b_rev = bit_reverse_index(b, log_b);

        // Phase 1: Copy tile into temp with partial bit-reversal of `a`.
        // temp[rev(a) << log_tile | c] = v[a << shift_a | b << log_tile | c]
        for a in 0..tile_size {
            let a_rev = bit_reverse_index(a, log_tile);
            let src_base = (a << shift_a) | (b << log_tile as usize);
            let dst_base = a_rev << log_tile as usize;
            for c in 0..tile_size {
                ptr::copy_nonoverlapping(
                    v_ptr.add(src_base | c),
                    temp[dst_base | c].as_mut_ptr(),
                    1,
                );
            }
        }

        // Phase 2: For idx < idx_rev, swap v[idx_rev] with temp[t_idx].
        for c in 0..tile_size {
            let c_rev = bit_reverse_index(c, log_tile);
            for a_rev in 0..tile_size {
                let a = bit_reverse_index(a_rev, log_tile);
                let idx = (a << shift_a) | (b << log_tile as usize) | c;
                let idx_rev = (c_rev << shift_a) | (b_rev << log_tile as usize) | a_rev;
                if idx < idx_rev {
                    let t_idx = (a_rev << log_tile as usize) | c;
                    ptr::swap_nonoverlapping(
                        v_ptr.add(idx_rev),
                        temp[t_idx].as_mut_ptr(),
                        1,
                    );
                }
            }
        }

        // Phase 3: For idx < idx_rev, swap v[idx] with temp[t_idx].
        for a in 0..tile_size {
            let a_rev = bit_reverse_index(a, log_tile);
            for c in 0..tile_size {
                let c_rev = bit_reverse_index(c, log_tile);
                let idx = (a << shift_a) | (b << log_tile as usize) | c;
                let idx_rev = (c_rev << shift_a) | (b_rev << log_tile as usize) | a_rev;
                if idx < idx_rev {
                    let t_idx = (a_rev << log_tile as usize) | c;
                    ptr::swap_nonoverlapping(
                        v_ptr.add(idx),
                        temp[t_idx].as_mut_ptr(),
                        1,
                    );
                }
            }
        }
    }
}

/// Returns the index of the previous element in a bit reversed
/// [crate::prover::poly::circle::CircleEvaluation] of log size `eval_log_size` relative to a
/// smaller domain of size `domain_log_size`.
pub const fn previous_bit_reversed_circle_domain_index(
    i: usize,
    domain_log_size: u32,
    eval_log_size: u32,
) -> usize {
    offset_bit_reversed_circle_domain_index(i, domain_log_size, eval_log_size, -1)
}

/// Returns the index of the offset element in a bit reversed
/// [crate::prover::poly::circle::CircleEvaluation] of log size `eval_log_size` relative to a
/// smaller domain of size `domain_log_size`.
pub const fn offset_bit_reversed_circle_domain_index(
    i: usize,
    domain_log_size: u32,
    eval_log_size: u32,
    offset: isize,
) -> usize {
    let mut prev_index = bit_reverse_index(i, eval_log_size);
    let half_size = 1 << (eval_log_size - 1);
    let step_size = offset * (1 << (eval_log_size - domain_log_size - 1)) as isize;
    if prev_index < half_size {
        prev_index = (prev_index as isize + step_size).rem_euclid(half_size as isize) as usize;
    } else {
        prev_index =
            ((prev_index as isize - step_size).rem_euclid(half_size as isize) as usize) + half_size;
    }
    bit_reverse_index(prev_index, eval_log_size)
}

// TODO(AlonH): Pair both functions below with bit reverse. Consider removing both and calculating
// the indices instead.
#[cfg(feature = "prover")]
pub(crate) fn circle_domain_order_to_coset_order(
    values: &[crate::core::fields::m31::BaseField],
) -> Vec<crate::core::fields::m31::BaseField> {
    let n = values.len();
    let mut coset_order = vec![];
    for i in 0..(n / 2) {
        coset_order.push(values[i]);
        coset_order.push(values[n - 1 - i]);
    }
    coset_order
}

pub fn coset_order_to_circle_domain_order<F: Field>(values: &[F]) -> Vec<F> {
    let mut circle_domain_order = Vec::with_capacity(values.len());
    let n = values.len();
    let half_len = n / 2;
    for i in 0..half_len {
        circle_domain_order.push(values[i << 1]);
    }
    for i in 0..half_len {
        circle_domain_order.push(values[n - 1 - (i << 1)]);
    }
    circle_domain_order
}

/// Converts an index within a [`CircleDomain`] to the corresponding index in a [`Coset`].
///
/// [`CircleDomain`]: crate::core::poly::circle::CircleDomain
/// [`Coset`]: crate::core::circle::Coset
pub const fn circle_domain_index_to_coset_index(
    circle_index: usize,
    log_domain_size: u32,
) -> usize {
    let n = 1 << log_domain_size;
    if circle_index < n / 2 {
        circle_index * 2
    } else {
        (n - 1 - circle_index) * 2 + 1
    }
}

/// Converts an index within a [`Coset`] to the corresponding index in a [`CircleDomain`].
///
/// [`CircleDomain`]: crate::core::poly::circle::CircleDomain
/// [`Coset`]: crate::core::circle::Coset
pub const fn coset_index_to_circle_domain_index(coset_index: usize, log_domain_size: u32) -> usize {
    if coset_index.is_multiple_of(2) {
        coset_index / 2
    } else {
        ((2 << log_domain_size) - coset_index) / 2
    }
}

/// Performs a coset-natural-order to circle-domain-bit-reversed-order permutation in-place.
///
/// # Panics
///
/// Panics if the length of the slice is not a power of two.
pub fn bit_reverse_coset_to_circle_domain_order<T>(v: &mut [T]) {
    let n = v.len();
    assert!(n.is_power_of_two());
    let log_n = n.ilog2();
    for i in 0..n {
        let j = bit_reverse_index(coset_index_to_circle_domain_index(i, log_n), log_n);
        if j > i {
            v.swap(i, j);
        }
    }
}

/// # Safety
///
/// The caller must ensure that the vector is initialized before use.
#[allow(clippy::uninit_vec)]
pub unsafe fn uninit_vec<T>(len: usize) -> Vec<T> {
    let mut vec = Vec::with_capacity(len);
    vec.set_len(len);
    vec
}

#[cfg(all(test, feature = "prover"))]
mod tests {
    use itertools::Itertools;

    use super::{
        offset_bit_reversed_circle_domain_index, previous_bit_reversed_circle_domain_index,
    };
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::utils::{
        circle_domain_index_to_coset_index, coset_index_to_circle_domain_index,
    };
    use crate::m31;
    use crate::prover::backend::cpu::CpuCircleEvaluation;
    use crate::prover::poly::NaturalOrder;

    #[test]
    fn test_offset_bit_reversed_circle_domain_index() {
        let domain_log_size = 3;
        let eval_log_size = 6;
        let initial_index = 5;

        let actual = offset_bit_reversed_circle_domain_index(
            initial_index,
            domain_log_size,
            eval_log_size,
            -2,
        );
        let expected_prev = previous_bit_reversed_circle_domain_index(
            initial_index,
            domain_log_size,
            eval_log_size,
        );
        let expected_prev2 = previous_bit_reversed_circle_domain_index(
            expected_prev,
            domain_log_size,
            eval_log_size,
        );
        assert_eq!(actual, expected_prev2);
    }

    #[test]
    fn test_previous_bit_reversed_circle_domain_index() {
        let log_size = 4;
        let n = 1 << log_size;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let values = (0..n).map(|i| m31!(i as u32)).collect_vec();
        let evaluation = CpuCircleEvaluation::<_, NaturalOrder>::new(domain, values);
        let bit_reversed_evaluation = evaluation.clone().bit_reverse();

        //            2   ·  14
        //         ·      |       ·
        //      13        |          1
        //    ·           |            ·
        //   3            |             15
        //  ·             |              ·
        // 12             |               0
        // ·--------------|---------------·
        // 4              |               8
        //  ·             |              ·
        //   11           |              7
        //    ·           |            ·
        //      5         |          9
        //         ·      |       ·
        //            10  ·   6
        let neighbor_pairs = (0..n)
            .map(|index| {
                let prev_index =
                    previous_bit_reversed_circle_domain_index(index, log_size - 3, log_size);
                (
                    bit_reversed_evaluation[index],
                    bit_reversed_evaluation[prev_index],
                )
            })
            .sorted()
            .collect_vec();
        let mut expected_neighbor_pairs = vec![
            (m31!(0), m31!(4)),
            (m31!(15), m31!(11)),
            (m31!(1), m31!(5)),
            (m31!(14), m31!(10)),
            (m31!(2), m31!(6)),
            (m31!(13), m31!(9)),
            (m31!(3), m31!(7)),
            (m31!(12), m31!(8)),
            (m31!(4), m31!(0)),
            (m31!(11), m31!(15)),
            (m31!(5), m31!(1)),
            (m31!(10), m31!(14)),
            (m31!(6), m31!(2)),
            (m31!(9), m31!(13)),
            (m31!(7), m31!(3)),
            (m31!(8), m31!(12)),
        ];
        expected_neighbor_pairs.sort();

        assert_eq!(neighbor_pairs, expected_neighbor_pairs);
    }

    #[test]
    fn test_circle_domain_and_coset_index_conversion() {
        let log_size = 3;
        let n = 1 << log_size;

        // Test that both functions are inverses of each other
        for i in 0..n {
            let coset_idx = circle_domain_index_to_coset_index(i, log_size);
            let circle_idx = coset_index_to_circle_domain_index(coset_idx, log_size);
            assert_eq!(i, circle_idx);
        }
    }
}
