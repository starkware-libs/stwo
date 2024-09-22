use std::iter::Peekable;

use super::fields::m31::BaseField;
use super::fields::Field;

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

/// Returns the bit reversed index of `i` which is represented by `log_size` bits.
pub const fn bit_reverse_index(i: usize, log_size: u32) -> usize {
    if log_size == 0 {
        return i;
    }
    i.reverse_bits() >> (usize::BITS - log_size)
}

/// Returns the index of the previous element in a bit reversed
/// [super::poly::circle::CircleEvaluation] of log size `eval_log_size` relative to a smaller domain
/// of size `domain_log_size`.
pub const fn previous_bit_reversed_circle_domain_index(
    i: usize,
    domain_log_size: u32,
    eval_log_size: u32,
) -> usize {
    offset_bit_reversed_circle_domain_index(i, domain_log_size, eval_log_size, -1)
}

/// Returns the index of the offset element in a bit reversed
/// [super::poly::circle::CircleEvaluation] of log size `eval_log_size` relative to a smaller domain
/// of size `domain_log_size`.
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
pub(crate) fn circle_domain_order_to_coset_order(values: &[BaseField]) -> Vec<BaseField> {
    let n = values.len();
    let mut coset_order = vec![];
    for i in 0..(n / 2) {
        coset_order.push(values[i]);
        coset_order.push(values[n - 1 - i]);
    }
    coset_order
}

pub(crate) fn coset_order_to_circle_domain_order<F: Field>(values: &[F]) -> Vec<F> {
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

/// Converts an index within a [`Coset`] to the corresponding index in a [`CircleDomain`].
///
/// [`CircleDomain`]: crate::core::poly::circle::CircleDomain
/// [`Coset`]: crate::core::circle::Coset
pub const fn coset_index_to_circle_domain_index(coset_index: usize, log_domain_size: u32) -> usize {
    if coset_index % 2 == 0 {
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

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::{
        offset_bit_reversed_circle_domain_index, previous_bit_reversed_circle_domain_index,
    };
    use crate::core::backend::cpu::CpuCircleEvaluation;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::poly::NaturalOrder;
    use crate::m31;

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
}
