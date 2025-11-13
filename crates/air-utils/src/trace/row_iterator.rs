use std::marker::PhantomData;

use itertools::Itertools;
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::prelude::*;
use stwo::prover::backend::simd::m31::PackedM31;

pub type MutRow<'trace, const N: usize> = Box<[&'trace mut PackedM31; N]>;

/// An iterator over mutable references to the rows of a [`super::component_trace::ComponentTrace`].
// TODO(Ohad): Iterating over single rows is not optimal, figure out optimal chunk size when using
// this iterator.
pub struct RowIterMut<'trace, const N: usize> {
    v: Box<[*mut [PackedM31]; N]>,
    phantom: PhantomData<&'trace ()>,
}
impl<'trace, const N: usize> RowIterMut<'trace, N> {
    pub fn new(slice: [&'trace mut [PackedM31]; N]) -> Self {
        Self {
            v: Box::new(
                slice
                    .into_iter()
                    .map(|s| s as *mut _)
                    .collect_vec()
                    .try_into()
                    .unwrap(),
            ),
            phantom: PhantomData,
        }
    }
}
impl<'trace, const N: usize> Iterator for RowIterMut<'trace, N> {
    type Item = MutRow<'trace, N>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.v[0].is_empty() {
            return None;
        }
        let item = std::array::from_fn(|i| unsafe {
            // SAFETY: The self.v contract ensures that any split_at_mut is valid.
            let (head, tail) = self.v[i].split_at_mut(1);
            self.v[i] = tail;
            &mut (*head)[0]
        });

        Some(Box::new(item))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.v[0].len();
        (len, Some(len))
    }
}
impl<const N: usize> ExactSizeIterator for RowIterMut<'_, N> {}
impl<const N: usize> DoubleEndedIterator for RowIterMut<'_, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.v[0].is_empty() {
            return None;
        }
        let item = std::array::from_fn(|i| unsafe {
            // SAFETY: The self.v contract ensures that any split_at_mut is valid.
            let (head, tail) = self.v[i].split_at_mut(self.v[i].len() - 1);
            self.v[i] = head;
            &mut (*tail)[0]
        });

        Some(Box::new(item))
    }
}

struct RowProducer<'trace, const N: usize> {
    data: Box<[&'trace mut [PackedM31]; N]>,
}
impl<'trace, const N: usize> Producer for RowProducer<'trace, N> {
    type Item = MutRow<'trace, N>;

    fn split_at(self, index: usize) -> (Self, Self) {
        let mut left: [_; N] = unsafe { std::mem::zeroed() };
        let mut right: [_; N] = unsafe { std::mem::zeroed() };
        for (i, slice) in self.data.into_iter().enumerate() {
            let (lhs, rhs) = slice.split_at_mut(index);
            left[i] = lhs;
            right[i] = rhs;
        }

        (
            RowProducer {
                data: Box::new(left),
            },
            RowProducer {
                data: Box::new(right),
            },
        )
    }

    type IntoIter = RowIterMut<'trace, N>;

    fn into_iter(self) -> Self::IntoIter {
        RowIterMut {
            v: Box::new(
                self.data
                    .into_iter()
                    .map(|s| s as *mut _)
                    .collect_vec()
                    .try_into()
                    .unwrap(),
            ),
            phantom: PhantomData,
        }
    }
}

/// A parallel iterator over mutable references to the rows of a
/// [`super::component_trace::ComponentTrace`]. [`super::component_trace::ComponentTrace`] is an
/// array of columns, hence iterating over rows is not trivial. Iteration is done by iterating over
/// `N` columns in parallel.
pub struct ParRowIterMut<'trace, const N: usize> {
    data: Box<[&'trace mut [PackedM31]; N]>,
}
impl<'trace, const N: usize> ParRowIterMut<'trace, N> {
    pub(super) fn new(data: [&'trace mut [PackedM31]; N]) -> Self {
        Self {
            data: Box::new(data),
        }
    }
}
impl<'trace, const N: usize> ParallelIterator for ParRowIterMut<'trace, N> {
    type Item = MutRow<'trace, N>;

    fn drive_unindexed<D>(self, consumer: D) -> D::Result
    where
        D: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}
impl<const N: usize> IndexedParallelIterator for ParRowIterMut<'_, N> {
    fn len(&self) -> usize {
        self.data[0].len()
    }

    fn drive<D: Consumer<Self::Item>>(self, consumer: D) -> D::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(RowProducer { data: self.data })
    }
}
