use std::marker::PhantomData;

use itertools::Itertools;
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::prelude::*;
use stwo_prover::core::backend::simd::m31::PackedM31;

pub type MutRow<'trace> = Vec<&'trace mut PackedM31>;

/// An iterator over mutable references to the rows of a [`super::component_trace::ComponentTrace`].
// TODO(Ohad): Iterating over single rows is not optimal, figure out optimal chunk size when using
// this iterator.
pub struct RowIterMut<'trace> {
    v: Vec<*mut [PackedM31]>,
    phantom: PhantomData<&'trace ()>,
}
impl<'trace> RowIterMut<'trace> {
    pub fn new(slice: Vec<&'trace mut [PackedM31]>) -> Self {
        Self {
            v: slice.into_iter().map(|s| s as *mut _).collect_vec(),
            phantom: PhantomData,
        }
    }
}
impl<'trace> Iterator for RowIterMut<'trace> {
    type Item = MutRow<'trace>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.v[0].is_empty() {
            return None;
        }
        let item: Vec<&mut PackedM31> = self
            .v
            .iter_mut()
            .map(|col_chunk| unsafe {
                // SAFETY: The self.v contract ensures that any split_at_mut is valid.
                let (head, tail) = col_chunk.split_at_mut(1);
                *col_chunk = tail;
                &mut (*head)[0]
            })
            .collect_vec();
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.v[0].len();
        (len, Some(len))
    }
}
impl ExactSizeIterator for RowIterMut<'_> {}
impl DoubleEndedIterator for RowIterMut<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.v[0].is_empty() {
            return None;
        }
        let item: Vec<&mut PackedM31> = self
            .v
            .iter_mut()
            .map(|col_chunk| unsafe {
                // SAFETY: The self.v contract ensures that any split_at_mut is valid.
                let (head, tail) = col_chunk.split_at_mut(col_chunk.len() - 1);
                *col_chunk = head;
                &mut (*tail)[0]
            })
            .collect_vec();
        Some(item)
    }
}

struct RowProducer<'trace> {
    data: Vec<&'trace mut [PackedM31]>,
}
impl<'trace> Producer for RowProducer<'trace> {
    type Item = MutRow<'trace>;

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right): (Vec<_>, Vec<_>) = self
            .data
            .into_iter()
            .map(|slice| slice.split_at_mut(index))
            .unzip();

        (RowProducer { data: left }, RowProducer { data: right })
    }

    type IntoIter = RowIterMut<'trace>;

    fn into_iter(self) -> Self::IntoIter {
        RowIterMut {
            v: self.data.into_iter().map(|s| s as *mut _).collect_vec(),
            phantom: PhantomData,
        }
    }
}

/// A parallel iterator over mutable references to the rows of a
/// [`super::component_trace::ComponentTrace`]. [`super::component_trace::ComponentTrace`] is an
/// array of columns, hence iterating over rows is not trivial. Iteration is done by iterating over
/// `N` columns in parallel.
pub struct ParRowIterMut<'trace> {
    data: Vec<&'trace mut [PackedM31]>,
}
impl<'trace> ParRowIterMut<'trace> {
    pub(super) fn new(data: Vec<&'trace mut [PackedM31]>) -> Self {
        Self { data }
    }
}
impl<'trace> ParallelIterator for ParRowIterMut<'trace> {
    type Item = MutRow<'trace>;

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
impl IndexedParallelIterator for ParRowIterMut<'_> {
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
