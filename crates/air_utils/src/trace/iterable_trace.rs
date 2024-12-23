use std::marker::PhantomData;

use bytemuck::{cast_slice, Zeroable};
use itertools::Itertools;
use stwo_prover::core::backend::simd::m31::{PackedM31, N_LANES};
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::fields::m31::M31;
use stwo_prover::core::poly::circle::CircleEvaluation;
use stwo_prover::core::poly::BitReversedOrder;

/// A 2D Matrix of [`PackedM31`] values.
/// Used for generating the witness of 'Stwo' proofs.
/// Stored in column-major order, exposes a vectorized iterator over rows.
///
/// # Example:
///
///  ```text
/// Computation trace of a^2 + (a + 1)^2 for a in 0..256
/// ```
/// ```
/// use stwo_air_utils::trace::iterable_trace::IterableTrace;
/// use itertools::Itertools;
/// use stwo_prover::core::backend::simd::m31::{PackedM31, N_LANES};
/// use stwo_prover::core::fields::m31::M31;
/// use stwo_prover::core::fields::FieldExpOps;
///
/// const N_COLUMNS: usize = 3;
/// const LOG_SIZE: u32 = 8;
/// let mut trace = IterableTrace::<N_COLUMNS>::zeroed(LOG_SIZE);
/// let example_input = (0..1 << LOG_SIZE).map(M31::from).collect_vec(); // 0..256
/// trace
///     .iter_mut()
///     .zip(example_input.chunks(N_LANES))
///     .chunks(4)
///     .into_iter()
///     .for_each(|chunk| {
///         chunk.into_iter().for_each(|(row, input)| {
///             *row[0] = PackedM31::from_array(input.try_into().unwrap());
///             *row[1] = *row[0] + PackedM31::broadcast(M31(1));
///             *row[2] = row[0].square() + row[1].square();
///         })
///     });
///
/// let expected_first_3_rows = "0,1,1\n1,2,5\n2,3,13\n";
///
/// assert_eq!(trace.pretty_print(3), expected_first_3_rows);
/// ```
#[derive(Debug)]
pub struct IterableTrace<const N: usize> {
    data: [Vec<PackedM31>; N],

    /// Number of M31 rows in each column.
    length: usize,
}

impl<const N: usize> IterableTrace<N> {
    pub fn zeroed(log_size: u32) -> Self {
        let length = 1 << log_size;
        let n_simd_elems = length / N_LANES;
        let data = [(); N].map(|_| vec![PackedM31::zeroed(); n_simd_elems]);
        Self { data, length }
    }

    /// # Safety
    /// The caller must ensure that the column is populated before being used.
    #[allow(clippy::uninit_vec)]
    pub unsafe fn uninitialized(_log_size: u32) -> Self {
        todo!()
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn iter_mut(&mut self) -> RowIterMut<'_, N> {
        let v = self
            .data
            .iter_mut()
            .map(|column| column.as_mut_slice())
            .collect_vec()
            .try_into()
            .unwrap();
        RowIterMut::new(v)
    }

    pub fn to_evals(self) -> [CircleEvaluation<SimdBackend, M31, BitReversedOrder>; N] {
        todo!()
    }

    pub fn pretty_print(&self, row_limit: usize) -> String {
        assert!(row_limit <= self.len());
        let cpu_trace: Vec<&[u32]> = self
            .data
            .iter()
            .map(|column| cast_slice(column))
            .collect_vec();
        let mut output = String::new();
        for row in 0..row_limit {
            for (j, col) in cpu_trace.iter().enumerate() {
                output.push_str(&format!("{:?}", col[row]));
                if j < N - 1 {
                    output.push(',');
                }
            }
            output.push('\n');
        }
        output
    }
}

pub type MutRow<'trace, const N: usize> = [&'trace mut PackedM31; N];

pub struct RowIterMut<'trace, const N: usize> {
    v: [*mut [PackedM31]; N],
    phantom: PhantomData<&'trace ()>,
}
impl<'trace, const N: usize> RowIterMut<'trace, N> {
    pub fn new(slice: [&'trace mut [PackedM31]; N]) -> Self {
        Self {
            v: slice.map(|s| s as *mut _),
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
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.v[0].len();
        (len, Some(len))
    }
}
impl<const N: usize> ExactSizeIterator for RowIterMut<'_, N> {}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use stwo_prover::core::backend::simd::m31::{PackedM31, N_LANES};
    use stwo_prover::core::fields::m31::M31;
    use stwo_prover::core::fields::FieldExpOps;

    #[test]
    fn test_trace_iter_mut() {
        const N_COLUMNS: usize = 3;
        const LOG_SIZE: u32 = 8;
        let mut trace = super::IterableTrace::<N_COLUMNS>::zeroed(LOG_SIZE);
        let arr = (0..1 << LOG_SIZE).map(M31::from).collect_vec();
        let expected = arr
            .iter()
            .map(|&a| {
                let b = a + M31::from(1);
                let c = a.square() + b.square();
                (a, b, c)
            })
            .multiunzip();

        trace
            .iter_mut()
            .zip(arr.chunks(N_LANES))
            .chunks(4)
            .into_iter()
            .for_each(|chunk| {
                chunk.into_iter().for_each(|(row, input)| {
                    *row[0] = PackedM31::from_array(input.try_into().unwrap());
                    *row[1] = *row[0] + PackedM31::broadcast(M31(1));
                    *row[2] = row[0].square() + row[1].square();
                })
            });
        let actual = trace
            .data
            .map(|c| {
                c.into_iter()
                    .flat_map(|packed| packed.to_array())
                    .collect_vec()
            })
            .into_iter()
            .next_tuple()
            .unwrap();

        assert_eq!(expected, actual);
    }
}
