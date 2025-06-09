use std::ops::{Mul, Sub};

use itertools::{multizip, Itertools};
use num_traits::{One, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use stwo_prover::core::backend::simd::column::SecureColumn;
use stwo_prover::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES, N_LANES};
use stwo_prover::core::backend::simd::prefix_sum::inclusive_prefix_sum;
use stwo_prover::core::backend::simd::qm31::{batch_inverse_packed_qm31, PackedSecureField};
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::backend::Column;
use stwo_prover::core::channel::Channel;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo_prover::core::lookups::utils::Fraction;
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation};
use stwo_prover::core::poly::BitReversedOrder;
use stwo_prover::core::secure_column::SecureColumnByCoords;
use stwo_prover::core::utils::uninit_vec;
use stwo_prover::core::ColumnVec;

use super::EvalAtRow;

/// Evaluates constraints for batched logups.
/// These constraint enforce the sum of multiplicity_i / (z + sum_j alpha^j * x_j) = claimed_sum.
pub struct LogupAtRow<E: EvalAtRow> {
    /// The index of the interaction used for the cumulative sum columns.
    pub interaction: usize,
    /// The total sum of all the fractions divided by n_rows.
    pub cumsum_shift: SecureField,
    /// The evaluation of the last cumulative sum column.
    pub fracs: Vec<Fraction<E::EF, E::EF>>,
    pub is_finalized: bool,
    pub log_size: u32,
}

impl<E: EvalAtRow> Default for LogupAtRow<E> {
    fn default() -> Self {
        Self::dummy()
    }
}
impl<E: EvalAtRow> LogupAtRow<E> {
    pub fn new(interaction: usize, claimed_sum: SecureField, log_size: u32) -> Self {
        Self {
            interaction,
            cumsum_shift: claimed_sum / BaseField::from_u32_unchecked(1 << log_size),
            fracs: vec![],
            is_finalized: true,
            log_size,
        }
    }

    // TODO(alont): Remove this once unnecessary LogupAtRows are gone.
    pub fn dummy() -> Self {
        Self {
            interaction: 100,
            cumsum_shift: SecureField::one(),
            fracs: vec![],
            is_finalized: true,
            log_size: 10,
        }
    }
}

/// Ensures that the LogupAtRow is finalized.
/// LogupAtRow should be finalized exactly once.
impl<E: EvalAtRow> Drop for LogupAtRow<E> {
    fn drop(&mut self) {
        assert!(self.is_finalized, "LogupAtRow was not finalized");
    }
}

/// Interaction elements for the logup protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupElements<const N: usize> {
    pub z: SecureField,
    pub alpha: SecureField,
    pub alpha_powers: [SecureField; N],
}
impl<const N: usize> LookupElements<N> {
    pub fn draw(channel: &mut impl Channel) -> Self {
        let [z, alpha] = channel.draw_felts(2).try_into().unwrap();
        let mut cur = SecureField::one();
        let alpha_powers = std::array::from_fn(|_| {
            let res = cur;
            cur *= alpha;
            res
        });
        Self {
            z,
            alpha,
            alpha_powers,
        }
    }
    pub fn combine<F: Clone, EF>(&self, values: &[F]) -> EF
    where
        EF: Clone + Zero + From<F> + From<SecureField> + Mul<F, Output = EF> + Sub<EF, Output = EF>,
    {
        assert!(
            self.alpha_powers.len() >= values.len(),
            "Not enough alpha powers to combine values"
        );
        values
            .iter()
            .zip(self.alpha_powers)
            .fold(EF::zero(), |acc, (value, power)| {
                acc + EF::from(power) * value.clone()
            })
            - EF::from(self.z)
    }

    pub fn dummy() -> Self {
        Self {
            z: SecureField::from_u32_unchecked(1, 2, 3, 4),
            alpha: SecureField::one(),
            alpha_powers: [SecureField::one(); N],
        }
    }
}

// SIMD backend generator for logup interaction trace.
pub struct LogupTraceGenerator {
    log_size: u32,
    /// Current allocated interaction columns.
    trace: Vec<SecureColumnByCoords<SimdBackend>>,
    /// Denominator expressions (z + sum_i alpha^i * x_i) being generated for the current lookup.
    denom: SecureColumn,
    batch_inverse_buffer: Vec<PackedSecureField>,
}
impl LogupTraceGenerator {
    pub fn new(log_size: u32) -> Self {
        let trace = vec![];
        let denom = SecureColumn::zeros(1 << log_size);
        let batch_inverse_buffer = unsafe { uninit_vec(1 << (log_size - LOG_N_LANES)) };
        Self {
            log_size,
            trace,
            denom,
            batch_inverse_buffer,
        }
    }

    /// # Safety
    ///
    /// Calling `finalize_last` on uninitialized LogupTraceGenerator leads to undefined behavior.
    pub unsafe fn uninitialized(log_size: u32) -> Self {
        let trace = vec![];
        let denom = SecureColumn::uninitialized(1 << log_size);
        let batch_inverse_buffer = unsafe { uninit_vec(1 << (log_size - LOG_N_LANES)) };
        Self {
            log_size,
            trace,
            denom,
            batch_inverse_buffer,
        }
    }

    /// Allocate a new lookup column.
    pub fn new_col(&mut self) -> LogupColGenerator<'_> {
        let log_size = self.log_size;
        LogupColGenerator {
            gen: self,
            numerator: SecureColumnByCoords::<SimdBackend>::zeros(1 << log_size),
        }
    }

    pub fn col_from_iter(
        &mut self,
        iter: impl ExactSizeIterator<Item = (PackedSecureField, PackedSecureField)>,
    ) {
        let length = 1 << self.log_size;
        assert_eq!(iter.len() * N_LANES, length);
        let mut col_gen = self.new_col();
        for (vec_row, (numerator, denom)) in iter.enumerate() {
            col_gen.write_frac(vec_row, numerator, denom);
        }
        col_gen.finalize_col();
    }

    #[cfg(feature = "parallel")]
    pub fn col_from_par_iter(
        &mut self,
        iter: impl IndexedParallelIterator<Item = (PackedSecureField, PackedSecureField)>,
    ) {
        use stwo_prover::core::backend::simd::column::BaseColumn;

        let length = 1 << self.log_size;
        assert_eq!(iter.len() * N_LANES, length);

        let (((c0, c1), c2), c3) = (self.denom.data.par_iter_mut())
            .zip(iter)
            .map(|(dst_denom, (numerator, denom))| {
                *dst_denom = denom;
                let [c0, c1, c2, c3] = numerator.into_packed_m31s();
                (((c0, c1), c2), c3)
            })
            .unzip();
        let columns = [c0, c1, c2, c3].map(BaseColumn::from_simd);
        let numerator = SecureColumnByCoords::<SimdBackend> { columns };

        LogupColGenerator {
            gen: self,
            numerator,
        }
        .finalize_col();
    }

    /// Finalize the trace. Returns the trace and the total sum of the last column.
    /// The last column is shifted by the cumsum_shift.
    pub fn finalize_last(
        mut self,
    ) -> (
        ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        SecureField,
    ) {
        let mut last_col_coords = self.trace.pop().unwrap().columns;

        // Compute cumsum_shift.
        let coordinate_sums = last_col_coords.each_ref().map(|c| {
            c.data
                .iter()
                .copied()
                .sum::<PackedBaseField>()
                .pointwise_sum()
        });
        let claimed_sum = SecureField::from_m31_array(coordinate_sums);
        let cumsum_shift = claimed_sum / BaseField::from_u32_unchecked(1 << self.log_size);
        let packed_cumsum_shift = PackedSecureField::broadcast(cumsum_shift);

        last_col_coords.iter_mut().enumerate().for_each(|(i, c)| {
            c.data
                .iter_mut()
                .for_each(|x| *x -= packed_cumsum_shift.into_packed_m31s()[i])
        });
        let coord_prefix_sum = last_col_coords.map(inclusive_prefix_sum);
        let secure_prefix_sum = SecureColumnByCoords {
            columns: coord_prefix_sum,
        };
        self.trace.push(secure_prefix_sum);
        let trace = self
            .trace
            .into_iter()
            .flat_map(|eval| {
                eval.columns.map(|col| {
                    CircleEvaluation::new(CanonicCoset::new(self.log_size).circle_domain(), col)
                })
            })
            .collect_vec();
        (trace, claimed_sum)
    }
}

/// Trace generator for a single lookup column.
pub struct LogupColGenerator<'a> {
    gen: &'a mut LogupTraceGenerator,
    /// Numerator expressions (i.e. multiplicities) being generated for the current lookup.
    numerator: SecureColumnByCoords<SimdBackend>,
}
impl LogupColGenerator<'_> {
    /// Write a fraction to the column at a row.
    pub fn write_frac(
        &mut self,
        vec_row: usize,
        numerator: PackedSecureField,
        denom: PackedSecureField,
    ) {
        debug_assert!(
            denom.to_array().iter().all(|x| *x != SecureField::zero()),
            "{:?}",
            ("denom at vec_row {} is zero {}", denom, vec_row)
        );
        unsafe {
            self.numerator.set_packed(vec_row, numerator);
            *self.gen.denom.data.get_unchecked_mut(vec_row) = denom;
        }
    }

    /// Finalizes generating the column.
    pub fn finalize_col(mut self) {
        // Column size is a power of 2.
        let chunk_size = std::cmp::min(4, self.gen.denom.data.len());
        batch_inverse_packed_qm31(&self.gen.denom.data, &mut self.gen.batch_inverse_buffer);

        #[cfg(feature = "parallel")]
        let chunks_iter = {
            let denom_inv_chunks = self.gen.batch_inverse_buffer.par_chunks(chunk_size);
            let numerator_chunks = self.numerator.par_chunks_mut(chunk_size);
            (numerator_chunks, denom_inv_chunks).into_par_iter()
        };

        #[cfg(not(feature = "parallel"))]
        let chunks_iter = {
            let denom_inv_chunks = self.gen.batch_inverse_buffer.chunks(chunk_size);
            let numerator_chunks = self.numerator.chunks_mut(chunk_size);
            numerator_chunks.zip(denom_inv_chunks)
        };

        chunks_iter
            .enumerate()
            .for_each(|(chunk_idx, (mut numerator_chunk, denom_inv_chunk))| {
                #[allow(clippy::needless_range_loop)]
                for idx_in_chunk in 0..chunk_size {
                    unsafe {
                        let vec_row = chunk_idx * chunk_size + idx_in_chunk;
                        let value =
                            numerator_chunk.packed_at(idx_in_chunk) * denom_inv_chunk[idx_in_chunk];
                        let prev_value = self
                            .gen
                            .trace
                            .last()
                            .map(|col| col.packed_at(vec_row))
                            .unwrap_or_else(PackedSecureField::zero);
                        numerator_chunk.set_packed(idx_in_chunk, value + prev_value)
                    }
                }
            });

        self.gen.trace.push(self.numerator)
    }

    // TODO(Ohad): remove.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = FractionWriter<'_>> {
        let denom = self.gen.denom.data.iter_mut();
        let [coord0, coord1, coord2, coord3] =
            self.numerator.columns.each_mut().map(|s| &mut s.data);
        multizip((coord0, coord1, coord2, coord3, denom)).map(|(n0, n1, n2, n3, d)| {
            FractionWriter {
                numerator: [n0, n1, n2, n3],
                denom: d,
            }
        })
    }

    // TODO(Ohad): remove.
    #[cfg(feature = "parallel")]
    pub fn par_iter_mut(&mut self) -> impl IndexedParallelIterator<Item = FractionWriter<'_>> {
        let [coord0, coord1, coord2, coord3] =
            self.numerator.columns.each_mut().map(|s| &mut s.data);
        (coord0, coord1, coord2, coord3, &mut self.gen.denom.data)
            .into_par_iter()
            .map(|(n0, n1, n2, n3, d)| FractionWriter {
                numerator: [n0, n1, n2, n3],
                denom: d,
            })
    }
}

/// Exposes a writer for writing a fraction to a single index in a column.
// TODO(Ohad): iterate in chunks, consider VeryPacked.
pub struct FractionWriter<'a> {
    numerator: [&'a mut PackedBaseField; SECURE_EXTENSION_DEGREE],
    denom: &'a mut PackedSecureField,
}
impl FractionWriter<'_> {
    pub fn write_frac(self, numerator: PackedSecureField, denom: PackedSecureField) {
        debug_assert!(
            denom.to_array().iter().all(|x| *x != SecureField::zero()),
            "denom is zero {:?}",
            denom
        );
        let [c0, c1, c2, c3] = numerator.into_packed_m31s();
        *self.numerator[0] = c0;
        *self.numerator[1] = c1;
        *self.numerator[2] = c2;
        *self.numerator[3] = c3;
        *self.denom = denom;
    }
}

#[cfg(test)]
mod tests {
    use stwo_prover::core::backend::simd::m31::LOG_N_LANES;
    use stwo_prover::core::backend::simd::qm31::PackedSecureField;
    use stwo_prover::core::channel::Blake2sChannel;
    use stwo_prover::core::fields::m31::BaseField;
    use stwo_prover::core::fields::qm31::SecureField;
    use stwo_prover::core::fields::FieldExpOps;

    use super::LookupElements;
    use crate::logup::LogupTraceGenerator;
    use crate::{m31, qm31};

    #[test]
    fn test_lookup_elements_combine() {
        let mut channel = Blake2sChannel::default();
        let lookup_elements = LookupElements::<3>::draw(&mut channel);
        let values = [
            BaseField::from_u32_unchecked(123),
            BaseField::from_u32_unchecked(456),
            BaseField::from_u32_unchecked(789),
        ];

        assert_eq!(
            lookup_elements.combine::<BaseField, SecureField>(&values),
            BaseField::from_u32_unchecked(123)
                + BaseField::from_u32_unchecked(456) * lookup_elements.alpha
                + BaseField::from_u32_unchecked(789) * lookup_elements.alpha.pow(2)
                - lookup_elements.z
        );
    }

    #[test]
    fn test_frac_writer() {
        let expected_sum = (qm31!(1, 2, 3, 4) * qm31!(5, 6, 7, 8).inverse()) * m31!(1 << 6);

        let mut log_gen = LogupTraceGenerator::new(6);
        let mut col_gen = log_gen.new_col();
        for writer in col_gen.iter_mut() {
            let num = PackedSecureField::broadcast(qm31!(1, 2, 3, 4));
            let den = PackedSecureField::broadcast(qm31!(5, 6, 7, 8));
            writer.write_frac(num, den);
        }

        col_gen.finalize_col();
        let (_, sum) = log_gen.finalize_last();
        assert_eq!(sum, expected_sum);
    }

    #[test]
    fn test_col_from_iter() {
        let log_size = 8;
        let expected_sum = (qm31!(1, 2, 3, 4) * qm31!(5, 6, 7, 8).inverse()) * m31!(1 << log_size);

        let mut log_gen = LogupTraceGenerator::new(log_size);
        let col_iter = (0..1 << (log_size - LOG_N_LANES)).map(|_| {
            let num = PackedSecureField::broadcast(qm31!(1, 2, 3, 4));
            let den = PackedSecureField::broadcast(qm31!(5, 6, 7, 8));
            (num, den)
        });
        log_gen.col_from_iter(col_iter);

        let (_, sum) = log_gen.finalize_last();
        assert_eq!(sum, expected_sum);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_col_from_par_iter() {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        let log_size = 8;
        let expected_sum = (qm31!(1, 2, 3, 4) * qm31!(5, 6, 7, 8).inverse()) * m31!(1 << log_size);

        let mut log_gen = LogupTraceGenerator::new(log_size);
        let col_iter = (0..1 << (log_size - LOG_N_LANES)).into_par_iter().map(|_| {
            let num = PackedSecureField::broadcast(qm31!(1, 2, 3, 4));
            let den = PackedSecureField::broadcast(qm31!(5, 6, 7, 8));
            (num, den)
        });
        log_gen.col_from_par_iter(col_iter);

        let (_, sum) = log_gen.finalize_last();
        assert_eq!(sum, expected_sum);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_frac_writer() {
        use std::array;

        use rayon::prelude::*;
        // Sequential version.
        let mut log_gen_seq = LogupTraceGenerator::new(6);
        let mut col_gen_seq = log_gen_seq.new_col();
        col_gen_seq.iter_mut().enumerate().for_each(|(i, writer)| {
            let num = array::from_fn(|j| qm31!(i as u32, j as u32, 0, 1));
            let den = array::from_fn(|j| qm31!(i as u32, j as u32, 2, 3));
            let [num, den] = [num, den].map(PackedSecureField::from_array);
            writer.write_frac(num, den);
        });
        col_gen_seq.finalize_col();
        let (_, sum_seq) = log_gen_seq.finalize_last();

        // Parallel version.
        let mut log_gen_par = LogupTraceGenerator::new(6);
        let mut col_gen_par = log_gen_par.new_col();
        col_gen_par
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, writer)| {
                let num = array::from_fn(|j| qm31!(i as u32, j as u32, 0, 1));
                let den = array::from_fn(|j| qm31!(i as u32, j as u32, 2, 3));
                let [num, den] = [num, den].map(PackedSecureField::from_array);
                writer.write_frac(num, den);
            });
        col_gen_par.finalize_col();
        let (_, sum_par) = log_gen_par.finalize_last();

        assert_eq!(sum_seq, sum_par);
    }
}
