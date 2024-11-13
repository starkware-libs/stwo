use std::ops::{Mul, Sub};

use itertools::Itertools;
use num_traits::{One, Zero};

use super::RelationEFTraitBound;
use crate::core::backend::simd::column::SecureColumn;
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::prefix_sum::inclusive_prefix_sum;
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::channel::Channel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumnByCoords;
use crate::core::fields::FieldExpOps;
use crate::core::lookups::utils::Fraction;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::{bit_reverse_index, coset_index_to_circle_domain_index};
use crate::core::ColumnVec;

/// Represents the value of the prefix sum column at some index.
/// Should be used to eliminate padded rows for the logup sum.
pub type ClaimedPrefixSum = (SecureField, usize);
// (total_sum, claimed_sum)
pub type LogupSums = (SecureField, Option<ClaimedPrefixSum>);

/// Evaluates constraints for batched logups.
/// These constraint enforce the sum of multiplicity_i / (z + sum_j alpha^j * x_j) = claimed_sum.
pub struct LogupAtRow<F: Clone, EF: RelationEFTraitBound<F>> {
    /// The index of the interaction used for the cumulative sum columns.
    pub interaction: usize,
    /// The total sum of all the fractions.
    pub total_sum: SecureField,
    /// The claimed sum of the relevant fractions.
    /// This is used for padding the component with default rows. Padding should be in bit-reverse.
    /// None if the claimed_sum is the total_sum.
    pub claimed_sum: Option<ClaimedPrefixSum>,
    /// The evaluation of the last cumulative sum column.
    pub prev_col_cumsum: EF,
    pub cur_frac: Option<Fraction<EF, EF>>,
    pub is_finalized: bool,
    /// The value of the `is_first` constant column at current row.
    /// See [`super::preprocessed_columns::gen_is_first()`].
    pub is_first: F,
    pub log_size: u32,
}

impl<F: Clone + Zero, EF: RelationEFTraitBound<F>> Default for LogupAtRow<F, EF> {
    fn default() -> Self {
        Self::dummy()
    }
}
impl<F: Clone + Zero, EF: RelationEFTraitBound<F>> LogupAtRow<F, EF> {
    pub fn new(
        interaction: usize,
        total_sum: SecureField,
        claimed_sum: Option<ClaimedPrefixSum>,
        log_size: u32,
    ) -> Self {
        Self {
            interaction,
            total_sum,
            claimed_sum,
            prev_col_cumsum: EF::zero(),
            cur_frac: None,
            is_finalized: true,
            is_first: F::zero(),
            log_size,
        }
    }

    // TODO(alont): Remove this once unnecessary LogupAtRows are gone.
    pub fn dummy() -> Self {
        Self {
            interaction: 100,
            total_sum: SecureField::one(),
            claimed_sum: None,
            prev_col_cumsum: EF::zero(),
            cur_frac: None,
            is_finalized: true,
            is_first: F::zero(),
            log_size: 10,
        }
    }
}

/// Ensures that the LogupAtRow is finalized.
/// LogupAtRow should be finalized exactly once.
impl<F: Clone, EF: RelationEFTraitBound<F>> Drop for LogupAtRow<F, EF> {
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
            z: SecureField::one(),
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
    /// Preallocated buffer for the Inverses of the denominators.
    denom_inv: SecureColumn,
}
impl LogupTraceGenerator {
    pub fn new(log_size: u32) -> Self {
        let trace = vec![];
        let denom = SecureColumn::zeros(1 << log_size);
        let denom_inv = SecureColumn::zeros(1 << log_size);
        Self {
            log_size,
            trace,
            denom,
            denom_inv,
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

    /// Finalize the trace. Returns the trace and the total sum of the last column.
    pub fn finalize_last(
        self,
    ) -> (
        ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        SecureField,
    ) {
        let log_size = self.log_size;
        let (trace, [total_sum]) = self.finalize_at([(1 << log_size) - 1]);
        (trace, total_sum)
    }

    /// Finalize the trace. Returns the trace and the prefix sum of the last column at
    /// the corresponding `indices`.
    pub fn finalize_at<const N: usize>(
        mut self,
        indices: [usize; N],
    ) -> (
        ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        [SecureField; N],
    ) {
        // Prefix sum the last column.
        let last_col_coords = self.trace.pop().unwrap().columns;
        let coord_prefix_sum = last_col_coords.map(inclusive_prefix_sum);
        let secure_prefix_sum = SecureColumnByCoords {
            columns: coord_prefix_sum,
        };
        let returned_prefix_sums = indices.map(|idx| {
            // Prefix sum column is in bit-reversed circle domain order.
            let fixed_index = bit_reverse_index(
                coset_index_to_circle_domain_index(idx, self.log_size),
                self.log_size,
            );
            secure_prefix_sum.at(fixed_index)
        });
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
        (trace, returned_prefix_sums)
    }
}

/// Trace generator for a single lookup column.
pub struct LogupColGenerator<'a> {
    gen: &'a mut LogupTraceGenerator,
    /// Numerator expressions (i.e. multiplicities) being generated for the current lookup.
    numerator: SecureColumnByCoords<SimdBackend>,
}
impl<'a> LogupColGenerator<'a> {
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
        FieldExpOps::batch_inverse(&self.gen.denom.data, &mut self.gen.denom_inv.data);

        for vec_row in 0..(1 << (self.gen.log_size - LOG_N_LANES)) {
            unsafe {
                let value = self.numerator.packed_at(vec_row)
                    * *self.gen.denom_inv.data.get_unchecked(vec_row);
                let prev_value = self
                    .gen
                    .trace
                    .last()
                    .map(|col| col.packed_at(vec_row))
                    .unwrap_or_else(PackedSecureField::zero);
                self.numerator.set_packed(vec_row, value + prev_value)
            };
        }

        self.gen.trace.push(self.numerator)
    }
}

#[cfg(test)]
mod tests {
    use super::LookupElements;
    use crate::core::channel::Blake2sChannel;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::FieldExpOps;

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
}
