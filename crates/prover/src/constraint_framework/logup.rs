use std::ops::{Mul, Sub};

use itertools::Itertools;
use num_traits::{One, Zero};

use super::EvalAtRow;
use crate::core::backend::simd::column::SecureColumn;
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::prefix_sum::inclusive_prefix_sum;
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::channel::Channel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::{SecureColumnByCoords, SECURE_EXTENSION_DEGREE};
use crate::core::fields::FieldExpOps;
use crate::core::lookups::utils::Fraction;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

/// Evaluates constraints for batched logups.
/// These constraint enforce the sum of multiplicity_i / (z + sum_j alpha^j * x_j) = claimed_sum.
/// BATCH_SIZE is the number of fractions to batch together. The degree of the resulting constraints
/// will be BATCH_SIZE + 1.
pub struct LogupAtRow<const BATCH_SIZE: usize, E: EvalAtRow> {
    /// The index of the interaction used for the cumulative sum columns.
    pub interaction: usize,
    /// Queue of fractions waiting to be batched together.
    pub queue: [(E::EF, E::EF); BATCH_SIZE],
    /// Number of fractions in the queue.
    pub queue_size: usize,
    /// A constant to subtract from each row, to make the totall sum of the last column zero.
    /// In other words, claimed_sum / 2^log_size.
    /// This is used to make the constraint uniform.
    pub cumsum_shift: SecureField,
    /// The evaluation of the last cumulative sum column.
    pub prev_col_cumsum: E::EF,
    is_finalized: bool,
}
impl<const BATCH_SIZE: usize, E: EvalAtRow> LogupAtRow<BATCH_SIZE, E> {
    pub fn new(interaction: usize, claimed_sum: SecureField, log_size: u32) -> Self {
        Self {
            interaction,
            queue: [(E::EF::zero(), E::EF::zero()); BATCH_SIZE],
            queue_size: 0,
            cumsum_shift: claimed_sum / BaseField::from_u32_unchecked(1 << log_size),
            prev_col_cumsum: E::EF::zero(),
            is_finalized: false,
        }
    }
    pub fn push_lookup<const N: usize>(
        &mut self,
        eval: &mut E,
        numerator: E::EF,
        values: &[E::F],
        lookup_elements: &LookupElements<N>,
    ) {
        let shifted_value = lookup_elements.combine(values);
        self.push_frac(eval, numerator, shifted_value);
    }

    pub fn push_frac(&mut self, eval: &mut E, numerator: E::EF, denominator: E::EF) {
        if self.queue_size < BATCH_SIZE {
            self.queue[self.queue_size] = (numerator, denominator);
            self.queue_size += 1;
            return;
        }

        // Compute sum_i pi/qi over batch, as a fraction, num/denom.
        let (num, denom) = self.fold_queue();

        self.queue[0] = (numerator, denominator);
        self.queue_size = 1;

        // Add a constraint that num / denom = diff.
        let cur_cumsum = eval.next_extension_interaction_mask(self.interaction, [0])[0];
        let diff = cur_cumsum - self.prev_col_cumsum;
        self.prev_col_cumsum = cur_cumsum;
        eval.add_constraint(diff * denom - num);
    }

    pub fn add_frac(&mut self, eval: &mut E, fraction: Fraction<E::EF, E::EF>) {
        // Add a constraint that num / denom = diff.
        let cur_cumsum = eval.next_extension_interaction_mask(self.interaction, [0])[0];
        let diff = cur_cumsum - self.prev_col_cumsum;
        self.prev_col_cumsum = cur_cumsum;
        eval.add_constraint(diff * fraction.denominator - fraction.numerator);
    }

    pub fn finalize(mut self, eval: &mut E) {
        assert!(!self.is_finalized, "LogupAtRow was already finalized");
        let (num, denom) = self.fold_queue();

        let [cur_cumsum, prev_row_cumsum] =
            eval.next_extension_interaction_mask(self.interaction, [0, -1]);

        let diff = cur_cumsum - prev_row_cumsum - self.prev_col_cumsum;
        // Instead of checking diff = num / denom, check diff = num / denom - cumsum_shift.
        // This makes (num / denom - cumsum_shift) have sum zero, which makes the constraint
        // uniform - apply on all rows.
        let fixed_diff = diff + self.cumsum_shift;

        eval.add_constraint(fixed_diff * denom - num);

        self.is_finalized = true;
    }

    fn fold_queue(&self) -> (E::EF, E::EF) {
        self.queue[0..self.queue_size]
            .iter()
            .copied()
            .fold((E::EF::zero(), E::EF::one()), |(p0, q0), (pi, qi)| {
                (p0 * qi + pi * q0, qi * q0)
            })
    }
}

/// Ensures that the LogupAtRow is finalized.
/// LogupAtRow should be finalized exactly once.
impl<const BATCH_SIZE: usize, E: EvalAtRow> Drop for LogupAtRow<BATCH_SIZE, E> {
    fn drop(&mut self) {
        assert!(self.is_finalized, "LogupAtRow was not finalized");
    }
}

/// Interaction elements for the logup protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupElements<const N: usize> {
    pub z: SecureField,
    pub alpha: SecureField,
    alpha_powers: [SecureField; N],
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
    pub fn combine<F: Copy, EF>(&self, values: &[F]) -> EF
    where
        EF: Copy + Zero + From<F> + From<SecureField> + Mul<F, Output = EF> + Sub<EF, Output = EF>,
    {
        EF::from(values[0])
            + values[1..]
                .iter()
                .zip(self.alpha_powers.iter())
                .fold(EF::zero(), |acc, (&value, &power)| {
                    acc + EF::from(power) * value
                })
            - EF::from(self.z)
    }
    // TODO(spapini): Try to remove this.
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

    /// Finalize the trace. Returns the trace and the claimed sum of the last column.
    pub fn finalize(
        mut self,
    ) -> (
        ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        SecureField,
    ) {
        // Compute claimed sum.
        let mut last_col_coords = self.trace.pop().unwrap().columns;
        let packed_sums: [PackedBaseField; SECURE_EXTENSION_DEGREE] = last_col_coords
            .each_ref()
            .map(|c| c.data.iter().copied().sum());
        let base_sums = packed_sums.map(|s| s.pointwise_sum());
        let claimed_sum = SecureField::from_m31_array(base_sums);

        // Shift the last column to make the sum zero.
        let cumsum_shift = claimed_sum / BaseField::from_u32_unchecked(1 << self.log_size);
        last_col_coords.iter_mut().enumerate().for_each(|(i, c)| {
            c.data
                .iter_mut()
                .for_each(|x| *x -= PackedBaseField::broadcast(cumsum_shift.to_m31_array()[i]))
        });

        // Prefix sum the last column.
        let coord_prefix_sum = last_col_coords.map(inclusive_prefix_sum);
        self.trace.push(SecureColumnByCoords {
            columns: coord_prefix_sum,
        });

        let trace = self
            .trace
            .into_iter()
            .flat_map(|eval| {
                eval.columns.map(|c| {
                    CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(
                        CanonicCoset::new(self.log_size).circle_domain(),
                        c,
                    )
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
    use num_traits::One;

    use super::LogupAtRow;
    use crate::constraint_framework::InfoEvaluator;
    use crate::core::fields::qm31::SecureField;

    #[test]
    #[should_panic]
    fn test_logup_not_finalized_panic() {
        let mut logup = LogupAtRow::<2, InfoEvaluator>::new(1, SecureField::one(), 7);
        logup.push_frac(
            &mut InfoEvaluator::default(),
            SecureField::one(),
            SecureField::one(),
        );
    }
}
