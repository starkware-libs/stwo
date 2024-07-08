use std::ops::{Add, Mul, Sub};

use itertools::Itertools;
use num_traits::{One, Zero};

use super::EvalAtRow;
use crate::core::backend::simd::column::SecureColumn;
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::prefix_sum::inclusive_prefix_sum;
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumnByCoords;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::shifted_secure_combination;
use crate::core::ColumnVec;

pub struct LogupAtRow<const BATCH_SIZE: usize, E: EvalAtRow> {
    pub interaction: usize,
    pub queue: [(E::EF, E::EF); BATCH_SIZE],
    pub queue_size: usize,
    pub claimed_sum: SecureField,
    pub prev_mask: E::EF,
    pub is_first: E::F,
}
impl<const BATCH_SIZE: usize, E: EvalAtRow> LogupAtRow<BATCH_SIZE, E> {
    pub fn new(interaction: usize, claimed_sum: SecureField, is_first: E::F) -> Self {
        Self {
            interaction,
            queue: [(E::EF::zero(), E::EF::zero()); BATCH_SIZE],
            queue_size: 0,
            claimed_sum,
            prev_mask: E::EF::zero(),
            is_first,
        }
    }
    pub fn push_lookup(
        &mut self,
        eval: &mut E,
        numerator: E::EF,
        values: &[E::F],
        lookup_elements: LookupElements,
    ) {
        let shifted_value = shifted_secure_combination(
            values,
            E::EF::zero() + lookup_elements.alpha,
            E::EF::zero() + lookup_elements.z,
        );
        self.push_frac(eval, numerator, shifted_value);
    }

    pub fn push_frac(&mut self, eval: &mut E, p: E::EF, q: E::EF) {
        if self.queue_size < BATCH_SIZE {
            self.queue[self.queue_size] = (p, q);
            self.queue_size += 1;
            return;
        }

        // Compute sum_i pi/qi over batch, as a fraction, p/q.
        let (num, denom) = self
            .queue
            .iter()
            .copied()
            .fold((E::EF::zero(), E::EF::one()), |(p0, q0), (pi, qi)| {
                (p0 * qi + pi * q0, qi * q0)
            });

        self.queue[0] = (p, q);
        self.queue_size = 1;

        // Add a constraint that p / q = diff.
        let cur = E::combine_ef(std::array::from_fn(|_| {
            eval.next_interaction_mask(1, [0])[0]
        }));
        let diff = cur - self.prev_mask;
        self.prev_mask = cur;
        eval.add_constraint(diff * denom - num);
    }

    pub fn finalize(self, eval: &mut E) {
        let (p, q) = self.queue[0..self.queue_size]
            .iter()
            .copied()
            .fold((E::EF::zero(), E::EF::one()), |(p0, q0), (pi, qi)| {
                (p0 * qi + pi * q0, qi * q0)
            });

        let cumulative_mask_values =
            std::array::from_fn(|_| eval.next_interaction_mask(self.interaction, [0, -1]));
        let cur = E::combine_ef(cumulative_mask_values.map(|[cur, _prev]| cur));
        let up = E::combine_ef(cumulative_mask_values.map(|[_cur, prev]| prev));
        let up = up - self.is_first * self.claimed_sum;
        let diff = cur - up - self.prev_mask;

        eval.add_constraint(diff * q - p);
    }
}

/// Interaction elements for the logup protocol.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LookupElements {
    pub z: SecureField,
    pub alpha: SecureField,
}
impl LookupElements {
    pub fn draw(channel: &mut Blake2sChannel) -> Self {
        let [z, alpha] = channel.draw_felts(2).try_into().unwrap();
        Self { z, alpha }
    }
    pub fn combine<F: Copy, EF>(&self, values: &[F]) -> EF
    where
        EF: Copy
            + Zero
            + Mul<EF, Output = EF>
            + Add<F, Output = EF>
            + Sub<EF, Output = EF>
            + From<SecureField>,
    {
        shifted_secure_combination(values, EF::from(self.alpha), EF::from(self.z))
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
        // Prefix sum the last column.
        let last_col_coords = self.trace.pop().unwrap().columns;
        let coord_prefix_sum = last_col_coords.map(inclusive_prefix_sum);
        self.trace.push(SecureColumnByCoords {
            columns: coord_prefix_sum,
        });
        let claimed_sum = self.trace.last().unwrap().at(1);

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
