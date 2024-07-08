use itertools::Itertools;
use num_traits::{One, Zero};
use tracing::{span, Level};

use super::EvalAtRow;
use crate::core::backend::simd::column::SecureFieldVec;
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Backend, Column};
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::{bit_reverse_index, shifted_secure_combination};
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
}

// SIMD backend generator.
pub struct LogupTraceGenerator {
    log_size: u32,
    trace: Vec<SecureColumn<SimdBackend>>,
    denom: SecureFieldVec,
    denom_inv: SecureFieldVec,
}
impl LogupTraceGenerator {
    pub fn new(log_size: u32) -> Self {
        let trace = vec![];
        let denom = SecureFieldVec::zeros(1 << log_size);
        let denom_inv = SecureFieldVec::zeros(1 << log_size);
        Self {
            log_size,
            trace,
            denom,
            denom_inv,
        }
    }

    pub fn new_col(&mut self) -> LogupColGenerator<'_> {
        let log_size = self.log_size;
        LogupColGenerator {
            gen: self,
            numerator: SecureColumn::<SimdBackend>::zeros(1 << log_size),
        }
    }

    pub fn finalize(
        mut self,
    ) -> (
        ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        SecureField,
    ) {
        let claimed_xor_sum = eval_order_prefix_sum(self.trace.last_mut().unwrap(), self.log_size);

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
        (trace, claimed_xor_sum)
    }
}

pub struct LogupColGenerator<'a> {
    gen: &'a mut LogupTraceGenerator,
    numerator: SecureColumn<SimdBackend>,
}
impl<'a> LogupColGenerator<'a> {
    pub fn write_frac(&mut self, vec_row: usize, p: PackedSecureField, q: PackedSecureField) {
        unsafe {
            self.numerator.set_packed(vec_row, p);
            *self.gen.denom.data.get_unchecked_mut(vec_row) = q;
        }
    }

    pub fn finalize_col(mut self) {
        FieldExpOps::batch_inverse(&self.gen.denom.data, &mut self.gen.denom_inv.data);

        #[allow(clippy::needless_range_loop)]
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

// TODO(spapini): Consider adding optional Ops.
pub fn eval_order_prefix_sum<B: Backend>(col: &mut SecureColumn<B>, log_size: u32) -> SecureField {
    let _span = span!(Level::INFO, "Prefix sum").entered();

    let mut cur = SecureField::zero();
    for i in 0..(1 << log_size) {
        let index = if i & 1 == 0 {
            i / 2
        } else {
            (1 << (log_size - 1)) + ((1 << log_size) - 1 - i) / 2
        };
        let index = bit_reverse_index(index, log_size);
        cur += col.at(index);
        col.set(index, cur);
    }
    cur
}
