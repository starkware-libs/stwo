use num_traits::{One, Zero};

use super::eval::EvalAtRow;
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::fields::qm31::SecureField;
use crate::core::utils::shifted_secure_combination;

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

#[derive(Copy, Clone, Debug)]
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
