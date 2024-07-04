use num_traits::{One, Zero};

use super::eval::EvalAtRow;
use crate::core::fields::qm31::SecureField;

// TODO: Finalize.

pub struct LogupAtRow<'a, E: EvalAtRow> {
    pub interaction: usize,
    pub batch_size: usize,
    pub queue: Vec<(E::EF, E::EF)>,
    pub claimed_sums: &'a [SecureField],
    pub is_first: E::F,
}
impl<'a, E: EvalAtRow> LogupAtRow<'a, E> {
    pub fn new(
        interaction: usize,
        batch_size: usize,
        claimed_sums: &'a [SecureField],
        is_first: E::F,
    ) -> Self {
        Self {
            interaction,
            batch_size,
            queue: Vec::with_capacity(batch_size),
            claimed_sums,
            is_first,
        }
    }
    pub fn push(&mut self, eval: &mut E, p: E::EF, q: E::EF) {
        self.queue.push((p, q));
        if self.queue.len() < self.batch_size {
            return;
        }
        let claimed_sum = self.claimed_sums[0];
        self.claimed_sums = &self.claimed_sums[1..];

        // Compute sum_i pi/qi over batch, as a fraction, p/q.
        let (p, q) = std::mem::take(&mut self.queue)
            .into_iter()
            .fold((E::EF::zero(), E::EF::one()), |(p0, q0), (pi, qi)| {
                (p0 * qi + pi * q0, qi * q0)
            });

        // Add a constraint that p / q = diff.
        let cumulative_mask_values =
            std::array::from_fn(|_| eval.next_interaction_mask(self.interaction, [0, -1]));
        let cur = E::combine_ef(cumulative_mask_values.map(|[cur, _prev]| cur));
        let prev = E::combine_ef(cumulative_mask_values.map(|[_cur, prev]| prev));
        let prev = prev - self.is_first * claimed_sum;
        let diff = cur - prev;

        eval.add_constraint(diff * q - p);
    }
}
