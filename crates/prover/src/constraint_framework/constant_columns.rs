use num_traits::One;

use crate::core::backend::{Backend, Col, Column};
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::{bit_reverse_index, coset_index_to_circle_domain_index};

/// Generates a column with a single one at the first position, and zeros elsewhere.
pub fn gen_is_first<B: Backend>(log_size: u32) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
    let mut col = Col::<B, BaseField>::zeros(1 << log_size);
    col.set(0, BaseField::one());
    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

/// Generates a column with a single one at every `2^log_step` positions, and zero elsewhere.
// TODO(andrew): Consider optimizing. Is a quotients of two coset_vanishing (use succinct rep for
// verifier).
pub fn gen_is_step_multiple<B: Backend>(
    log_size: u32,
    log_step: u32,
) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
    let mut col = Col::<B, BaseField>::zeros(1 << log_size);

    for i in (0..1 << log_size).step_by(1 << log_step) {
        let circle_domain_index = coset_index_to_circle_domain_index(i, log_size);
        let circle_domain_index_bit_rev = bit_reverse_index(circle_domain_index, log_size);
        col.set(circle_domain_index_bit_rev, BaseField::one());
    }

    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}
