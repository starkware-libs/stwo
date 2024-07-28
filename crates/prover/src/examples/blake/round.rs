use num_traits::Zero;

use super::round_constraints::BlakeRoundEval;
use super::XorLookupElements;
use crate::constraint_framework::logup::{LogupAtRow, LookupElements};
use crate::constraint_framework::{EvalAtRow, FrameworkComponent, InfoEvaluator};
use crate::core::fields::qm31::SecureField;

pub fn blake_round_info() -> InfoEvaluator {
    let component = BlakeRoundComponent {
        log_size: 1,
        xor_lookup_elements: XorLookupElements {
            xor12: LookupElements::dummy(3),
            xor9: LookupElements::dummy(3),
            xor8: LookupElements::dummy(3),
            xor7: LookupElements::dummy(3),
            xor4: LookupElements::dummy(3),
        },
        round_lookup_elements: LookupElements::dummy(16 * 3 * 2),
        claimed_sum: SecureField::zero(),
    };
    component.evaluate(InfoEvaluator::default())
}

pub struct BlakeRoundComponent {
    pub log_size: u32,
    pub xor_lookup_elements: XorLookupElements,
    pub round_lookup_elements: LookupElements,
    pub claimed_sum: SecureField,
}

impl FrameworkComponent for BlakeRoundComponent {
    fn log_size(&self) -> u32 {
        self.log_size
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let [is_first] = eval.next_interaction_mask(2, [0]);
        let blake_eval = BlakeRoundEval {
            eval,
            xor_lookup_elements: &self.xor_lookup_elements,
            round_lookup_elements: &self.round_lookup_elements,
            logup: LogupAtRow::new(1, self.claimed_sum, is_first),
        };
        blake_eval.eval()
    }
}
