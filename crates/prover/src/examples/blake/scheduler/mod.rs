mod constraints;
mod gen;

use constraints::BlakeSchedulerEval;
use num_traits::Zero;

use crate::constraint_framework::logup::{LogupAtRow, LookupElements};
use crate::constraint_framework::{EvalAtRow, FrameworkComponent, InfoEvaluator};
use crate::core::fields::qm31::SecureField;

pub fn blake_scheduler_info() -> InfoEvaluator {
    let component = BlakeSchedulerComponent {
        log_size: 1,
        blake_lookup_elements: LookupElements::dummy(16 * 3 * 2),
        round_lookup_elements: LookupElements::dummy(16 * 3 * 2),
        claimed_sum: SecureField::zero(),
    };
    component.evaluate(InfoEvaluator::default())
}

pub struct BlakeSchedulerComponent {
    pub log_size: u32,
    pub blake_lookup_elements: LookupElements,
    pub round_lookup_elements: LookupElements,
    pub claimed_sum: SecureField,
}
impl FrameworkComponent for BlakeSchedulerComponent {
    fn log_size(&self) -> u32 {
        self.log_size
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let [is_first] = eval.next_interaction_mask(2, [0]);
        let blake_eval = BlakeSchedulerEval {
            eval,
            blake_lookup_elements: &self.blake_lookup_elements,
            round_lookup_elements: &self.round_lookup_elements,
            logup: LogupAtRow::new(1, self.claimed_sum, is_first),
        };
        blake_eval.eval()
    }
}

#[cfg(test)]
mod tests {
    use std::simd::Simd;

    use itertools::Itertools;

    use crate::constraint_framework::constant_columns::gen_is_first;
    use crate::constraint_framework::logup::LookupElements;
    use crate::constraint_framework::FrameworkComponent;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use crate::core::poly::BitReversedOrder;
    use crate::examples::blake::scheduler::r#gen::{gen_interaction_trace, gen_trace, BlakeInput};
    use crate::examples::blake::scheduler::BlakeSchedulerComponent;
    use crate::examples::blake::{XorAccums, XorLookupElements};

    #[test]
    fn test_blake_scheduler() {
        use crate::core::pcs::TreeVec;

        const LOG_SIZE: u32 = 10;

        let (trace, lookup_data, _round_inputs) = gen_trace(
            LOG_SIZE,
            &(0..(1 << LOG_SIZE))
                .map(|i| BlakeInput {
                    v: std::array::from_fn(|i| Simd::splat(i as u32)),
                    m: std::array::from_fn(|i| Simd::splat((i + 1) as u32)),
                })
                .collect_vec(),
        );

        let round_lookup_elements = LookupElements::dummy(16 * 3 * 2);
        let blake_lookup_elements = LookupElements::dummy(16 * 3 * 2);
        let (interaction_trace, claimed_sum) = gen_interaction_trace(
            LOG_SIZE,
            lookup_data,
            &round_lookup_elements,
            &blake_lookup_elements,
        );

        let trace = TreeVec::new(vec![trace, interaction_trace, vec![gen_is_first(LOG_SIZE)]]);
        let trace_polys = trace.map_cols(|c| c.interpolate());

        let component = BlakeSchedulerComponent {
            log_size: LOG_SIZE,
            blake_lookup_elements,
            round_lookup_elements,
            claimed_sum,
        };
        crate::constraint_framework::assert_constraints(
            &trace_polys,
            CanonicCoset::new(LOG_SIZE),
            |eval| {
                component.evaluate(eval);
            },
        )
    }
}
