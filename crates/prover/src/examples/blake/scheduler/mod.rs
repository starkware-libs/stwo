mod constraints;
mod gen;

use constraints::BlakeSchedulerEval;
pub use gen::{gen_interaction_trace, gen_trace, BlakeInput};
use num_traits::Zero;

use super::round::RoundElements;
use super::N_ROUND_INPUT_FELTS;
use crate::constraint_framework::logup::{LogupAtRow, LookupElements};
use crate::constraint_framework::{EvalAtRow, FrameworkComponent, InfoEvaluator};
use crate::core::fields::qm31::SecureField;

pub type BlakeElements = LookupElements<N_ROUND_INPUT_FELTS>;

pub fn blake_scheduler_info() -> InfoEvaluator {
    let component = BlakeSchedulerComponent {
        log_size: 1,
        blake_lookup_elements: BlakeElements::dummy(),
        round_lookup_elements: RoundElements::dummy(),
        claimed_sum: SecureField::zero(),
    };
    component.evaluate(InfoEvaluator::default())
}

pub struct BlakeSchedulerComponent {
    pub log_size: u32,
    pub blake_lookup_elements: BlakeElements,
    pub round_lookup_elements: RoundElements,
    pub claimed_sum: SecureField,
}
impl FrameworkComponent for BlakeSchedulerComponent {
    fn log_size(&self) -> u32 {
        self.log_size
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }
    fn evaluate<E: EvalAtRow>(&self, eval: E) -> E {
        let blake_eval = BlakeSchedulerEval {
            eval,
            blake_lookup_elements: &self.blake_lookup_elements,
            round_lookup_elements: &self.round_lookup_elements,
            logup: LogupAtRow::new(1, self.claimed_sum, self.log_size),
        };
        blake_eval.eval()
    }
}

#[cfg(test)]
mod tests {
    use std::simd::Simd;

    use itertools::Itertools;

    use crate::constraint_framework::FrameworkComponent;
    use crate::core::poly::circle::CanonicCoset;
    use crate::examples::blake::round::RoundElements;
    use crate::examples::blake::scheduler::r#gen::{gen_interaction_trace, gen_trace, BlakeInput};
    use crate::examples::blake::scheduler::{BlakeElements, BlakeSchedulerComponent};

    #[test]
    fn test_blake_scheduler() {
        use crate::core::pcs::TreeVec;

        const LOG_SIZE: u32 = 10;

        let (trace, lookup_data, _round_inputs) = gen_trace(
            LOG_SIZE,
            &(0..(1 << LOG_SIZE))
                .map(|_| BlakeInput {
                    v: std::array::from_fn(|i| Simd::splat(i as u32)),
                    m: std::array::from_fn(|i| Simd::splat((i + 1) as u32)),
                })
                .collect_vec(),
        );

        let round_lookup_elements = RoundElements::dummy();
        let blake_lookup_elements = BlakeElements::dummy();
        let (interaction_trace, claimed_sum) = gen_interaction_trace(
            LOG_SIZE,
            lookup_data,
            &round_lookup_elements,
            &blake_lookup_elements,
        );

        let trace = TreeVec::new(vec![trace, interaction_trace]);
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
