mod constraints;
mod gen;

use constraints::eval_blake_scheduler_constraints;
pub use gen::{gen_interaction_trace, gen_trace, BlakeInput};
use num_traits::Zero;

use super::round::RoundElements;
use super::N_ROUND_INPUT_FELTS;
use crate::constraint_framework::{
    relation, EvalAtRow, FrameworkComponent, FrameworkEval, InfoEvaluator,
};
use crate::core::fields::qm31::SecureField;

pub type BlakeSchedulerComponent = FrameworkComponent<BlakeSchedulerEval>;

relation!(BlakeElements, N_ROUND_INPUT_FELTS);

pub struct BlakeSchedulerEval {
    pub log_size: u32,
    pub blake_lookup_elements: BlakeElements,
    pub round_lookup_elements: RoundElements,
    pub total_sum: SecureField,
}
impl FrameworkEval for BlakeSchedulerEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        eval_blake_scheduler_constraints(
            &mut eval,
            &self.blake_lookup_elements,
            &self.round_lookup_elements,
        );
        eval
    }
}

pub fn blake_scheduler_info() -> InfoEvaluator {
    let component = BlakeSchedulerEval {
        log_size: 1,
        blake_lookup_elements: BlakeElements::dummy(),
        round_lookup_elements: RoundElements::dummy(),
        total_sum: SecureField::zero(),
    };
    component.evaluate(InfoEvaluator::empty())
}

#[cfg(test)]
mod tests {
    use std::simd::Simd;

    use itertools::Itertools;

    use crate::constraint_framework::preprocessed_columns::gen_is_first;
    use crate::constraint_framework::FrameworkEval;
    use crate::core::poly::circle::CanonicCoset;
    use crate::examples::blake::round::RoundElements;
    use crate::examples::blake::scheduler::r#gen::{gen_interaction_trace, gen_trace, BlakeInput};
    use crate::examples::blake::scheduler::{BlakeElements, BlakeSchedulerEval};

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
        let (interaction_trace, total_sum) = gen_interaction_trace(
            LOG_SIZE,
            lookup_data,
            &round_lookup_elements,
            &blake_lookup_elements,
        );

        let trace = TreeVec::new(vec![vec![gen_is_first(LOG_SIZE)], trace, interaction_trace]);
        let trace_polys = trace.map_cols(|c| c.interpolate());

        let component = BlakeSchedulerEval {
            log_size: LOG_SIZE,
            blake_lookup_elements,
            round_lookup_elements,
            total_sum,
        };
        crate::constraint_framework::assert_constraints(
            &trace_polys,
            CanonicCoset::new(LOG_SIZE),
            |eval| {
                component.evaluate(eval);
            },
            (total_sum, None),
        )
    }
}
