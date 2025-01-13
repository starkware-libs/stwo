mod constraints;
mod gen;

pub use gen::{generate_interaction_trace, generate_trace, BlakeRoundInput};
use num_traits::Zero;

use super::{BlakeXorElements, N_ROUND_INPUT_FELTS};
use crate::constraint_framework::{
    relation, EvalAtRow, FrameworkComponent, FrameworkEval, InfoEvaluator,
};
use crate::core::fields::qm31::SecureField;

pub type BlakeRoundComponent = FrameworkComponent<BlakeRoundEval>;

relation!(RoundElements, N_ROUND_INPUT_FELTS);

pub struct BlakeRoundEval {
    pub log_size: u32,
    pub xor_lookup_elements: BlakeXorElements,
    pub round_lookup_elements: RoundElements,
    pub claimed_sum: SecureField,
}

impl FrameworkEval for BlakeRoundEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }
    fn evaluate<E: EvalAtRow>(&self, eval: E) -> E {
        let blake_eval = constraints::BlakeRoundEval {
            eval,
            xor_lookup_elements: &self.xor_lookup_elements,
            round_lookup_elements: &self.round_lookup_elements,
            _claimed_sum: self.claimed_sum,
            _log_size: self.log_size,
        };
        blake_eval.eval()
    }
}

pub fn blake_round_info() -> InfoEvaluator {
    let component = BlakeRoundEval {
        log_size: 1,
        xor_lookup_elements: BlakeXorElements::dummy(),
        round_lookup_elements: RoundElements::dummy(),
        claimed_sum: SecureField::zero(),
    };
    component.evaluate(InfoEvaluator::empty())
}

#[cfg(test)]
mod tests {
    use std::simd::Simd;

    use itertools::Itertools;

    use crate::constraint_framework::preprocessed_columns::IsFirst;
    use crate::constraint_framework::FrameworkEval;
    use crate::core::poly::circle::CanonicCoset;
    use crate::examples::blake::round::r#gen::{
        generate_interaction_trace, generate_trace, BlakeRoundInput,
    };
    use crate::examples::blake::round::{BlakeRoundEval, RoundElements};
    use crate::examples::blake::{BlakeXorElements, XorAccums};

    #[test]
    fn test_blake_round() {
        use crate::core::pcs::TreeVec;

        const LOG_SIZE: u32 = 10;

        let mut xor_accum = XorAccums::default();
        let (trace, lookup_data) = generate_trace(
            LOG_SIZE,
            &(0..(1 << LOG_SIZE))
                .map(|_| BlakeRoundInput {
                    v: std::array::from_fn(|i| Simd::splat(i as u32)),
                    m: std::array::from_fn(|i| Simd::splat((i + 1) as u32)),
                })
                .collect_vec(),
            &mut xor_accum,
        );

        let xor_lookup_elements = BlakeXorElements::dummy();
        let round_lookup_elements = RoundElements::dummy();
        let (interaction_trace, claimed_sum) = generate_interaction_trace(
            LOG_SIZE,
            lookup_data,
            &xor_lookup_elements,
            &round_lookup_elements,
        );

        let trace = TreeVec::new(vec![
            vec![IsFirst::new(LOG_SIZE).gen_column_simd()],
            trace,
            interaction_trace,
        ]);
        let trace_polys = trace.map_cols(|c| c.interpolate());

        let component = BlakeRoundEval {
            log_size: LOG_SIZE,
            xor_lookup_elements,
            round_lookup_elements,
            claimed_sum,
        };
        crate::constraint_framework::assert_constraints(
            &trace_polys,
            CanonicCoset::new(LOG_SIZE),
            |eval| {
                component.evaluate(eval);
            },
            claimed_sum,
        )
    }
}
