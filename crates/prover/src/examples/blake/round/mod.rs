mod constraints;
mod gen;

pub use gen::{generate_interaction_trace, generate_trace, BlakeRoundInput};
use num_traits::Zero;

use super::{BlakeXorElements, N_ROUND_INPUT_FELTS};
use crate::constraint_framework::logup::{LogupAtRow, LookupElements};
use crate::constraint_framework::preprocessed_columns::PreprocessedColumn;
use crate::constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, InfoEvaluator};
use crate::core::fields::qm31::SecureField;

pub type BlakeRoundComponent = FrameworkComponent<BlakeRoundEval>;

pub type RoundElements = LookupElements<N_ROUND_INPUT_FELTS>;

use crate::constraint_framework::INTERACTION_TRACE_IDX;

pub struct BlakeRoundEval {
    pub log_size: u32,
    pub xor_lookup_elements: BlakeXorElements,
    pub round_lookup_elements: RoundElements,
    pub total_sum: SecureField,
}

impl FrameworkEval for BlakeRoundEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let is_first = eval.get_preprocessed_column(PreprocessedColumn::IsFirst(self.log_size()));
        let blake_eval = constraints::BlakeRoundEval {
            eval,
            xor_lookup_elements: &self.xor_lookup_elements,
            round_lookup_elements: &self.round_lookup_elements,
            logup: LogupAtRow::new(INTERACTION_TRACE_IDX, self.total_sum, None, is_first),
        };
        blake_eval.eval()
    }
}

pub fn blake_round_info() -> InfoEvaluator {
    let component = BlakeRoundEval {
        log_size: 1,
        xor_lookup_elements: BlakeXorElements::dummy(),
        round_lookup_elements: RoundElements::dummy(),
        total_sum: SecureField::zero(),
    };
    component.evaluate(InfoEvaluator::default())
}

#[cfg(test)]
mod tests {
    use std::simd::Simd;

    use itertools::Itertools;

    use crate::constraint_framework::preprocessed_columns::gen_is_first;
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
        let (interaction_trace, total_sum) = generate_interaction_trace(
            LOG_SIZE,
            lookup_data,
            &xor_lookup_elements,
            &round_lookup_elements,
        );

        let trace = TreeVec::new(vec![trace, interaction_trace, vec![gen_is_first(LOG_SIZE)]]);
        let trace_polys = trace.map_cols(|c| c.interpolate());

        let component = BlakeRoundEval {
            log_size: LOG_SIZE,
            xor_lookup_elements,
            round_lookup_elements,
            total_sum,
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
