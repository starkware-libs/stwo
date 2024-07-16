use num_traits::Zero;
use tracing::{span, Level};

use super::scheduler_constraints::BlakeSchedulerEval;
use super::DomainEvalHelper;
use crate::constraint_framework::logup::{LogupAtRow, LookupElements};
use crate::constraint_framework::{DomainEvaluator, EvalAtRow, InfoEvaluator, PointEvaluator};
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentProver, ComponentTrace};
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::SimdBackend;
use crate::core::circle::CirclePoint;
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::CanonicCoset;
use crate::core::{ColumnVec, InteractionElements, LookupValues};

pub fn blake_scheduler_info() -> InfoEvaluator {
    let mut counter = BlakeSchedulerEval {
        eval: InfoEvaluator::default(),
        blake_lookup_elements: &LookupElements::dummy(16 * 3 * 2),
        round_lookup_elements: &LookupElements::dummy(16 * 3 * 2),
        logup: LogupAtRow::new(1, SecureField::zero(), BaseField::zero()),
    };
    // Constant column.
    counter.eval.next_interaction_mask(2, [0]);
    counter.eval()
}

pub struct BlakeSchedulerComponent {
    pub log_size: u32,
    pub blake_lookup_elements: LookupElements,
    pub round_lookup_elements: LookupElements,
    pub claimed_sum: SecureField,
}
impl Component for BlakeSchedulerComponent {
    fn n_constraints(&self) -> usize {
        let counter = blake_scheduler_info();
        counter.n_constraints
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    fn n_interaction_phases(&self) -> u32 {
        1
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::new(
            blake_scheduler_info()
                .mask_offsets
                .iter()
                .map(|tree_masks| vec![self.log_size; tree_masks.len()])
                .collect(),
        )
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let info = blake_scheduler_info();
        let trace_step = CanonicCoset::new(self.log_size).step();
        info.mask_offsets.map(|tree_mask| {
            tree_mask
                .iter()
                .map(|col_mask| {
                    col_mask
                        .iter()
                        .map(|off| point + trace_step.mul_signed(*off).into_ef())
                        .collect()
                })
                .collect()
        })
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<Vec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        let constraint_zero_domain = CanonicCoset::new(self.log_size).coset;
        let denom = coset_vanishing(constraint_zero_domain, point);
        let denom_inverse = denom.inverse();
        let mut eval = PointEvaluator::new(mask.as_ref(), evaluation_accumulator, denom_inverse);
        let [is_first] = eval.next_interaction_mask(2, [0]);
        let blake_eval = BlakeSchedulerEval {
            eval,
            blake_lookup_elements: &self.blake_lookup_elements,
            round_lookup_elements: &self.round_lookup_elements,
            logup: LogupAtRow::new(1, self.claimed_sum, is_first),
        };
        blake_eval.eval();
    }
}

impl ComponentProver<SimdBackend> for BlakeSchedulerComponent {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, SimdBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<SimdBackend>,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        let mut domain_eval = DomainEvalHelper::new(
            self.log_size,
            self.log_size + 1,
            trace,
            evaluation_accumulator,
            self.max_constraint_log_degree_bound(),
            self.n_constraints(),
        );

        // TODO:
        let _span = span!(Level::INFO, "Constraint pointwise eval").entered();
        for vec_row in 0..(1 << (domain_eval.eval_domain.log_size() - LOG_N_LANES)) {
            let mut eval = DomainEvaluator::new(
                &domain_eval.trace.evals,
                vec_row,
                &domain_eval.accum.random_coeff_powers,
                domain_eval.trace_domain.log_size,
                domain_eval.eval_domain.log_size(),
            );
            // Constant column is_first.
            let [is_first] = eval.next_interaction_mask(2, [0]);
            let logup = LogupAtRow::new(1, self.claimed_sum, is_first);
            let blake_eval = BlakeSchedulerEval {
                eval,
                blake_lookup_elements: &self.blake_lookup_elements,
                round_lookup_elements: &self.round_lookup_elements,
                logup,
            };
            let eval = blake_eval.eval();
            domain_eval.finalize_row(vec_row, eval.row_res);
        }
    }

    fn lookup_values(&self, _trace: &ComponentTrace<'_, SimdBackend>) -> LookupValues {
        LookupValues::default()
    }
}
