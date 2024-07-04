use bytemuck::cast_slice;
use itertools::Itertools;
use num_traits::{One, Zero};
use tracing::{span, Level};

use super::eval::{AssertEvalAtRow, ConstraintCounter, EvalAtRow};
use super::lookup::LogupAtRow;
use super::round_constraints::BlakeEvalAtRow;
use super::LookupElements;
use crate::core::air::accumulation::{
    ColumnAccumulator, DomainEvaluationAccumulator, PointEvaluationAccumulator,
};
use crate::core::air::{Component, ComponentProver, ComponentTrace};
use crate::core::backend::simd::column::BaseFieldVec;
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Column, ColumnOps};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{FieldExpOps, FieldOps};
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{CanonicCoset, CircleDomain, CirclePoly};
use crate::core::utils::circle_domain_order_to_coset_order;
use crate::core::{ColumnVec, InteractionElements};
use crate::examples::blake::eval::{EvalAtDomain, EvalAtPoint};

// TODO: Fix.
pub struct BlakeRoundComponent {
    pub log_size: u32,
    pub lookup_elements: LookupElements,
    pub claimed_xor_sums: Vec<SecureField>,
}
impl BlakeRoundComponent {
    pub fn new(
        log_size: u32,
        lookup_elements: LookupElements,
        claimed_xor_sums: Vec<SecureField>,
    ) -> Self {
        Self {
            log_size,
            lookup_elements,
            claimed_xor_sums,
        }
    }
}
impl Component for BlakeRoundComponent {
    fn n_constraints(&self) -> usize {
        let counter = blake_counter();
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
            blake_counter()
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
        let counter = blake_counter();
        println!("counter trees: {}", counter.mask_offsets.len());
        let trace_step = CanonicCoset::new(self.log_size).step();
        counter.mask_offsets.map(|tree_mask| {
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

    fn interaction_element_ids(&self) -> Vec<String> {
        vec![]
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<Vec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        _interaction_elements: &InteractionElements,
    ) {
        println!("trees: {}", mask.len());
        let constraint_zero_domain = CanonicCoset::new(self.log_size).coset;
        let denom = coset_vanishing(constraint_zero_domain, point);
        let denom_inverse = denom.inverse();
        let mut eval = EvalAtPoint::new(mask.as_ref(), evaluation_accumulator, denom_inverse);
        let [is_first] = eval.next_interaction_mask(2, [0]);
        let mut blake_eval = BlakeEvalAtRow {
            eval,
            lookup_elements: self.lookup_elements,
            xor_logup: LogupAtRow::new(1, 2, &self.claimed_xor_sums, is_first),
        };
        blake_eval.eval();
    }
}

pub fn blake_counter() -> ConstraintCounter {
    let dummy_claimed_values = [SecureField::zero(); 512];
    let mut counter = BlakeEvalAtRow {
        eval: ConstraintCounter::default(),
        lookup_elements: LookupElements {
            z: SecureField::one(),
            alpha: SecureField::one(),
        },
        xor_logup: LogupAtRow::new(1, 2, &dummy_claimed_values, BaseField::zero()),
    };
    counter.eval.next_interaction_mask(2, [0]);
    counter.eval();
    counter.eval
}

impl ComponentProver<SimdBackend> for BlakeRoundComponent {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, SimdBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<SimdBackend>,
        _interaction_elements: &InteractionElements,
    ) {
        let mut domain_eval = DomainEvaluator::new(
            trace,
            evaluation_accumulator,
            self.max_constraint_log_degree_bound(),
            self.n_constraints(),
        );

        // TODO:
        for vec_row in 0..(1 << (domain_eval.eval_domain.log_size() - LOG_N_LANES)) {
            let mut eval = EvalAtDomain::new(
                &domain_eval.trace.evals,
                vec_row,
                &domain_eval.accum.random_coeff_powers,
                domain_eval.trace_domain.log_size,
                domain_eval.eval_domain.log_size(),
            );
            // Constant column is_first.
            let [is_first] = eval.next_interaction_mask(2, [0]);
            let xor_logup = LogupAtRow::new(1, 2, &self.claimed_xor_sums, is_first);
            let mut blake_eval = BlakeEvalAtRow {
                eval,
                lookup_elements: self.lookup_elements,
                xor_logup,
            };
            blake_eval.eval();
            domain_eval.finalize_row(vec_row, blake_eval.eval.row_res);
        }
    }
}

struct DomainEvaluator<'a> {
    eval_domain: CircleDomain,
    trace_domain: Coset,
    trace: &'a ComponentTrace<'a, SimdBackend>,
    denom_inv: BaseFieldVec,
    accum: ColumnAccumulator<'a, SimdBackend>,
}
impl<'a> DomainEvaluator<'a> {
    fn new(
        trace: &'a ComponentTrace<'a, SimdBackend>,
        evaluation_accumulator: &'a mut DomainEvaluationAccumulator<SimdBackend>,
        constraint_log_degree_bound: u32,
        n_constraints: usize,
    ) -> Self {
        println!("trees: {}", trace.evals.len());
        let log_eval_domain_size = trace.evals[0][0].domain.log_size();
        assert_eq!(
            log_eval_domain_size, constraint_log_degree_bound,
            "Extension not yet supported in generic evaluator"
        );
        let eval_domain = trace.evals[0][0].domain;
        let row_log_size = trace.polys[0][0].log_size();

        // Denoms.
        let trace_domain = CanonicCoset::new(row_log_size).coset;
        let span = span!(Level::INFO, "Constraint eval denominators").entered();

        let mut ys = BaseFieldVec::from_iter(eval_domain.iter().map(|p| p.y));
        <SimdBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut ys);

        let mut xs = BaseFieldVec::from_iter(eval_domain.iter().map(|p| p.x));
        <SimdBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut xs);
        let mut x_invs = BaseFieldVec::zeros(xs.len());
        <SimdBackend as FieldOps<BaseField>>::batch_inverse(&xs, &mut x_invs);

        let mut denoms =
            BaseFieldVec::from_iter(eval_domain.iter().map(|p| coset_vanishing(trace_domain, p)));
        <SimdBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut denoms);
        let mut denom_inv = BaseFieldVec::zeros(denoms.len());
        <SimdBackend as FieldOps<BaseField>>::batch_inverse(&denoms, &mut denom_inv);

        span.exit();

        let _span = span!(Level::INFO, "Constraint pointwise eval").entered();
        let [mut accum] =
            evaluation_accumulator.columns([(constraint_log_degree_bound, n_constraints)]);
        accum.random_coeff_powers.reverse();

        Self {
            eval_domain,
            trace_domain,
            trace,
            denom_inv,
            accum,
        }
    }
    fn finalize_row(&mut self, vec_row: usize, row_res: PackedSecureField) {
        unsafe {
            self.accum.col.set_packed(
                vec_row,
                self.accum.col.packed_at(vec_row) + row_res * self.denom_inv.data[vec_row],
            )
        }
    }
}

#[allow(dead_code)]
pub fn check_constraints_on_trace(
    log_size: u32,
    lookup_elements: LookupElements,
    claimed_sums: &[SecureField],
    trace: TreeVec<&[CirclePoly<SimdBackend>]>,
) {
    let trace_domain = CanonicCoset::new(log_size);
    let trace = trace.map(|trace| {
        trace
            .iter()
            .map(|poly| {
                let mut eval = poly.evaluate(trace_domain.circle_domain()).values;
                <SimdBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut eval);
                circle_domain_order_to_coset_order(cast_slice(&eval.data))
            })
            .collect_vec()
    });
    for i in 0..(1 << log_size) {
        let eval = AssertEvalAtRow {
            trace: &trace,
            col_index: vec![0; 2],
            row: i,
        };
        // TODO: take constant col.
        let mut blake_eval = BlakeEvalAtRow {
            eval,
            lookup_elements,
            xor_logup: LogupAtRow::new(
                1,
                2,
                claimed_sums,
                if i == 0 {
                    BaseField::one()
                } else {
                    BaseField::zero()
                },
            ),
        };
        blake_eval.eval();
    }
}
