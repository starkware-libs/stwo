use std::iter::zip;

use educe::Educe;
use itertools::Itertools;
use num_traits::{One, Zero};
use once_cell::sync::OnceCell;

use crate::constraint_framework::DomainEvaluator;
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentProver, ComponentTrace};
use crate::core::backend::{Backend, ColumnOps, CpuBackend};
use crate::core::circle::CirclePoint;
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::lookups::gkr_prover::GkrOps;
use crate::core::lookups::mle::Mle;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::INTERACTION_TRACE;
use crate::core::{ColumnVec, InteractionElements, LookupValues};

const LOG_EXPAND: u32 = 1;

#[derive(Educe)]
#[educe(Debug, Clone)]
pub struct MleEvalComponent<B: Backend> {
    pub n_variables: u32,
    // Only prover state.
    pub mle: OnceCell<Mle<B, SecureField>>,
    // Both prover and verifier state.
    pub eval_point: OnceCell<Vec<SecureField>>,
    pub claim: OnceCell<SecureField>,
}

impl<B: Backend> MleEvalComponent<B> {
    pub fn new(n_variables: u32) -> Self {
        Self {
            n_variables,
            mle: OnceCell::new(),
            eval_point: OnceCell::new(),
            claim: OnceCell::new(),
        }
    }

    pub fn eval_point(&self) -> &[SecureField] {
        self.eval_point.get().unwrap()
    }

    pub fn mle(&self) -> &Mle<B, SecureField> {
        self.mle.get().unwrap()
    }
}

impl MleEvalComponent<CpuBackend> {
    pub fn write_interaction_trace(
        &self,
    ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        let mle_basis_evals_bit_rev = &*CpuBackend::gen_eq_evals(self.eval_point(), One::one());
        let mut mle_basis_evals = mle_basis_evals_bit_rev.clone();
        CpuBackend::bit_reverse_column(&mut mle_basis_evals);

        let mle_coeffs_bit_rev = &**self.mle();
        let mut mle_coeffs = mle_coeffs_bit_rev.clone();
        CpuBackend::bit_reverse_column(&mut mle_coeffs);

        let mle_term_prefix_sum = prefix_sum(
            zip(mle_basis_evals, mle_coeffs).map(|(basis_eval, coeff)| basis_eval * coeff),
        );
        let mut mle_term_prefix_sum_bit_rev = mle_term_prefix_sum.clone();
        CpuBackend::bit_reverse_column(&mut mle_term_prefix_sum_bit_rev);

        let mut eq_evals_col0 = Vec::with_capacity(1 << self.n_variables);
        let mut eq_evals_col1 = Vec::with_capacity(1 << self.n_variables);
        let mut eq_evals_col2 = Vec::with_capacity(1 << self.n_variables);
        let mut eq_evals_col3 = Vec::with_capacity(1 << self.n_variables);

        for eq_eval in mle_basis_evals_bit_rev {
            let [v0, v1, v2, v3] = eq_eval.to_m31_array();
            eq_evals_col0.push(v0);
            eq_evals_col1.push(v1);
            eq_evals_col2.push(v2);
            eq_evals_col3.push(v3);
        }

        let mut terms_perfix_sum_col0 = Vec::with_capacity(1 << self.n_variables);
        let mut terms_perfix_sum_col1 = Vec::with_capacity(1 << self.n_variables);
        let mut terms_perfix_sum_col2 = Vec::with_capacity(1 << self.n_variables);
        let mut terms_perfix_sum_col3 = Vec::with_capacity(1 << self.n_variables);

        for term_sum in mle_term_prefix_sum_bit_rev {
            let [v0, v1, v2, v3] = term_sum.to_m31_array();
            terms_perfix_sum_col0.push(v0);
            terms_perfix_sum_col1.push(v1);
            terms_perfix_sum_col2.push(v2);
            terms_perfix_sum_col3.push(v3);
        }

        let trace_domain = CanonicCoset::new(self.n_variables).circle_domain();

        vec![
            CircleEvaluation::new(trace_domain, eq_evals_col0),
            CircleEvaluation::new(trace_domain, eq_evals_col1),
            CircleEvaluation::new(trace_domain, eq_evals_col2),
            CircleEvaluation::new(trace_domain, eq_evals_col3),
            CircleEvaluation::new(trace_domain, terms_perfix_sum_col0),
            CircleEvaluation::new(trace_domain, terms_perfix_sum_col1),
            CircleEvaluation::new(trace_domain, terms_perfix_sum_col2),
            CircleEvaluation::new(trace_domain, terms_perfix_sum_col3),
        ]
    }
}

// TODO: make generic
impl Component for MleEvalComponent<CpuBackend> {
    fn n_constraints(&self) -> usize {
        // TODO(andrew): Use constraint counter.
        // TODO(andrew): Prevent code duplication in verifier.
        // 1. eq eval column initial value constraint.
        // 2. eq eval column periodic constraints (`n_variable` many)
        // TODO: let n_eq_eval_periodic_constraints = self.multilinear_n_variables;
        let n_eq_eval_periodic_constraints = 0;
        // 3. multilinear term column constant coefficient constraint.
        // 4. Inner product was computed correctly.
        // n_eq_eval_periodic_constraints + 3
        n_eq_eval_periodic_constraints + 1
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        // TODO(andrew): Prevent code duplication in verifier.
        self.n_variables + LOG_EXPAND
    }

    fn n_interaction_phases(&self) -> u32 {
        2
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        let mut interaction_trace_degree_bounds = Vec::new();

        let eq_eval_col_log_degree_bounds = [self.n_variables; SECURE_EXTENSION_DEGREE];
        interaction_trace_degree_bounds.extend(eq_eval_col_log_degree_bounds);

        let mle_terms_prefix_sum_col_degree_bounds = [self.n_variables; SECURE_EXTENSION_DEGREE];
        interaction_trace_degree_bounds.extend(mle_terms_prefix_sum_col_degree_bounds);

        TreeVec::new(vec![vec![], interaction_trace_degree_bounds])
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let trace_domain_log_size = self.n_variables;
        let _trace_step = CanonicCoset::new(trace_domain_log_size).step();

        // For checking eq evals
        let eq_evals_secure_col_mask_points = vec![point];

        // TODO: For periodic constraints:
        // for log_step in 0..self.multilinear_n_variables {
        //     eq_evals_col_mask_points.push(point + trace_step.repeated_double(log_step))
        // }

        let mut interaction_trace_mask_points = Vec::new();
        // Copy `SECURE_EXTENSION_DEGREE` many times since columns are stored as base field columns.
        (0..SECURE_EXTENSION_DEGREE).for_each(|_| {
            interaction_trace_mask_points.push(eq_evals_secure_col_mask_points.clone())
        });
        // (0..SECURE_EXTENSION_DEGREE)
        //     .for_each(|_| interaction_mask_points.push(multilinear_term_mask_points.clone()));

        TreeVec::new(vec![vec![], interaction_trace_mask_points])
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        _point: CirclePoint<SecureField>,
        _mask: &TreeVec<Vec<Vec<SecureField>>>,
        _evaluation_accumulator: &mut PointEvaluationAccumulator,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        todo!()
    }
}

impl ComponentProver<CpuBackend> for MleEvalComponent<CpuBackend> {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, CpuBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CpuBackend>,
        interaction_elements: &InteractionElements,
        lookup_values: &LookupValues,
    ) {
        // Obtain low degree extension of the MLE coeffs column.
        let mut mle_coeffs_cols = [
            Vec::with_capacity(1 << self.n_variables),
            Vec::with_capacity(1 << self.n_variables),
            Vec::with_capacity(1 << self.n_variables),
            Vec::with_capacity(1 << self.n_variables),
        ];

        for mle_coeff in &**self.mle.get().unwrap() {
            let [v0, v1, v2, v3] = mle_coeff.to_m31_array();
            mle_coeffs_cols[0].push(v0);
            mle_coeffs_cols[1].push(v1);
            mle_coeffs_cols[2].push(v2);
            mle_coeffs_cols[3].push(v3);
        }

        let trace_domain = CanonicCoset::new(self.n_variables);
        let evaluation_domain_log_size = trace.evals[INTERACTION_TRACE][0].domain.log_size();
        let evaluation_domain = CanonicCoset::new(evaluation_domain_log_size);

        let mle_coeffs_evals = mle_coeffs_cols.map(|col| {
            CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
                trace_domain.circle_domain(),
                col,
            )
            .interpolate()
            .evaluate(evaluation_domain.circle_domain())
        });

        let mle_coeffs_evals = [
            &mle_coeffs_evals[0],
            &mle_coeffs_evals[1],
            &mle_coeffs_evals[2],
            &mle_coeffs_evals[3],
        ];

        let eq_eval_evals = [
            &trace.evals[INTERACTION_TRACE][0],
            &trace.evals[INTERACTION_TRACE][1],
            &trace.evals[INTERACTION_TRACE][2],
            &trace.evals[INTERACTION_TRACE][3],
        ];

        let log_n_rows = self.n_variables;
        let log_n_packed_eval_rows = log_n_rows + LOG_EXPAND;

        for packed_row_i in 0..1 << log_n_packed_eval_rows {
            let mut eval = DomainEvaluator::new(
                &trace.evals,
                packed_row_i,
                &pows,
                log_n_trace_rows,
                log_n_trace_rows + LOG_EXPAND,
            );

            let at = EqEvalsMaskAt::draw::<BASE_TRACE>(&mut eval);
            let is = EqEvalsMaskIs::draw::<CONST_TRACE>(&mut eval);
            eq_evals_check(&mut eval, point_meta, &at, &is);
            debug_assert_eq!(eval.constraint_index, n_constraints);

            let vanish_on_trace_domain_eval_inv =
                vanish_on_trace_domain_evals_bit_rev_inv.data[packed_row_i];
            let quotient = eval.row_res * vanish_on_trace_domain_eval_inv;

            unsafe {
                let acc_prev = acc_col.col.packed_at(packed_row_i);
                acc_col.col.set_packed(packed_row_i, acc_prev + quotient)
            }
        }

        // let mut eq_eval_mul_coeffs_evals_bit_rev = Vec::new();

        // // Prefix sum constraint
        // for i in 0..evaluation_domain.size() {
        //     let eq_eval = SecureField::from_m31_array(eq_eval_evals.map(|c| c[i]));
        //     let mle_coeff = SecureField::from_m31_array(mle_coeffs_evals.map(|c| c[i]));
        //     eq_eval_mul_coeffs_evals_bit_rev.push(eq_eval * mle_coeff);
        // }

        // let mut eq_eval_mul_coeffs_evals = eq_eval_mul_coeffs_evals_bit_rev;
        // CpuBackend::bit_reverse_column(&mut eq_eval_mul_coeffs_evals);

        // let prefix_sum_evals = [
        //     &trace.evals[INTERACTION_TRACE][4],
        //     &trace.evals[INTERACTION_TRACE][5],
        //     &trace.evals[INTERACTION_TRACE][6],
        //     &trace.evals[INTERACTION_TRACE][7],
        // ];

        // let mut prefix_sum_evals_bit_rev = Vec::new();

        // for i in 0..evaluation_domain.size() {
        //     let v = SecureField::from_m31_array(prefix_sum_evals.map(|c| c[i]));
        //     prefix_sum_evals_bit_rev.push(v);
        // }

        // let mut prefix_sum_evals = prefix_sum_evals_bit_rev;
        // CpuBackend::bit_reverse_column(&mut prefix_sum_evals);

        // // TODO: Only generate and use the periodic evals.
        // let mut vanish_on_trace_domain_evals = evaluation_domain
        //     .circle_domain()
        //     .iter()
        //     .map(|p| coset_vanishing(trace_domain.coset, p))
        //     .collect_vec();
        // let vanish_on_trace_domain_evals_inv = vanish_on_trace_domain_evals;
        // CpuBackend::bit_reverse_column(&mut vanish_on_trace_domain_evals);

        // let prefix_sum_quotient_evals = Vec::new();

        // for i in 0..evaluation_domain.size() {
        //     // let prefix_sum_evals[i]

        //     prefix_sum_quotient_evals[i]
        // }

        // // TODO: Redundant re-computation.
        // let mle_basis_evals_bit_rev = &*CpuBackend::gen_eq_evals(self.eval_point(), One::one());
        // let mut mle_basis_evals = mle_basis_evals_bit_rev.clone();
        // CpuBackend::bit_reverse_column(&mut mle_basis_evals);

        // let mle_coeffs_bit_rev = &**self.mle.get().unwrap();
        // let mut mle_coeffs = mle_coeffs_bit_rev.clone();
        // CpuBackend::bit_reverse_column(&mut mle_coeffs);

        // let mle_coeffs_prefix_sum = prefix_sum(mle_coeffs);
        // let mut mle_coeffs_prefix_sum_bit_rev = mle_coeffs_prefix_sum.clone();
        // CpuBackend::bit_reverse_column(&mut mle_coeffs_prefix_sum_bit_rev);

        // // Obtain evaluations on evaluation domain.
        // let mut coeffs_perfix_sum_cols = [
        //     Vec::with_capacity(1 << self.n_variables),
        //     Vec::with_capacity(1 << self.n_variables),
        //     Vec::with_capacity(1 << self.n_variables),
        //     Vec::with_capacity(1 << self.n_variables),
        // ];

        // for term_sum in mle_coeffs_prefix_sum_bit_rev {
        //     let [v0, v1, v2, v3] = term_sum.to_m31_array();
        //     coeffs_perfix_sum_cols[0].push(v0);
        //     coeffs_perfix_sum_cols[1].push(v1);
        //     coeffs_perfix_sum_cols[2].push(v2);
        //     coeffs_perfix_sum_cols[3].push(v3);
        // }

        // let trace_domain = CanonicCoset::new(self.n_variables).circle_domain();
        // let evaluation_domain = trace.evals[INTERACTION_TRACE][0].domain;
        // let coeffs_perfix_sum_evals = coeffs_perfix_sum_cols.map(|col| {
        //     CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(trace_domain, col)
        //         .interpolate()
        //         .evaluate(evaluation_domain)
        // });

        // todo!()
    }

    fn lookup_values(&self, _trace: &ComponentTrace<'_, CpuBackend>) -> LookupValues {
        LookupValues::default()
    }
}

fn prefix_sum(v: impl IntoIterator<Item = SecureField>) -> Vec<SecureField> {
    v.into_iter()
        .scan(SecureField::zero(), |sum, coeff| {
            *sum += coeff;
            Some(*sum)
        })
        .collect()
}
