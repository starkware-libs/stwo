//! Multilinear extension (MLE) eval at point constraints.
// TODO(andrew): Remove in downstream PR.
#![allow(dead_code)]

use std::iter::zip;

use itertools::{chain, zip_eq, Itertools};
use num_traits::{One, Zero};
use tracing::{span, Level};

use crate::constraint_framework::preprocessed_columns::IsFirst;
use crate::constraint_framework::{
    EvalAtRow, InfoEvaluator, PointEvaluator, SimdDomainEvaluator, TraceLocationAllocator,
};
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentProver, Trace};
use crate::core::backend::cpu::bit_reverse;
use crate::core::backend::simd::column::{SecureColumn, VeryPackedSecureColumnByCoords};
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::prefix_sum::inclusive_prefix_sum;
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::very_packed_m31::{VeryPackedBaseField, LOG_N_VERY_PACKED_ELEMS};
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::constraints::{coset_vanishing, point_vanishing};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumnByCoords;
use crate::core::fields::{Field, FieldExpOps};
use crate::core::lookups::gkr_prover::GkrOps;
use crate::core::lookups::mle::Mle;
use crate::core::lookups::utils::eq;
use crate::core::pcs::{TreeSubspan, TreeVec};
use crate::core::poly::circle::{
    CanonicCoset, CircleEvaluation, SecureCirclePoly, SecureEvaluation,
};
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::BitReversedOrder;
use crate::core::utils::{bit_reverse_index, coset_index_to_circle_domain_index};
use crate::core::ColumnVec;

/// Prover component that carries out a univariate IOP for multilinear eval at point.
///
/// See <https://eprint.iacr.org/2023/1284.pdf> (Section 5.1).
#[allow(dead_code)]
pub struct MleEvalProverComponent<'twiddles, 'oracle, O: MleCoeffColumnOracle> {
    /// Polynomials encoding the multilinear Lagrange basis coefficients of the MLE.
    mle_coeff_column_poly: SecureCirclePoly<SimdBackend>,
    /// Oracle for the polynomial encoding the multilinear Lagrange basis coefficients of the MLE.
    ///
    /// The oracle values should match `mle_coeff_column_poly` for any given evaluation point. The
    /// polynomial is only stored directly to speed up constraint evaluation. The oracle is stored
    /// to perform consistency checks with `mle_coeff_column_poly`.
    mle_coeff_column_oracle: &'oracle O,
    /// Multilinear evaluation point.
    mle_eval_point: MleEvalPoint,
    /// Equals `mle_claim / 2^mle_n_variables`.
    mle_claim_shift: SecureField,
    /// Commitment tree index for the trace.
    interaction: usize,
    /// Location in the trace for the this component.
    trace_locations: TreeVec<TreeSubspan>,
    /// Precomputed twiddles tree.
    twiddles: &'twiddles TwiddleTree<SimdBackend>,
}

impl<'twiddles, 'oracle, O: MleCoeffColumnOracle> MleEvalProverComponent<'twiddles, 'oracle, O> {
    /// Generates prover component that carries out univariate IOP for MLE eval at point.
    ///
    /// # Panics
    ///
    /// Panics if the eval point has a coordinate that is zero or one. This is a completeness bug.
    pub fn generate(
        location_allocator: &mut TraceLocationAllocator,
        mle_coeff_column_oracle: &'oracle O,
        mle_eval_point: &[SecureField],
        mle: Mle<SimdBackend, SecureField>,
        mle_claim: SecureField,
        twiddles: &'twiddles TwiddleTree<SimdBackend>,
        interaction: usize,
    ) -> Self {
        #[cfg(test)]
        assert_eq!(mle_claim, mle.eval_at_point(mle_eval_point));
        let n_variables = mle.n_variables();
        let mle_claim_shift = mle_claim / BaseField::from(1 << n_variables);

        let domain = CanonicCoset::new(n_variables as u32).circle_domain();
        let values = mle.into_evals().into_secure_column_by_coords();
        let mle_trace = SecureEvaluation::<SimdBackend, BitReversedOrder>::new(domain, values);
        let mle_coeff_column_poly = mle_trace.interpolate_with_twiddles(twiddles);

        let trace_structure = mle_eval_info(interaction, n_variables).mask_offsets;
        let trace_locations = location_allocator.next_for_structure(&trace_structure);

        Self {
            mle_coeff_column_poly,
            mle_coeff_column_oracle,
            mle_eval_point: MleEvalPoint::new(mle_eval_point),
            mle_claim_shift,
            interaction,
            trace_locations,
            twiddles,
        }
    }

    /// Size of this component's trace columns.
    pub fn log_size(&self) -> u32 {
        self.mle_eval_point.n_variables() as u32
    }

    pub fn eval_info(&self) -> InfoEvaluator {
        let n_variables = self.mle_eval_point.n_variables();
        mle_eval_info(self.interaction, n_variables)
    }
}

impl<O: MleCoeffColumnOracle> Component for MleEvalProverComponent<'_, '_, O> {
    fn n_constraints(&self) -> usize {
        self.eval_info().n_constraints
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size() + 1
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        let log_size = self.log_size();
        let InfoEvaluator { mask_offsets, .. } = self.eval_info();
        mask_offsets.map(|tree_offsets| vec![log_size; tree_offsets.len()])
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let trace_step = CanonicCoset::new(self.log_size()).step();
        let InfoEvaluator { mask_offsets, .. } = self.eval_info();
        mask_offsets.map_cols(|col_offsets| {
            col_offsets
                .iter()
                .map(|offset| point + trace_step.mul_signed(*offset).into_ef())
                .collect()
        })
    }

    fn preproccessed_column_indices(&self) -> ColumnVec<usize> {
        vec![]
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        accumulator: &mut PointEvaluationAccumulator,
    ) {
        // Consistency check the MLE coeffs column polynomial and oracle.
        let mle_coeff_col_eval = self.mle_coeff_column_poly.eval_at_point(point);
        let oracle_mle_coeff_col_eval = self.mle_coeff_column_oracle.evaluate_at_point(point, mask);
        assert_eq!(mle_coeff_col_eval, oracle_mle_coeff_col_eval);

        let component_mask = mask.sub_tree(&self.trace_locations);
        let trace_coset = CanonicCoset::new(self.log_size()).coset;
        let vanish_on_trace_eval_inv = coset_vanishing(trace_coset, point).inverse();
        let mut eval = PointEvaluator::new(
            component_mask,
            accumulator,
            vanish_on_trace_eval_inv,
            self.log_size(),
            SecureField::zero(),
        );

        let carry_quotients_col_eval = eval_carry_quotient_col(&self.mle_eval_point, point);
        let is_first = eval_is_first(trace_coset, point);
        let is_second = eval_is_first(trace_coset, point - trace_coset.step.into_ef());

        // TODO(andrew): Consider evaluating `is_first` and `is_second` inside
        // `eval_mle_eval_constraints` once constant column approach updated.
        eval_mle_eval_constraints(
            self.interaction,
            &mut eval,
            mle_coeff_col_eval,
            &self.mle_eval_point,
            self.mle_claim_shift,
            carry_quotients_col_eval,
            is_first,
            is_second,
        )
    }
}

impl<O: MleCoeffColumnOracle> ComponentProver<SimdBackend> for MleEvalProverComponent<'_, '_, O> {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, SimdBackend>,
        accumulator: &mut DomainEvaluationAccumulator<SimdBackend>,
    ) {
        let eval_domain = CanonicCoset::new(self.max_constraint_log_degree_bound()).circle_domain();
        let trace_domain = CanonicCoset::new(self.log_size());

        let mut component_trace = trace.evals.sub_tree(&self.trace_locations).map_cols(|c| *c);

        // Build auxiliary trace.
        let span = span!(Level::INFO, "Extension").entered();
        let mle_coeffs_column_lde = self
            .mle_coeff_column_poly
            .evaluate_with_twiddles(eval_domain, self.twiddles)
            .into_coordinate_evals();
        let carry_quotients_column_lde = gen_carry_quotient_col(&self.mle_eval_point.p)
            .interpolate_with_twiddles(self.twiddles)
            .evaluate_with_twiddles(eval_domain, self.twiddles)
            .into_coordinate_evals();
        let is_first_lde = IsFirst::new(self.log_size())
            .gen_column_simd()
            .interpolate_with_twiddles(self.twiddles)
            .evaluate_with_twiddles(eval_domain, self.twiddles);
        let aux_interaction = component_trace.len();
        let aux_trace = chain![
            &mle_coeffs_column_lde,
            &carry_quotients_column_lde,
            [&is_first_lde]
        ]
        .collect();
        component_trace.push(aux_trace);
        span.exit();

        // Denom inverses.
        let log_expand = eval_domain.log_size() - trace_domain.log_size();
        let mut denom_inv = (0..1 << log_expand)
            .map(|i| coset_vanishing(trace_domain.coset(), eval_domain.at(i)).inverse())
            .collect_vec();
        bit_reverse(&mut denom_inv);

        // Accumulator.
        let [mut acc] = accumulator.columns([(eval_domain.log_size(), self.n_constraints())]);
        acc.random_coeff_powers.reverse();
        let acc_col = unsafe { VeryPackedSecureColumnByCoords::transform_under_mut(acc.col) };

        let _span = span!(Level::INFO, "Constraint pointwise eval").entered();
        let n_very_packed_rows =
            1 << (eval_domain.log_size() - LOG_N_LANES - LOG_N_VERY_PACKED_ELEMS);
        for vec_row in 0..n_very_packed_rows {
            // Evaluate constrains at row.
            let mut eval = SimdDomainEvaluator::new(
                &component_trace,
                vec_row,
                &acc.random_coeff_powers,
                trace_domain.log_size(),
                eval_domain.log_size(),
                self.log_size(),
                SecureField::zero(),
            );
            let [mle_coeffs_col_eval] = eval.next_extension_interaction_mask(aux_interaction, [0]);
            let [carry_quotients_col_eval] =
                eval.next_extension_interaction_mask(aux_interaction, [0]);
            let [is_first, is_second] = eval.next_interaction_mask(aux_interaction, [0, -1]);
            eval_mle_eval_constraints(
                self.interaction,
                &mut eval,
                mle_coeffs_col_eval,
                &self.mle_eval_point,
                self.mle_claim_shift,
                carry_quotients_col_eval,
                is_first,
                is_second,
            );

            // Finalize row.
            let row_res = eval.row_res;
            let denom_inv = VeryPackedBaseField::broadcast(
                denom_inv
                    [vec_row >> (trace_domain.log_size() - LOG_N_LANES - LOG_N_VERY_PACKED_ELEMS)],
            );
            unsafe { acc_col.set_packed(vec_row, acc_col.packed_at(vec_row) + row_res * denom_inv) }
        }
    }
}

/// Verifier component that carries out a univariate IOP for multilinear eval at point.
///
/// See <https://eprint.iacr.org/2023/1284.pdf> (Section 5.1).
pub struct MleEvalVerifierComponent<'oracle, O: MleCoeffColumnOracle> {
    /// Oracle for the polynomial encoding the multilinear Lagrange basis coefficients of the MLE.
    mle_coeff_column_oracle: &'oracle O,
    /// Multilinear evaluation point.
    mle_eval_point: MleEvalPoint,
    /// Equals `mle_claim / 2^mle_n_variables`.
    mle_claim_shift: SecureField,
    /// Commitment tree index for the trace.
    interaction: usize,
    /// Location in the trace for the this component.
    trace_location: TreeVec<TreeSubspan>,
}

impl<'oracle, O: MleCoeffColumnOracle> MleEvalVerifierComponent<'oracle, O> {
    pub fn new(
        location_allocator: &mut TraceLocationAllocator,
        mle_coeff_column_oracle: &'oracle O,
        eval_point: &[SecureField],
        claim: SecureField,
        interaction: usize,
    ) -> Self {
        let mle_eval_point = MleEvalPoint::new(eval_point);
        let n_variables = mle_eval_point.n_variables();
        let mle_claim_shift = claim / BaseField::from(1 << n_variables);

        let trace_structure = mle_eval_info(interaction, n_variables).mask_offsets;
        let trace_location = location_allocator.next_for_structure(&trace_structure);

        Self {
            mle_coeff_column_oracle,
            mle_eval_point,
            mle_claim_shift,
            interaction,
            trace_location,
        }
    }

    /// Size of this component's trace columns.
    pub fn log_size(&self) -> u32 {
        self.mle_eval_point.n_variables() as u32
    }

    pub fn eval_info(&self) -> InfoEvaluator {
        let n_variables = self.mle_eval_point.n_variables();
        mle_eval_info(self.interaction, n_variables)
    }
}

impl<O: MleCoeffColumnOracle> Component for MleEvalVerifierComponent<'_, O> {
    fn n_constraints(&self) -> usize {
        self.eval_info().n_constraints
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size() + 1
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        let log_size = self.log_size();
        let InfoEvaluator { mask_offsets, .. } = self.eval_info();
        mask_offsets.map(|tree_offsets| vec![log_size; tree_offsets.len()])
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let trace_step = CanonicCoset::new(self.log_size()).step();
        let InfoEvaluator { mask_offsets, .. } = self.eval_info();
        mask_offsets.map_cols(|col_offsets| {
            col_offsets
                .iter()
                .map(|offset| point + trace_step.mul_signed(*offset).into_ef())
                .collect()
        })
    }

    fn preproccessed_column_indices(&self) -> ColumnVec<usize> {
        vec![]
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        accumulator: &mut PointEvaluationAccumulator,
    ) {
        let component_mask = mask.sub_tree(&self.trace_location);
        let trace_coset = CanonicCoset::new(self.log_size()).coset;
        let vanish_on_trace_eval_inv = coset_vanishing(trace_coset, point).inverse();
        let mut eval = PointEvaluator::new(
            component_mask,
            accumulator,
            vanish_on_trace_eval_inv,
            self.log_size(),
            SecureField::zero(),
        );

        let mle_coeff_col_eval = self.mle_coeff_column_oracle.evaluate_at_point(point, mask);
        let carry_quotients_col_eval = eval_carry_quotient_col(&self.mle_eval_point, point);
        let is_first = eval_is_first(trace_coset, point);
        let is_second = eval_is_first(trace_coset, point - trace_coset.step.into_ef());

        eval_mle_eval_constraints(
            self.interaction,
            &mut eval,
            mle_coeff_col_eval,
            &self.mle_eval_point,
            self.mle_claim_shift,
            carry_quotients_col_eval,
            is_first,
            is_second,
        )
    }
}

fn mle_eval_info(interaction: usize, n_variables: usize) -> InfoEvaluator {
    let mut eval = InfoEvaluator::empty();
    let mle_eval_point = MleEvalPoint::new(&vec![SecureField::from(2); n_variables]);
    let mle_claim_shift = SecureField::zero();
    let mle_coeffs_col_eval = SecureField::zero().into();
    let carry_quotients_col_eval = SecureField::zero().into();
    let is_first = BaseField::zero().into();
    let is_second = BaseField::zero().into();
    eval_mle_eval_constraints(
        interaction,
        &mut eval,
        mle_coeffs_col_eval,
        &mle_eval_point,
        mle_claim_shift,
        carry_quotients_col_eval,
        is_first,
        is_second,
    );
    eval
}

/// Univariate polynomial oracle that encodes multilinear Lagrange basis coefficients of a MLE.
///
/// The column should encode the MLE coefficients ordered on a circle domain.
pub trait MleCoeffColumnOracle {
    fn evaluate_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
    ) -> SecureField;
}

/// Evaluates constraints that guarantee an MLE evaluates to a claim at a given point.
///
/// `mle_coeffs_col_eval` should be the evaluation of the column containing the coefficients of the
/// MLE in the multilinear Lagrange basis. `mle_claim_shift` should equal `claim / 2^N_VARIABLES`.
#[allow(clippy::too_many_arguments)]
pub fn eval_mle_eval_constraints<E: EvalAtRow>(
    interaction: usize,
    eval: &mut E,
    mle_coeffs_col_eval: E::EF,
    mle_eval_point: &MleEvalPoint,
    mle_claim_shift: SecureField,
    carry_quotients_col_eval: E::EF,
    is_first: E::F,
    is_second: E::F,
) {
    let eq_col_eval = eval_eq_constraints(
        interaction,
        eval,
        mle_eval_point,
        carry_quotients_col_eval,
        is_first,
        is_second,
    );
    let terms_col_eval = mle_coeffs_col_eval * eq_col_eval;
    eval_prefix_sum_constraints(interaction, eval, terms_col_eval, mle_claim_shift)
}

#[derive(Debug, Clone)]
pub struct MleEvalPoint {
    // Equals `eq({0}^|p|, p)`.
    eq_0_p: SecureField,
    // Equals `eq({1}^|p|, p)`.
    eq_1_p: SecureField,
    // Index `i` stores `eq(({1}^|i|, 0), p[0..i+1]) / eq(({0}^|i|, 1), p[0..i+1])`.
    eq_carry_quotients: Vec<SecureField>,
    // Point `p`.
    p: Vec<SecureField>,
}

impl MleEvalPoint {
    /// Creates new metadata from point `p`.
    ///
    /// # Panics
    ///
    /// Panics if the point is empty or has a coordinate that is zero or one.
    pub fn new(p: &[SecureField]) -> Self {
        assert!(!p.is_empty());
        let n_variables = p.len();
        let zero = SecureField::zero();
        let one = SecureField::one();

        Self {
            eq_0_p: eq(&vec![zero; n_variables], p),
            eq_1_p: eq(&vec![one; n_variables], p),
            eq_carry_quotients: (0..n_variables)
                .map(|i| {
                    let mut numer_assignment = vec![one; i + 1];
                    numer_assignment[i] = zero;
                    let mut denom_assignment = vec![zero; i + 1];
                    denom_assignment[i] = one;
                    eq(&numer_assignment, &p[..i + 1]) / eq(&denom_assignment, &p[..i + 1])
                })
                .collect(),
            p: p.to_vec(),
        }
    }

    pub fn n_variables(&self) -> usize {
        self.p.len()
    }
}

/// Evaluates EqEvals constraints on a column.
///
/// Returns the evaluation at offset 0 on the column.
///
/// Given a column `c(P)` defined on a circle domain `D`, and an MLE eval point `(r0, r1, ...)`
/// evaluates constraints that guarantee: `c(D[b0, b1, ...]) = eq((b0, b1, ...), (r0, r1, ...))`.
///
/// See <https://eprint.iacr.org/2023/1284.pdf> (Section 5.1).
fn eval_eq_constraints<E: EvalAtRow>(
    eq_interaction: usize,
    eval: &mut E,
    mle_eval_point: &MleEvalPoint,
    carry_quotients_col_eval: E::EF,
    is_first: E::F,
    is_second: E::F,
) -> E::EF {
    let [curr, next_next] = eval.next_extension_interaction_mask(eq_interaction, [0, 2]);

    // Check the initial value on half_coset0 and final value on half_coset1.
    // Combining these constraints is safe because `is_first` and `is_second` are never
    // non-zero at the same time on the trace.
    let half_coset0_initial_check = (curr.clone() - mle_eval_point.eq_0_p) * is_first;
    let half_coset1_final_check = (curr.clone() - mle_eval_point.eq_1_p) * is_second;
    eval.add_constraint(half_coset0_initial_check + half_coset1_final_check);

    // Check all the steps.
    eval.add_constraint(curr.clone() - next_next * carry_quotients_col_eval);

    curr
}

/// Evaluates inclusive prefix sum constraints on a column.
///
/// Note the column values must be shifted by `cumulative_sum_shift` so the last value equals zero.
/// `cumulative_sum_shift` should equal `cumulative_sum / column_size`.
fn eval_prefix_sum_constraints<E: EvalAtRow>(
    interaction: usize,
    eval: &mut E,
    row_diff: E::EF,
    cumulative_sum_shift: SecureField,
) {
    let [curr, prev] = eval.next_extension_interaction_mask(interaction, [0, -1]);
    eval.add_constraint(curr - prev - row_diff + cumulative_sum_shift);
}

/// Generates a trace.
///
/// Trace structure:
///
/// ```text
/// ---------------------------------------------------------
/// |       EqEvals (basis)     |   MLE terms (prefix sum)  |
/// ---------------------------------------------------------
/// |  c0  |  c1  |  c2  |  c3  |  c4  |  c5  |  c6 |  c7 |
/// ---------------------------------------------------------
/// ```
pub fn build_trace(
    mle: &Mle<SimdBackend, SecureField>,
    eval_point: &[SecureField],
    claim: SecureField,
) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let eq_evals = SimdBackend::gen_eq_evals(eval_point, SecureField::one()).into_evals();
    let mle_terms = hadamard_product(mle, &eq_evals);

    let eq_evals_cols = eq_evals.into_secure_column_by_coords().columns;
    let mle_terms_cols = mle_terms.into_secure_column_by_coords().columns;

    #[cfg(test)]
    debug_assert_eq!(claim, mle.eval_at_point(eval_point));
    let shift = claim / BaseField::from(mle.len());
    let packed_shift_coords = PackedSecureField::broadcast(shift).into_packed_m31s();
    let mut shifted_mle_terms_cols = mle_terms_cols;
    zip(&mut shifted_mle_terms_cols, packed_shift_coords)
        .for_each(|(col, shift_coord)| col.data.iter_mut().for_each(|v| *v -= shift_coord));
    let shifted_prefix_sum_cols = shifted_mle_terms_cols.map(inclusive_prefix_sum);

    let log_trace_domain_size = mle.n_variables() as u32;
    let trace_domain = CanonicCoset::new(log_trace_domain_size).circle_domain();

    chain![eq_evals_cols, shifted_prefix_sum_cols]
        .map(|c| CircleEvaluation::new(trace_domain, c))
        .collect()
}

/// Returns succinct Eq carry quotients column.
///
/// Given column `c(P)` defined on a [`CircleDomain`] `D = +-C`, and an MLE eval point
/// `(r0, r1, ...)` let `c(D[b0, b1, ...]) = eq((b0, b1, ...), (r0, r1, ...))`. This function
/// returns column `q(P)` such that all `c(C[i]) = c(C[i + 1]) * q(C[i])` and
/// `c(-C[i]) = c(-C[i + 1]) * q(-C[i])`.
///
/// [`CircleDomain`]: crate::core::poly::circle::CircleDomain
fn gen_carry_quotient_col(
    eval_point: &[SecureField],
) -> SecureEvaluation<SimdBackend, BitReversedOrder> {
    assert!(!eval_point.is_empty());
    let mle_eval_point = MleEvalPoint::new(eval_point);
    let (half_coset0_carry_quotients, half_coset1_carry_quotients) =
        gen_half_coset_carry_quotients(&mle_eval_point);

    let log_size = mle_eval_point.n_variables() as u32;
    let size = 1 << log_size;
    let half_coset_size = size / 2;
    let mut col = SecureColumnByCoords::<SimdBackend>::zeros(size);

    // TODO(andrew): Optimize.
    for i in 0..half_coset_size {
        let half_coset0_index = coset_index_to_circle_domain_index(i * 2, log_size);
        let half_coset1_index = coset_index_to_circle_domain_index(i * 2 + 1, log_size);
        let half_coset0_index_bit_rev = bit_reverse_index(half_coset0_index, log_size);
        let half_coset1_index_bit_rev = bit_reverse_index(half_coset1_index, log_size);

        let n_trailing_ones = i.trailing_ones() as usize;
        let half_coset0_carry_quotient = half_coset0_carry_quotients[n_trailing_ones];
        let half_coset1_carry_quotient = half_coset1_carry_quotients[n_trailing_ones];

        col.set(half_coset0_index_bit_rev, half_coset0_carry_quotient);
        col.set(half_coset1_index_bit_rev, half_coset1_carry_quotient);
    }

    let domain = CanonicCoset::new(log_size).circle_domain();
    SecureEvaluation::new(domain, col)
}

/// Evaluates the succinct Eq carry quotients column at point `p`.
///
/// See [`gen_carry_quotient_col`].
// TODO(andrew): Optimize further. Inline `eval_step_selector` and get runtime down to
// O(N_VARIABLES) vs current O(N_VARIABLES^2). Can also use vanishing evals to compute
// half_coset0_last half_coset1_first.
fn eval_carry_quotient_col(eval_point: &MleEvalPoint, p: CirclePoint<SecureField>) -> SecureField {
    let n_variables = eval_point.n_variables();
    let log_size = n_variables as u32;
    let coset = CanonicCoset::new(log_size).coset();

    let (half_coset0_carry_quotients, half_coset1_carry_quotients) =
        gen_half_coset_carry_quotients(eval_point);

    let mut eval = SecureField::zero();

    for variable_i in 0..n_variables.saturating_sub(1) {
        let log_step = variable_i as u32 + 2;
        let offset = (1 << (log_step - 1)) - 2;
        let half_coset0_selector = eval_step_selector_with_offset(coset, offset, log_step, p);
        let half_coset1_selector = eval_step_selector_with_offset(coset, offset + 1, log_step, p);
        let half_coset0_carry_quotient = half_coset0_carry_quotients[variable_i];
        let half_coset1_carry_quotient = half_coset1_carry_quotients[variable_i];
        eval += half_coset0_selector * half_coset0_carry_quotient;
        eval += half_coset1_selector * half_coset1_carry_quotient;
    }

    let half_coset0_last = eval_is_first(coset, p + coset.step.double().into_ef());
    let half_coset1_first = eval_is_first(coset, p + coset.step.into_ef());
    eval += *half_coset0_carry_quotients.last().unwrap() * half_coset0_last;
    eval += *half_coset1_carry_quotients.last().unwrap() * half_coset1_first;

    eval
}

/// Evaluates a polynomial that's `1` every `2^log_step` coset points, shifted by an offset, and `0`
/// elsewhere on coset.
fn eval_step_selector_with_offset(
    coset: Coset,
    offset: usize,
    log_step: u32,
    p: CirclePoint<SecureField>,
) -> SecureField {
    let offset_step = coset.step.mul(offset as u128);
    eval_step_selector(coset, log_step, p - offset_step.into_ef())
}

/// Evaluates a polynomial that's `1` every `2^log_step` coset points and `0` elsewhere on coset.
fn eval_step_selector(coset: Coset, log_step: u32, p: CirclePoint<SecureField>) -> SecureField {
    if log_step == 0 {
        return SecureField::one();
    }

    // Rotate the coset so its first point is the identity element.
    let p = p - coset.initial.into_ef();
    let mut vanish_at_log_step = (0..coset.log_size)
        .scan(p, |p, _| {
            let res = *p;
            *p = p.double();
            Some(res.y)
        })
        .collect_vec();
    vanish_at_log_step.reverse();
    // We only need the first `log_step` many values.
    vanish_at_log_step.truncate(log_step as usize);
    let vanish_at_log_step_inv = SecureField::batch_inverse(&vanish_at_log_step);

    let half_coset_selector_dbl = (vanish_at_log_step[0] * vanish_at_log_step_inv[1]).square();
    let vanish_substep_inv_sum = vanish_at_log_step_inv[1..].iter().sum::<SecureField>();
    (half_coset_selector_dbl + vanish_at_log_step[0] * vanish_substep_inv_sum.double())
        / BaseField::from(1 << (log_step + 1))
}

fn eval_is_first(coset: Coset, p: CirclePoint<SecureField>) -> SecureField {
    coset_vanishing(coset, p)
        / (point_vanishing(coset.initial, p) * BaseField::from(1 << coset.log_size))
}

/// Output of the form: `(half_coset0_carry_quotients, half_coset1_carry_quotients)`.
fn gen_half_coset_carry_quotients(
    eval_point: &MleEvalPoint,
) -> (Vec<SecureField>, Vec<SecureField>) {
    let last_variable = *eval_point.p.last().unwrap();
    let mut half_coset0_carry_quotients = eval_point.eq_carry_quotients.clone();
    *half_coset0_carry_quotients.last_mut().unwrap() *=
        eq(&[SecureField::one()], &[last_variable]) / eq(&[SecureField::zero()], &[last_variable]);
    let half_coset1_carry_quotients = half_coset0_carry_quotients
        .iter()
        .map(|v| v.inverse())
        .collect();
    (half_coset0_carry_quotients, half_coset1_carry_quotients)
}

/// Returns the element-wise product of `a` and `b`.
fn hadamard_product(
    a: &Col<SimdBackend, SecureField>,
    b: &Col<SimdBackend, SecureField>,
) -> Col<SimdBackend, SecureField> {
    assert_eq!(a.len(), b.len());
    SecureColumn {
        data: zip_eq(&a.data, &b.data).map(|(&a, &b)| a * b).collect(),
        length: a.len(),
    }
}

#[cfg(test)]
mod tests {
    use std::array;
    use std::iter::{repeat, zip};

    use itertools::{chain, Itertools};
    use mle_coeff_column::{MleCoeffColumnComponent, MleCoeffColumnEval};
    use num_traits::{One, Zero};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::{
        build_trace, eval_carry_quotient_col, eval_eq_constraints, eval_mle_eval_constraints,
        eval_prefix_sum_constraints, gen_carry_quotient_col, MleEvalPoint, MleEvalProverComponent,
        MleEvalVerifierComponent,
    };
    use crate::constraint_framework::preprocessed_columns::IsFirst;
    use crate::constraint_framework::{assert_constraints, EvalAtRow, TraceLocationAllocator};
    use crate::core::air::{Component, ComponentProver, Components};
    use crate::core::backend::cpu::bit_reverse;
    use crate::core::backend::simd::prefix_sum::inclusive_prefix_sum;
    use crate::core::backend::simd::qm31::PackedSecureField;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::channel::Blake2sChannel;
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::secure_column::SecureColumnByCoords;
    use crate::core::lookups::mle::Mle;
    use crate::core::pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
    use crate::core::poly::BitReversedOrder;
    use crate::core::prover::{prove, verify, VerificationError};
    use crate::core::utils::coset_order_to_circle_domain_order;
    use crate::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use crate::examples::xor::gkr_lookups::accumulation::MIN_LOG_BLOWUP_FACTOR;
    use crate::examples::xor::gkr_lookups::mle_eval::eval_step_selector_with_offset;
    use crate::examples::xor::gkr_lookups::preprocessed_columns::IsStepWithOffset;

    #[test]
    fn mle_eval_prover_component() -> Result<(), VerificationError> {
        const N_VARIABLES: usize = 8;
        const COEFFS_COL_TRACE: usize = 1;
        const MLE_EVAL_TRACE: usize = 2;
        const LOG_EXPAND: u32 = 1;
        // Create the test MLE.
        let mut rng = SmallRng::seed_from_u64(0);
        let log_size = N_VARIABLES as u32;
        let size = 1 << log_size;
        let mle_coeffs = (0..size).map(|_| rng.gen::<SecureField>()).collect();
        let mle = Mle::<SimdBackend, SecureField>::new(mle_coeffs);
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());
        let claim = mle.eval_at_point(&eval_point);
        // Setup protocol.
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(log_size + LOG_EXPAND + MIN_LOG_BLOWUP_FACTOR)
                .circle_domain()
                .half_coset,
        );
        let config = PcsConfig::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<_, Blake2sMerkleChannel>::new(config, &twiddles);
        let channel = &mut Blake2sChannel::default();
        // TODO(ilya): remove the following once preproccessed columns are not mandatory.
        // Preprocessed trace
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals([]);
        tree_builder.commit(channel);
        // Build trace.
        // 1. MLE coeffs trace.
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(mle_coeff_column::build_trace(&mle));
        tree_builder.commit(channel);
        // 2. MLE eval trace (eq evals + prefix sum).
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(build_trace(&mle, &eval_point, claim));
        tree_builder.commit(channel);
        // Create components.
        let trace_location_allocator = &mut TraceLocationAllocator::default();
        let mle_coeffs_col_component = MleCoeffColumnComponent::new(
            trace_location_allocator,
            MleCoeffColumnEval::new(COEFFS_COL_TRACE, mle.n_variables()),
            SecureField::zero(),
        );
        let mle_eval_component = MleEvalProverComponent::generate(
            trace_location_allocator,
            &mle_coeffs_col_component,
            &eval_point,
            mle,
            claim,
            &twiddles,
            MLE_EVAL_TRACE,
        );
        let components: &[&dyn ComponentProver<SimdBackend>] =
            &[&mle_coeffs_col_component, &mle_eval_component];
        // Generate proof.
        let proof = prove(components, channel, commitment_scheme).unwrap();

        // Verify.
        let components = Components {
            components: components.iter().map(|&c| c as &dyn Component).collect(),
            n_preprocessed_columns: 0,
        };

        let log_sizes = components.column_log_sizes();
        let channel = &mut Blake2sChannel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(proof.config);
        commitment_scheme.commit(proof.commitments[0], &[], channel);
        commitment_scheme.commit(proof.commitments[1], &log_sizes[1], channel);
        commitment_scheme.commit(proof.commitments[2], &log_sizes[2], channel);
        verify(&components.components, channel, commitment_scheme, proof)
    }

    #[test]
    fn mle_eval_verifier_component() -> Result<(), VerificationError> {
        const N_VARIABLES: usize = 8;
        const COEFFS_COL_TRACE: usize = 1;
        const MLE_EVAL_TRACE: usize = 2;
        const CONST_TRACE: usize = 2;
        const LOG_EXPAND: u32 = 1;
        // Create the test MLE.
        let mut rng = SmallRng::seed_from_u64(0);
        let log_size = N_VARIABLES as u32;
        let size = 1 << log_size;
        let mle_coeffs = (0..size).map(|_| rng.gen::<SecureField>()).collect();
        let mle = Mle::<SimdBackend, SecureField>::new(mle_coeffs);
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());
        let claim = mle.eval_at_point(&eval_point);
        // Setup protocol.
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(log_size + LOG_EXPAND + MIN_LOG_BLOWUP_FACTOR)
                .circle_domain()
                .half_coset,
        );
        let config = PcsConfig::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<_, Blake2sMerkleChannel>::new(config, &twiddles);
        let channel = &mut Blake2sChannel::default();

        // TODO(ilya): remove the following once preproccessed columns are not mandatory.
        // Preprocessed trace
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals([]);
        tree_builder.commit(channel);

        // Build trace.
        // 1. MLE coeffs trace.
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(mle_coeff_column::build_trace(&mle));
        tree_builder.commit(channel);
        // 2. MLE eval trace (eq evals + prefix sum).
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(build_trace(&mle, &eval_point, claim));
        tree_builder.commit(channel);
        // Create components.
        let trace_location_allocator = &mut TraceLocationAllocator::default();
        let mle_coeffs_col_component = MleCoeffColumnComponent::new(
            trace_location_allocator,
            MleCoeffColumnEval::new(COEFFS_COL_TRACE, mle.n_variables()),
            SecureField::zero(),
        );
        let mle_eval_component = MleEvalProverComponent::generate(
            trace_location_allocator,
            &mle_coeffs_col_component,
            &eval_point,
            mle,
            claim,
            &twiddles,
            MLE_EVAL_TRACE,
        );
        let components: &[&dyn ComponentProver<SimdBackend>] =
            &[&mle_coeffs_col_component, &mle_eval_component];
        // Generate proof.
        let proof = prove(components, channel, commitment_scheme).unwrap();

        // Verify.
        let trace_location_allocator = &mut TraceLocationAllocator::default();
        let mle_coeffs_col_component = MleCoeffColumnComponent::new(
            trace_location_allocator,
            MleCoeffColumnEval::new(COEFFS_COL_TRACE, N_VARIABLES),
            SecureField::zero(),
        );
        let mle_eval_component = MleEvalVerifierComponent::new(
            trace_location_allocator,
            &mle_coeffs_col_component,
            &eval_point,
            claim,
            MLE_EVAL_TRACE,
        );
        let components = Components {
            components: vec![&mle_coeffs_col_component, &mle_eval_component],
            n_preprocessed_columns: 0,
        };

        let log_sizes = components.column_log_sizes();
        let channel = &mut Blake2sChannel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(proof.config);
        commitment_scheme.commit(proof.commitments[0], &[], channel);
        commitment_scheme.commit(proof.commitments[1], &log_sizes[1], channel);
        commitment_scheme.commit(proof.commitments[2], &log_sizes[2], channel);
        verify(&components.components, channel, commitment_scheme, proof)
    }

    #[test]
    fn test_mle_eval_constraints_with_log_size_5() {
        const N_VARIABLES: usize = 5;
        const COEFFS_COL_TRACE: usize = 0;
        const MLE_EVAL_TRACE: usize = 1;
        const AUX_TRACE: usize = 2;
        let mut rng = SmallRng::seed_from_u64(0);
        let log_size = N_VARIABLES as u32;
        let size = 1 << log_size;
        let mle_coeffs = (0..size).map(|_| rng.gen::<SecureField>()).collect();
        let mle = Mle::<SimdBackend, SecureField>::new(mle_coeffs);
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());
        let claim = mle.eval_at_point(&eval_point);
        let mle_eval_point = MleEvalPoint::new(&eval_point);
        let mle_eval_trace = build_trace(&mle, &eval_point, claim);
        let mle_coeffs_col_trace = mle_coeff_column::build_trace(&mle);
        let claim_shift = claim / BaseField::from(size);
        let carry_quotients_col = gen_carry_quotient_col(&eval_point).into_coordinate_evals();
        let is_first_col = [IsFirst::new(log_size).gen_column_simd()];
        let aux_trace = chain![carry_quotients_col, is_first_col].collect();
        let traces = TreeVec::new(vec![mle_coeffs_col_trace, mle_eval_trace, aux_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(log_size);

        assert_constraints(
            &trace_polys,
            trace_domain,
            |mut eval| {
                let [mle_coeff_col_eval] =
                    eval.next_extension_interaction_mask(COEFFS_COL_TRACE, [0]);
                let [carry_quotients_col_eval] =
                    eval.next_extension_interaction_mask(AUX_TRACE, [0]);
                let [is_first_eval, is_second_eval] =
                    eval.next_interaction_mask(AUX_TRACE, [0, -1]);
                eval_mle_eval_constraints(
                    MLE_EVAL_TRACE,
                    &mut eval,
                    mle_coeff_col_eval,
                    &mle_eval_point,
                    claim_shift,
                    carry_quotients_col_eval,
                    is_first_eval,
                    is_second_eval,
                )
            },
            SecureField::zero(),
        )
    }

    #[test]
    fn eq_constraints_with_4_variables() {
        const N_VARIABLES: usize = 4;
        const EQ_EVAL_TRACE: usize = 0;
        const AUX_TRACE: usize = 1;
        let mut rng = SmallRng::seed_from_u64(0);
        let mle = Mle::new(repeat(SecureField::one()).take(1 << N_VARIABLES).collect());
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());
        let mle_eval_point = MleEvalPoint::new(&eval_point);
        let trace = build_trace(&mle, &eval_point, mle.eval_at_point(&eval_point));
        let carry_quotients_col = gen_carry_quotient_col(&eval_point).into_coordinate_evals();
        let is_first_col = [IsFirst::new(N_VARIABLES as u32).gen_column_simd()];
        let aux_trace = chain![carry_quotients_col, is_first_col].collect();
        let traces = TreeVec::new(vec![trace, aux_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(N_VARIABLES as u32);

        assert_constraints(
            &trace_polys,
            trace_domain,
            |mut eval| {
                let [carry_quotients_col_eval] =
                    eval.next_extension_interaction_mask(AUX_TRACE, [0]);
                let [is_first, is_second] = eval.next_interaction_mask(AUX_TRACE, [0, -1]);
                eval_eq_constraints(
                    EQ_EVAL_TRACE,
                    &mut eval,
                    &mle_eval_point,
                    carry_quotients_col_eval,
                    is_first,
                    is_second,
                );
            },
            SecureField::zero(),
        );
    }

    #[test]
    fn eq_constraints_with_5_variables() {
        const N_VARIABLES: usize = 5;
        const EQ_EVAL_TRACE: usize = 0;
        const AUX_TRACE: usize = 1;
        let mut rng = SmallRng::seed_from_u64(0);
        let mle = Mle::new(repeat(SecureField::one()).take(1 << N_VARIABLES).collect());
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());
        let mle_eval_point = MleEvalPoint::new(&eval_point);
        let trace = build_trace(&mle, &eval_point, mle.eval_at_point(&eval_point));
        let carry_quotients_col = gen_carry_quotient_col(&eval_point).into_coordinate_evals();
        let is_first_col = [IsFirst::new(N_VARIABLES as u32).gen_column_simd()];
        let aux_trace = chain![carry_quotients_col, is_first_col].collect();
        let traces = TreeVec::new(vec![trace, aux_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(N_VARIABLES as u32);

        assert_constraints(
            &trace_polys,
            trace_domain,
            |mut eval| {
                let [carry_quotients_col_eval] =
                    eval.next_extension_interaction_mask(AUX_TRACE, [0]);
                let [is_first, is_second] = eval.next_interaction_mask(AUX_TRACE, [0, -1]);
                eval_eq_constraints(
                    EQ_EVAL_TRACE,
                    &mut eval,
                    &mle_eval_point,
                    carry_quotients_col_eval,
                    is_first,
                    is_second,
                );
            },
            SecureField::zero(),
        );
    }

    #[test]
    fn eq_constraints_with_8_variables() {
        const N_VARIABLES: usize = 8;
        const EQ_EVAL_TRACE: usize = 0;
        const AUX_TRACE: usize = 1;
        let mut rng = SmallRng::seed_from_u64(0);
        let mle = Mle::new(repeat(SecureField::one()).take(1 << N_VARIABLES).collect());
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());
        let mle_eval_point = MleEvalPoint::new(&eval_point);
        let trace = build_trace(&mle, &eval_point, mle.eval_at_point(&eval_point));
        let carry_quotients_col = gen_carry_quotient_col(&eval_point).into_coordinate_evals();
        let is_first_col = [IsFirst::new(N_VARIABLES as u32).gen_column_simd()];
        let aux_trace = chain![carry_quotients_col, is_first_col].collect();
        let traces = TreeVec::new(vec![trace, aux_trace]);
        let trace_polys = traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(N_VARIABLES as u32);

        assert_constraints(
            &trace_polys,
            trace_domain,
            |mut eval| {
                let [carry_quotients_col_eval] =
                    eval.next_extension_interaction_mask(AUX_TRACE, [0]);
                let [is_first, is_second] = eval.next_interaction_mask(AUX_TRACE, [0, -1]);
                eval_eq_constraints(
                    EQ_EVAL_TRACE,
                    &mut eval,
                    &mle_eval_point,
                    carry_quotients_col_eval,
                    is_first,
                    is_second,
                );
            },
            SecureField::zero(),
        );
    }

    #[test]
    fn inclusive_prefix_sum_constraints_with_log_size_5() {
        const LOG_SIZE: u32 = 5;
        let mut rng = SmallRng::seed_from_u64(0);
        let vals = (0..1 << LOG_SIZE).map(|_| rng.gen()).collect_vec();
        let cumulative_sum = vals.iter().sum::<SecureField>();
        let cumulative_sum_shift = cumulative_sum / BaseField::from(vals.len());
        let trace = TreeVec::new(vec![gen_prefix_sum_trace(vals)]);
        let trace_polys = trace.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect());
        let trace_domain = CanonicCoset::new(LOG_SIZE);

        assert_constraints(
            &trace_polys,
            trace_domain,
            |mut eval| {
                let [row_diff] = eval.next_extension_interaction_mask(0, [0]);
                eval_prefix_sum_constraints(0, &mut eval, row_diff, cumulative_sum_shift)
            },
            SecureField::zero(),
        );
    }

    #[test]
    fn eval_step_selector_with_offset_works() {
        const LOG_SIZE: u32 = 5;
        const OFFSET: usize = 1;
        const LOG_STEP: u32 = 2;
        let coset = CanonicCoset::new(LOG_SIZE).coset();
        let col_eval = IsStepWithOffset::new(LOG_SIZE, LOG_STEP, OFFSET).gen_column_simd();
        let col_poly = col_eval.interpolate();
        let p = SECURE_FIELD_CIRCLE_GEN;

        let eval = eval_step_selector_with_offset(coset, OFFSET, LOG_STEP, p);

        assert_eq!(eval, col_poly.eval_at_point(p));
    }

    #[test]
    fn eval_carry_quotient_col_works() {
        const N_VARIABLES: usize = 5;
        let mut rng = SmallRng::seed_from_u64(0);
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());
        let mle_eval_point = MleEvalPoint::new(&eval_point);
        let col_eval = gen_carry_quotient_col(&eval_point);
        let twiddles = SimdBackend::precompute_twiddles(col_eval.domain.half_coset);
        let col_poly = col_eval.interpolate_with_twiddles(&twiddles);
        let p = SECURE_FIELD_CIRCLE_GEN;

        let eval = eval_carry_quotient_col(&mle_eval_point, p);

        assert_eq!(eval, col_poly.eval_at_point(p));
    }

    /// Generates a trace.
    ///
    /// Trace structure:
    ///
    /// ```text
    /// ---------------------------------------------------------
    /// |           Values          |     Values prefix sum     |
    /// ---------------------------------------------------------
    /// |  c0  |  c1  |  c2  |  c3  |  c4  |  c5  |  c6  |  c7  |
    /// ---------------------------------------------------------
    /// ```
    fn gen_prefix_sum_trace(
        values: Vec<SecureField>,
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        assert!(values.len().is_power_of_two());

        let vals_circle_domain_order = coset_order_to_circle_domain_order(&values);
        let mut vals_bit_rev_circle_domain_order = vals_circle_domain_order;
        bit_reverse(&mut vals_bit_rev_circle_domain_order);
        let vals_secure_col: SecureColumnByCoords<SimdBackend> =
            vals_bit_rev_circle_domain_order.into_iter().collect();
        let vals_cols = vals_secure_col.columns;

        let cumulative_sum = values.iter().sum::<SecureField>();
        let cumulative_sum_shift = cumulative_sum / BaseField::from(values.len());
        let packed_cumulative_sum_shift = PackedSecureField::broadcast(cumulative_sum_shift);
        let packed_shifts = packed_cumulative_sum_shift.into_packed_m31s();
        let mut shifted_cols = vals_cols.clone();
        zip(&mut shifted_cols, packed_shifts)
            .for_each(|(col, packed_shift)| col.data.iter_mut().for_each(|v| *v -= packed_shift));
        let shifted_prefix_sum_cols = shifted_cols.map(inclusive_prefix_sum);

        let log_size = values.len().ilog2();
        let trace_domain = CanonicCoset::new(log_size).circle_domain();

        chain![vals_cols, shifted_prefix_sum_cols]
            .map(|c| CircleEvaluation::new(trace_domain, c))
            .collect()
    }

    mod mle_coeff_column {
        use num_traits::{One, Zero};

        use crate::constraint_framework::{
            EvalAtRow, FrameworkComponent, FrameworkEval, PointEvaluator,
        };
        use crate::core::air::accumulation::PointEvaluationAccumulator;
        use crate::core::backend::simd::SimdBackend;
        use crate::core::circle::CirclePoint;
        use crate::core::fields::m31::BaseField;
        use crate::core::fields::qm31::SecureField;
        use crate::core::lookups::mle::Mle;
        use crate::core::pcs::TreeVec;
        use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, SecureEvaluation};
        use crate::core::poly::BitReversedOrder;
        use crate::core::ColumnVec;
        use crate::examples::xor::gkr_lookups::mle_eval::MleCoeffColumnOracle;

        pub type MleCoeffColumnComponent = FrameworkComponent<MleCoeffColumnEval>;

        pub struct MleCoeffColumnEval {
            interaction: usize,
            n_variables: usize,
        }

        impl MleCoeffColumnEval {
            pub const fn new(interaction: usize, n_variables: usize) -> Self {
                Self {
                    interaction,
                    n_variables,
                }
            }
        }

        impl FrameworkEval for MleCoeffColumnEval {
            fn log_size(&self) -> u32 {
                self.n_variables as u32
            }

            fn max_constraint_log_degree_bound(&self) -> u32 {
                self.log_size()
            }

            fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
                let _ = eval_mle_coeff_col(self.interaction, &mut eval);
                eval
            }
        }

        impl MleCoeffColumnOracle for MleCoeffColumnComponent {
            fn evaluate_at_point(
                &self,
                _point: CirclePoint<SecureField>,
                mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
            ) -> SecureField {
                // Create dummy point evaluator just to extract the value we need from the mask
                let mut accumulator = PointEvaluationAccumulator::new(SecureField::one());
                let mut eval = PointEvaluator::new(
                    mask.sub_tree(self.trace_locations()),
                    &mut accumulator,
                    SecureField::one(),
                    self.log_size(),
                    SecureField::zero(),
                );

                eval_mle_coeff_col(self.interaction, &mut eval)
            }
        }

        fn eval_mle_coeff_col<E: EvalAtRow>(interaction: usize, eval: &mut E) -> E::EF {
            let [mle_coeff_col_eval] = eval.next_extension_interaction_mask(interaction, [0]);
            mle_coeff_col_eval
        }

        /// Generates a trace.
        ///
        /// Trace structure:
        ///
        /// ```text
        /// -----------------------------
        /// |      MLE coeffs col       |
        /// -----------------------------
        /// |  c0  |  c1  |  c2  |  c3  |
        /// -----------------------------
        /// ```
        pub fn build_trace(
            mle: &Mle<SimdBackend, SecureField>,
        ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
            let log_size = mle.n_variables() as u32;
            let trace_domain = CanonicCoset::new(log_size).circle_domain();
            let mle_coeffs_col_by_coords = mle.clone().into_evals().into_secure_column_by_coords();
            SecureEvaluation::new(trace_domain, mle_coeffs_col_by_coords)
                .into_coordinate_evals()
                .into_iter()
                .collect()
        }
    }
}
