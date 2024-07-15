#![allow(dead_code)]

use tracing::{span, Level};

use super::constraints::{eq_evals_check, PointMeta};
use super::trace::{gen_constants_trace, gen_evals_trace};
use crate::constraint_framework::{EvalAtRow, InfoEvaluator, PointEvaluator, SimdDomainEvaluator};
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Air, AirExt, AirProver, Component, ComponentProver, ComponentTrace};
use crate::core::backend::simd::column::BaseFieldVec;
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Column, ColumnOps};
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::circle::CirclePoint;
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::fields::{FieldExpOps, FieldOps, IntoSlice};
use crate::core::pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, TreeVec};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::{
    prove_without_commit, verify_without_commit, StarkProof, VerificationError, LOG_BLOWUP_FACTOR,
};
use crate::core::vcs::blake2_hash::Blake2sHasher;
use crate::core::vcs::hasher::Hasher;
use crate::core::{ColumnVec, InteractionElements, LookupValues};
use crate::examples::xor::eq_evals::constraints::{EqEvalsMaskAt, EqEvalsMaskIs};
use crate::trace_generation::{AirTraceGenerator, AirTraceVerifier};

/// Log constraint blowup degree.
const LOG_EXPAND: u32 = 1;

/// Base trace commitment index.
const BASE_TRACE: usize = 0;

/// Constants trace commitment index.
const CONST_TRACE: usize = 1;

#[derive(Debug, Clone)]
struct EqEvalsComponent {
    eval_point: Vec<SecureField>,
}

impl EqEvalsComponent {
    pub fn new(eval_point: &[SecureField]) -> Self {
        let eval_point = eval_point.to_vec();
        Self { eval_point }
    }

    pub fn log_column_size(&self) -> u32 {
        self.eval_point.len() as u32
    }

    pub fn n_columns(&self) -> usize {
        // Single SecureField column (comprising of `SECURE_EXTENSION_DEGREE` base field columns).
        SECURE_EXTENSION_DEGREE
    }

    fn evaluate_constraint_quotients_on_domain_const<const N_VARIABLES: usize>(
        &self,
        trace: &ComponentTrace<'_, SimdBackend>,
        accumulator: &mut DomainEvaluationAccumulator<SimdBackend>,
    ) where
        // Ensure the type exists.
        [(); N_VARIABLES + 1]:,
    {
        assert_eq!(trace.evals[0].len(), self.n_columns());
        let trace_domain = CanonicCoset::new(self.log_column_size()).coset;
        let eval_domain = CanonicCoset::new(self.log_column_size() + LOG_EXPAND).circle_domain();

        // Denoms.
        let span = span!(Level::INFO, "Constraint eval denominators").entered();
        let vanish_on_trace_domain_evals =
            BaseFieldVec::from_iter(eval_domain.iter().map(|p| coset_vanishing(trace_domain, p)));
        let mut vanish_on_trace_domain_evals_bit_rev = vanish_on_trace_domain_evals;
        <SimdBackend as ColumnOps<BaseField>>::bit_reverse_column(
            &mut vanish_on_trace_domain_evals_bit_rev,
        );
        let mut vanish_on_trace_domain_evals_bit_rev_inv =
            BaseFieldVec::zeros(vanish_on_trace_domain_evals_bit_rev.len());
        <SimdBackend as FieldOps<BaseField>>::batch_inverse(
            &vanish_on_trace_domain_evals_bit_rev,
            &mut vanish_on_trace_domain_evals_bit_rev_inv,
        );
        span.exit();

        let _span = span!(Level::INFO, "Constraint pointwise eval").entered();
        let constraint_log_degree_bound = self.max_constraint_log_degree_bound();
        let n_constraints = self.n_constraints();
        let [acc_col] = accumulator.columns([(constraint_log_degree_bound, n_constraints)]);
        let mut pows = acc_col.random_coeff_powers.clone();
        pows.reverse();

        let eval_point: [SecureField; N_VARIABLES] = self.eval_point.as_slice().try_into().unwrap();
        let point_meta = PointMeta::new(eval_point);

        let log_n_trace_rows = trace_domain.log_size();
        let log_n_packed_eval_rows = log_n_trace_rows + LOG_EXPAND - LOG_N_LANES;

        for packed_row_i in 0..1 << log_n_packed_eval_rows {
            let mut eval = SimdDomainEvaluator::new(
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
    }
}

impl Component for EqEvalsComponent {
    fn n_constraints(&self) -> usize {
        eq_evals_info(&self.eval_point).n_constraints
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_column_size() + LOG_EXPAND
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::new(
            eq_evals_info(&self.eval_point)
                .mask_offsets
                .iter()
                .map(|tree_masks| vec![self.log_column_size(); tree_masks.len()])
                .collect(),
        )
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let trace_step = CanonicCoset::new(self.log_column_size()).step();
        let counter = eq_evals_info(&self.eval_point);
        counter.mask_offsets.map(|tree_mask| {
            tree_mask
                .iter()
                .map(|col_mask| {
                    col_mask
                        .iter()
                        .map(|&off| point + trace_step.mul_signed(off).into_ef())
                        .collect()
                })
                .collect()
        })
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<Vec<Vec<SecureField>>>,
        accumulator: &mut PointEvaluationAccumulator,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        let trace_domain = CanonicCoset::new(self.log_column_size()).coset();
        let vanish_on_trace_domain = coset_vanishing(trace_domain, point);
        let vanish_on_trace_domain_inv = vanish_on_trace_domain.inverse();
        let mut eval = PointEvaluator::new(mask.as_ref(), accumulator, vanish_on_trace_domain_inv);
        eq_evals_check_wrapper(&mut eval, &self.eval_point);
    }
}

impl ComponentProver<SimdBackend> for EqEvalsComponent {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, SimdBackend>,
        accumulator: &mut DomainEvaluationAccumulator<SimdBackend>,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        match self.eval_point.len() {
            0 => self.evaluate_constraint_quotients_on_domain_const::<0>(trace, accumulator),
            1 => self.evaluate_constraint_quotients_on_domain_const::<1>(trace, accumulator),
            2 => self.evaluate_constraint_quotients_on_domain_const::<2>(trace, accumulator),
            3 => self.evaluate_constraint_quotients_on_domain_const::<3>(trace, accumulator),
            4 => self.evaluate_constraint_quotients_on_domain_const::<4>(trace, accumulator),
            5 => self.evaluate_constraint_quotients_on_domain_const::<5>(trace, accumulator),
            6 => self.evaluate_constraint_quotients_on_domain_const::<6>(trace, accumulator),
            7 => self.evaluate_constraint_quotients_on_domain_const::<7>(trace, accumulator),
            8 => self.evaluate_constraint_quotients_on_domain_const::<8>(trace, accumulator),
            9 => self.evaluate_constraint_quotients_on_domain_const::<9>(trace, accumulator),
            10 => self.evaluate_constraint_quotients_on_domain_const::<10>(trace, accumulator),
            11 => self.evaluate_constraint_quotients_on_domain_const::<11>(trace, accumulator),
            12 => self.evaluate_constraint_quotients_on_domain_const::<12>(trace, accumulator),
            13 => self.evaluate_constraint_quotients_on_domain_const::<13>(trace, accumulator),
            14 => self.evaluate_constraint_quotients_on_domain_const::<14>(trace, accumulator),
            15 => self.evaluate_constraint_quotients_on_domain_const::<15>(trace, accumulator),
            16 => self.evaluate_constraint_quotients_on_domain_const::<16>(trace, accumulator),
            17 => self.evaluate_constraint_quotients_on_domain_const::<17>(trace, accumulator),
            18 => self.evaluate_constraint_quotients_on_domain_const::<18>(trace, accumulator),
            19 => self.evaluate_constraint_quotients_on_domain_const::<19>(trace, accumulator),
            20 => self.evaluate_constraint_quotients_on_domain_const::<20>(trace, accumulator),
            21 => self.evaluate_constraint_quotients_on_domain_const::<21>(trace, accumulator),
            22 => self.evaluate_constraint_quotients_on_domain_const::<22>(trace, accumulator),
            23 => self.evaluate_constraint_quotients_on_domain_const::<23>(trace, accumulator),
            24 => self.evaluate_constraint_quotients_on_domain_const::<24>(trace, accumulator),
            25 => self.evaluate_constraint_quotients_on_domain_const::<25>(trace, accumulator),
            26 => self.evaluate_constraint_quotients_on_domain_const::<26>(trace, accumulator),
            27 => self.evaluate_constraint_quotients_on_domain_const::<27>(trace, accumulator),
            28 => self.evaluate_constraint_quotients_on_domain_const::<28>(trace, accumulator),
            29 => self.evaluate_constraint_quotients_on_domain_const::<29>(trace, accumulator),
            _ => panic!("unsupported"),
        }
    }

    fn lookup_values(&self, _trace: &ComponentTrace<'_, SimdBackend>) -> LookupValues {
        LookupValues::default()
    }
}

#[derive(Debug, Clone)]
struct EqEvalsAir {
    component: EqEvalsComponent,
}

impl Air for EqEvalsAir {
    fn components(&self) -> Vec<&dyn Component> {
        vec![&self.component]
    }

    fn verify_lookups(&self, _lookup_values: &LookupValues) -> Result<(), VerificationError> {
        Ok(())
    }
}

impl AirTraceVerifier for EqEvalsAir {
    fn interaction_elements(&self, _channel: &mut Blake2sChannel) -> InteractionElements {
        InteractionElements::default()
    }
}

impl AirProver<SimdBackend> for EqEvalsAir {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<SimdBackend>> {
        vec![&self.component]
    }
}

impl AirTraceGenerator<SimdBackend> for EqEvalsAir {
    fn interact(
        &self,
        _trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        _elements: &InteractionElements,
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        vec![]
    }

    fn to_air_prover(&self) -> impl AirProver<SimdBackend> {
        self.clone()
    }

    fn composition_log_degree_bound(&self) -> u32 {
        self.component.max_constraint_log_degree_bound()
    }
}

fn eq_evals_check_wrapper<E: EvalAtRow>(eval: &mut E, eval_point: &[SecureField]) {
    fn wrapper_const<E: EvalAtRow, const N_VARIABLES: usize>(
        eval: &mut E,
        eval_point: [SecureField; N_VARIABLES],
    ) where
        // Ensure the type exists.
        [(); N_VARIABLES + 1]:,
    {
        let point_meta = PointMeta::new(eval_point);
        let at = EqEvalsMaskAt::draw::<BASE_TRACE>(eval);
        let is = EqEvalsMaskIs::draw::<CONST_TRACE>(eval);
        eq_evals_check(eval, point_meta, &at, &is);
    }

    match eval_point.len() {
        0 => wrapper_const::<E, 0>(eval, eval_point.try_into().unwrap()),
        1 => wrapper_const::<E, 1>(eval, eval_point.try_into().unwrap()),
        2 => wrapper_const::<E, 2>(eval, eval_point.try_into().unwrap()),
        3 => wrapper_const::<E, 3>(eval, eval_point.try_into().unwrap()),
        4 => wrapper_const::<E, 4>(eval, eval_point.try_into().unwrap()),
        5 => wrapper_const::<E, 5>(eval, eval_point.try_into().unwrap()),
        6 => wrapper_const::<E, 6>(eval, eval_point.try_into().unwrap()),
        7 => wrapper_const::<E, 7>(eval, eval_point.try_into().unwrap()),
        8 => wrapper_const::<E, 8>(eval, eval_point.try_into().unwrap()),
        9 => wrapper_const::<E, 9>(eval, eval_point.try_into().unwrap()),
        10 => wrapper_const::<E, 10>(eval, eval_point.try_into().unwrap()),
        11 => wrapper_const::<E, 11>(eval, eval_point.try_into().unwrap()),
        12 => wrapper_const::<E, 12>(eval, eval_point.try_into().unwrap()),
        13 => wrapper_const::<E, 13>(eval, eval_point.try_into().unwrap()),
        14 => wrapper_const::<E, 14>(eval, eval_point.try_into().unwrap()),
        15 => wrapper_const::<E, 15>(eval, eval_point.try_into().unwrap()),
        16 => wrapper_const::<E, 16>(eval, eval_point.try_into().unwrap()),
        17 => wrapper_const::<E, 17>(eval, eval_point.try_into().unwrap()),
        18 => wrapper_const::<E, 18>(eval, eval_point.try_into().unwrap()),
        19 => wrapper_const::<E, 19>(eval, eval_point.try_into().unwrap()),
        20 => wrapper_const::<E, 20>(eval, eval_point.try_into().unwrap()),
        21 => wrapper_const::<E, 21>(eval, eval_point.try_into().unwrap()),
        22 => wrapper_const::<E, 22>(eval, eval_point.try_into().unwrap()),
        23 => wrapper_const::<E, 23>(eval, eval_point.try_into().unwrap()),
        24 => wrapper_const::<E, 24>(eval, eval_point.try_into().unwrap()),
        25 => wrapper_const::<E, 25>(eval, eval_point.try_into().unwrap()),
        26 => wrapper_const::<E, 26>(eval, eval_point.try_into().unwrap()),
        27 => wrapper_const::<E, 27>(eval, eval_point.try_into().unwrap()),
        28 => wrapper_const::<E, 28>(eval, eval_point.try_into().unwrap()),
        29 => wrapper_const::<E, 29>(eval, eval_point.try_into().unwrap()),
        _ => panic!("unsupported"),
    }
}

fn eq_evals_info(eval_point: &[SecureField]) -> InfoEvaluator {
    let mut eval = InfoEvaluator::default();
    eq_evals_check_wrapper(&mut eval, eval_point);
    eval
}

fn prove_eq_evals(eval_point: &[SecureField]) -> (EqEvalsAir, StarkProof) {
    let n_variables = eval_point.len();
    let log_n_rows = n_variables as u32;

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + LOG_EXPAND + LOG_BLOWUP_FACTOR)
            .circle_domain()
            .half_coset,
    );
    span.exit();

    // Setup protocol.
    let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
    let commitment_scheme = &mut CommitmentSchemeProver::new(LOG_BLOWUP_FACTOR);

    // Trace.
    let span = span!(Level::INFO, "Trace").entered();
    let span1 = span!(Level::INFO, "Generation").entered();
    let base_trace = gen_evals_trace(eval_point);
    span1.exit();
    commitment_scheme.commit_on_evals(base_trace, channel, &twiddles);
    span.exit();

    // Constant trace.
    let span = span!(Level::INFO, "Constant").entered();
    let constants_trace = gen_constants_trace(n_variables);
    commitment_scheme.commit_on_evals(constants_trace, channel, &twiddles);
    span.exit();

    // Prove constraints.
    let component = EqEvalsComponent::new(eval_point);
    let air = EqEvalsAir { component };
    let proof = prove_without_commit::<SimdBackend>(
        &air,
        channel,
        &InteractionElements::default(),
        &twiddles,
        commitment_scheme,
    )
    .unwrap();

    (air, proof)
}

fn verify_eq_evals(eval_point: &[SecureField], proof: StarkProof) -> Result<(), VerificationError> {
    let n_variables = eval_point.len();
    let component = EqEvalsComponent::new(eval_point);
    let air = EqEvalsAir { component };

    // Verify.
    // TODO: Create Air instance independently.
    let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
    let commitment_scheme = &mut CommitmentSchemeVerifier::new();
    let sizes = air.column_log_sizes();

    // Base trace columns.
    commitment_scheme.commit(proof.commitments[BASE_TRACE], &sizes[BASE_TRACE], channel);

    // Constant columns.
    let log_constant_colum_size = n_variables as u32;
    let n_constant_columns = n_variables;
    commitment_scheme.commit(
        proof.commitments[CONST_TRACE],
        &vec![log_constant_colum_size; n_constant_columns],
        channel,
    );

    verify_without_commit(
        &air,
        channel,
        &InteractionElements::default(),
        commitment_scheme,
        proof,
    )
}

#[cfg(test)]
mod tests {
    use std::array;

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use test_log::test;

    use crate::core::fields::qm31::SecureField;
    use crate::core::prover::VerificationError;
    use crate::examples::xor::eq_evals::component::{prove_eq_evals, verify_eq_evals};

    #[test]
    fn prove_eq_evals_with_8_variables_works() -> Result<(), VerificationError> {
        const N_VARIABLES: usize = 8;
        let mut rng = SmallRng::seed_from_u64(0);
        let eval_point: [SecureField; N_VARIABLES] = array::from_fn(|_| rng.gen());

        let (_air, proof) = prove_eq_evals(&eval_point);

        verify_eq_evals(&eval_point, proof)
    }
}
