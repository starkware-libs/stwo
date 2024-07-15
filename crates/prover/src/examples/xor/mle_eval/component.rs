use num_traits::Zero;

use crate::constraint_framework::{EvalAtRow, InfoEvaluator};
use crate::core::air::accumulation::PointEvaluationAccumulator;
use crate::core::air::Component;
use crate::core::backend::Backend;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::lookups::mle::Mle;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::CanonicCoset;
use crate::core::{ColumnVec, InteractionElements, LookupValues};
use crate::examples::xor::eq_evals::constraints::{EqEvalsMaskAt, EqEvalsMaskIs, PointMeta};
use crate::examples::xor::mle_eval::constraints::mle_eval_check;
use crate::examples::xor::prefix_sum::{PrefixSumMaskAt, PrefixSumMaskIs};

/// Log constraint blowup degree.
const LOG_EXPAND: u32 = 1;

/// Interaction trace commitment index.
const _BASE_TRACE_INDEX: usize = 0;

/// Interaction trace commitment index.
const INTERACTION_TRACE_INDEX: usize = 1;

/// Constants trace commitment index.
const CONSTANTS_TRACE_INDEX: usize = 2;

#[derive(Debug, Clone)]
struct MleEvalComponentVerifier {
    claim: SecureField,
    eval_point: Vec<SecureField>,
}

impl MleEvalComponentVerifier {
    pub fn new(eval_point: &[SecureField], claim: SecureField) -> Self {
        let eval_point = eval_point.to_vec();
        Self { eval_point, claim }
    }

    // TODO: Duplicated impl to `MleEvalComponentProver`.
    pub fn log_column_size(&self) -> u32 {
        self.eval_point.len() as u32
    }

    // TODO: Duplicated impl to `MleEvalComponentProver`.
    pub fn n_columns(&self) -> usize {
        SECURE_EXTENSION_DEGREE * 2
    }
}

impl Component for MleEvalComponentVerifier {
    fn n_constraints(&self) -> usize {
        mle_eval_info(self.eval_point.len()).n_constraints
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_column_size() + LOG_EXPAND
    }

    fn n_interaction_phases(&self) -> u32 {
        0
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        let mask_offsets = mle_eval_info(self.eval_point.len()).mask_offsets;
        let log_col_size = self.log_column_size();
        TreeVec::new(
            mask_offsets
                .iter()
                .map(|trace| vec![log_col_size; trace.len()])
                .collect(),
        )
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let trace_step = CanonicCoset::new(self.log_column_size()).step();
        let counter = mle_eval_info(self.eval_point.len());
        counter.mask_offsets.map(|tree_mask| {
            tree_mask
                .iter()
                .map(|col_mask| {
                    col_mask
                        .iter()
                        .map(|&off| point + (trace_step * off).into_ef())
                        .collect()
                })
                .collect()
        })
    }

    // TODO: Can't implement this yet.
    fn evaluate_constraint_quotients_at_point(
        &self,
        _point: CirclePoint<SecureField>,
        _mask: &TreeVec<Vec<Vec<SecureField>>>,
        _accumulator: &mut PointEvaluationAccumulator,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        unimplemented!()
    }
}

#[derive(Debug, Clone)]
pub struct MleEvalComponentProver<B: Backend> {
    claim: SecureField,
    eval_point: Vec<SecureField>,
    mle: Mle<B, SecureField>,
}

impl<B: Backend> MleEvalComponentProver<B> {
    pub fn new(mle: Mle<B, SecureField>, eval_point: &[SecureField], claim: SecureField) -> Self {
        #[cfg(test)]
        debug_assert_eq!(mle.eval_at_point(eval_point), claim);
        let eval_point = eval_point.to_vec();
        Self {
            claim,
            eval_point,
            mle,
        }
    }

    pub fn to_verifier_component(&self) -> MleEvalComponentVerifier {
        MleEvalComponentVerifier::new(&self.eval_point, self.claim)
    }
}

impl<B: Backend> Component for MleEvalComponentProver<B> {
    fn n_constraints(&self) -> usize {
        self.to_verifier_component().n_constraints()
    }

    // TODO: Duplicated impl to `MleEvalComponentVerifier`.
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.to_verifier_component()
            .max_constraint_log_degree_bound()
    }

    // TODO: Duplicated impl to `MleEvalComponentVerifier`.
    fn n_interaction_phases(&self) -> u32 {
        self.to_verifier_component().n_interaction_phases()
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        self.to_verifier_component().trace_log_degree_bounds()
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        self.to_verifier_component().mask_points(point)
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        _point: CirclePoint<SecureField>,
        _mask: &TreeVec<Vec<Vec<SecureField>>>,
        _accumulator: &mut PointEvaluationAccumulator,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        unimplemented!()
    }
}

fn mle_eval_check_wrapper<E: EvalAtRow>(
    eval: &mut E,
    eval_point: &[SecureField],
    mle_coeff: SecureField,
    mle_claim: SecureField,
) {
    fn wrapper_const<E: EvalAtRow, const N_VARIABLES: usize>(
        eval: &mut E,
        eval_point: [SecureField; N_VARIABLES],
        mle_coeff: SecureField,
        mle_claim: SecureField,
    ) where
        // Ensure the type exists.
        [(); N_VARIABLES + 1]:,
    {
        // Flags for first and last point in trace domain.
        let [first, last] = eval.next_interaction_mask(CONSTANTS_TRACE_INDEX, [0, 1]);
        let point_meta = PointMeta::new(eval_point);
        let eq_evals_at = EqEvalsMaskAt::draw::<INTERACTION_TRACE_INDEX>(eval);
        let eq_evals_is = EqEvalsMaskIs::draw_steps::<CONSTANTS_TRACE_INDEX>(eval, first);
        let prefix_sum_at = PrefixSumMaskAt::draw::<INTERACTION_TRACE_INDEX>(eval);
        let prefix_sum_is = PrefixSumMaskIs { first, last };
        mle_eval_check(
            eval,
            point_meta,
            // TODO: Consider adding `EF: ... + From<SecureField>`.
            E::EF::zero() + mle_coeff,
            mle_claim,
            &eq_evals_at,
            &eq_evals_is,
            &prefix_sum_at,
            &prefix_sum_is,
        );
    }

    match eval_point.len() {
        0 => wrapper_const::<E, 0>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        1 => wrapper_const::<E, 1>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        2 => wrapper_const::<E, 2>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        3 => wrapper_const::<E, 3>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        4 => wrapper_const::<E, 4>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        5 => wrapper_const::<E, 5>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        6 => wrapper_const::<E, 6>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        7 => wrapper_const::<E, 7>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        8 => wrapper_const::<E, 8>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        9 => wrapper_const::<E, 9>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        10 => wrapper_const::<E, 10>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        11 => wrapper_const::<E, 11>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        12 => wrapper_const::<E, 12>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        13 => wrapper_const::<E, 13>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        14 => wrapper_const::<E, 14>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        15 => wrapper_const::<E, 15>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        16 => wrapper_const::<E, 16>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        17 => wrapper_const::<E, 17>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        18 => wrapper_const::<E, 18>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        19 => wrapper_const::<E, 19>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        20 => wrapper_const::<E, 20>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        21 => wrapper_const::<E, 21>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        22 => wrapper_const::<E, 22>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        23 => wrapper_const::<E, 23>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        24 => wrapper_const::<E, 24>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        25 => wrapper_const::<E, 25>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        26 => wrapper_const::<E, 26>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        27 => wrapper_const::<E, 27>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        28 => wrapper_const::<E, 28>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        29 => wrapper_const::<E, 29>(eval, eval_point.try_into().unwrap(), mle_coeff, mle_claim),
        _ => panic!("unsupported"),
    }
}

fn mle_eval_info(mle_n_variables: usize) -> InfoEvaluator {
    let eval_point = vec![SecureField::zero(); mle_n_variables];
    let mut eval = InfoEvaluator::default();
    mle_eval_check_wrapper(
        &mut eval,
        &eval_point,
        SecureField::zero(),
        SecureField::zero(),
    );
    eval
}

#[cfg(test)]
mod tests {
    // use tracing::{instrument, span, Level};
    // use super::{MleEvalComponentProver, LOG_EXPAND};
    // use crate::core::backend::simd::SimdBackend;
    // use crate::core::channel::{Blake2sChannel, Channel};
    // use crate::core::fields::m31::BaseField;
    // use crate::core::fields::qm31::SecureField;
    // use crate::core::fields::IntoSlice;
    // use crate::core::lookups::mle::Mle;
    // use crate::core::pcs::CommitmentSchemeProver;
    // use crate::core::poly::circle::{CanonicCoset, PolyOps};
    // use crate::core::prover::{StarkProof, VerificationError, LOG_BLOWUP_FACTOR};
    // use crate::core::vcs::blake2_hash::Blake2sHasher;
    // use crate::core::vcs::hasher::Hasher;
    // use crate::examples::xor::eq_evals::trace::{gen_constants_trace, gen_evals_trace};

    // #[instrument(skip_all)]
    // fn prove_eq_evals(
    //     mle: Mle<SimdBackend, SecureField>,
    //     eval_point: &[SecureField],
    // ) -> (MleEvalComponentProver<SimdBackend>, StarkProof) {
    //     let n_variables = eval_point.len();
    //     let log_n_rows = n_variables as u32;

    //     // Precompute twiddles.
    //     let span = span!(Level::INFO, "Precompute twiddles").entered();
    //     let twiddles = SimdBackend::precompute_twiddles(
    //         CanonicCoset::new(log_n_rows + LOG_EXPAND + LOG_BLOWUP_FACTOR)
    //             .circle_domain()
    //             .half_coset,
    //     );
    //     span.exit();

    //     // Setup protocol.
    //     let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
    //     let commitment_scheme = &mut CommitmentSchemeProver::new(LOG_BLOWUP_FACTOR);

    //     // Base trace.
    //     let span = span!(Level::INFO, "Trace").entered();
    //     let span1 = span!(Level::INFO, "Generation").entered();
    //     let base_trace = gen_evals_trace(eval_point);
    //     span1.exit();
    //     commitment_scheme.commit_on_evals(base_trace, channel, &twiddles);
    //     span.exit();

    //     // Interaction trace.
    //     let span = span!(Level::INFO, "Trace").entered();
    //     let span1 = span!(Level::INFO, "Generation").entered();
    //     let base_trace = gen_evals_trace(eval_point);
    //     span1.exit();
    //     commitment_scheme.commit_on_evals(base_trace, channel, &twiddles);
    //     span.exit();

    //     // Constants trace.
    //     let span = span!(Level::INFO, "Constants").entered();
    //     let constants_trace = gen_constants_trace(n_variables);
    //     commitment_scheme.commit_on_evals(constants_trace, channel, &twiddles);
    //     span.exit();

    //     // Prove constraints.
    //     let component = EqEvalsComponent::new(eval_point);
    //     let air = EqEvalsAir { component };
    //     let proof = prove_without_commit::<SimdBackend>(
    //         &air,
    //         channel,
    //         &InteractionElements::default(),
    //         &twiddles,
    //         commitment_scheme,
    //     )
    //     .unwrap();

    //     (air, proof)
    // }

    // #[instrument(skip_all)]
    // fn verify_eq_evals(
    //     eval_point: &[SecureField],
    //     proof: StarkProof,
    // ) -> Result<(), VerificationError> {
    //     let n_variables = eval_point.len();
    //     let component = EqEvalsComponent::new(eval_point);
    //     let air = EqEvalsAir { component };

    //     // Verify.
    //     // TODO: Create Air instance independently.
    //     let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
    //     let commitment_scheme = &mut CommitmentSchemeVerifier::new();
    //     let sizes = air.column_log_sizes();

    //     // Base trace columns.
    //     commitment_scheme.commit(proof.commitments[BASE_TRACE], &sizes[BASE_TRACE], channel);

    //     // Constant columns.
    //     let log_constant_colum_size = n_variables as u32;
    //     let n_constant_columns = n_variables;
    //     commitment_scheme.commit(
    //         proof.commitments[CONST_TRACE],
    //         &vec![log_constant_colum_size; n_constant_columns],
    //         channel,
    //     );

    //     verify_without_commit(
    //         &air,
    //         channel,
    //         &InteractionElements::default(),
    //         commitment_scheme,
    //         proof,
    //     )
    // }
}
