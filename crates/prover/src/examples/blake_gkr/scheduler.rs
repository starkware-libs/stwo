use std::array;

use itertools::{chain, Itertools};
use num_traits::{One, Zero};
use tracing::{span, Level};

use super::air::InvalidClaimError;
use super::gkr_lookups::accumulation::{MleClaimAccumulator, MleCollection};
use super::gkr_lookups::AccumulatedMleCoeffColumnOracle;
use crate::constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, PointEvaluator};
use crate::core::air::accumulation::PointEvaluationAccumulator;
use crate::core::backend::simd::blake2s::SIGMA;
use crate::core::backend::simd::column::{BaseColumn, SecureColumn};
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr_prover::Layer;
use crate::core::lookups::gkr_verifier::{LogUpArtifactInstance, LookupArtifactInstance};
use crate::core::lookups::mle::Mle;
use crate::core::pcs::TreeVec;
use crate::core::ColumnVec;
use crate::examples::blake::round::RoundElements;
use crate::examples::blake::scheduler::{eval_next_u32, BlakeElements, BlakeSchedulerLookupData};
use crate::examples::blake::{Fu32, N_ROUNDS, STATE_SIZE};

pub type BlakeSchedulerComponent = FrameworkComponent<BlakeSchedulerEval>;

pub struct BlakeSchedulerEval {
    pub log_size: u32,
    pub blake_lookup_elements: BlakeElements,
    pub round_lookup_elements: RoundElements,
}

impl FrameworkEval for BlakeSchedulerEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let _ = SchedulerEvals::new(&mut eval);
        eval
    }
}

impl AccumulatedMleCoeffColumnOracle for BlakeSchedulerComponent {
    fn accumulate_at_point(
        &self,
        _point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        acc: &mut PointEvaluationAccumulator,
    ) {
        // Create dummy point evaluator just to extract the value we need from the mask
        let mut _accumulator = PointEvaluationAccumulator::new(SecureField::one());
        let mut eval = PointEvaluator::new(
            mask.sub_tree(self.trace_locations()),
            &mut _accumulator,
            SecureField::one(),
        );

        let SchedulerEvals { messages, states } = SchedulerEvals::new(&mut eval);

        // Schedule.
        for i in 0..N_ROUNDS {
            let input_state = &states[i];
            let output_state = &states[i + 1];
            let round_messages = SIGMA[i].map(|j| messages[j as usize]);
            // Use triplet in round lookup.
            let lookup_values = &chain![
                input_state.iter().copied().flat_map(Fu32::to_felts),
                output_state.iter().copied().flat_map(Fu32::to_felts),
                round_messages.iter().copied().flat_map(Fu32::to_felts)
            ]
            .collect_vec();
            let denoms_mle_coeff_col_eval = self.round_lookup_elements.combine(lookup_values);
            acc.accumulate(denoms_mle_coeff_col_eval);
        }

        let input_state = &states[0];
        let output_state = &states[N_ROUNDS];
        let lookup_values = &chain![
            input_state.iter().copied().flat_map(Fu32::to_felts),
            output_state.iter().copied().flat_map(Fu32::to_felts),
            messages.iter().copied().flat_map(Fu32::to_felts)
        ]
        .collect_vec();
        let denoms_mle_coeff_col_eval = self.blake_lookup_elements.combine(lookup_values);
        acc.accumulate(denoms_mle_coeff_col_eval);
    }
}

struct SchedulerEvals<E: EvalAtRow> {
    messages: [Fu32<E::F>; STATE_SIZE],
    states: [[Fu32<E::F>; STATE_SIZE]; N_ROUNDS + 1],
}

impl<E: EvalAtRow> SchedulerEvals<E> {
    fn new(eval: &mut E) -> Self {
        Self {
            messages: array::from_fn(|_| eval_next_u32(eval)),
            states: array::from_fn(|_| array::from_fn(|_| eval_next_u32(eval))),
        }
    }
}

pub struct SchedulerLookupArtifact {
    scheduler: LogUpArtifactInstance,
    rounds: [LogUpArtifactInstance; N_ROUNDS],
}

impl SchedulerLookupArtifact {
    pub fn new_from_iter(mut iter: impl Iterator<Item = LookupArtifactInstance>) -> Self {
        let rounds = array::from_fn(|_| match iter.next() {
            Some(LookupArtifactInstance::LogUp(artifact)) => artifact,
            _ => panic!(),
        });

        let scheduler = match iter.next() {
            Some(LookupArtifactInstance::LogUp(artifact)) => artifact,
            _ => panic!(),
        };

        Self { scheduler, rounds }
    }

    pub fn verify_succinct_mle_claims(&self) -> Result<(), InvalidClaimError> {
        let Self { scheduler, rounds } = self;

        // TODO(andrew): Consider checking the n_variables is correct.
        // if !self.scheduler.input_numerators_claim.is_one() {
        if !scheduler.input_numerators_claim.is_zero() {
            return Err(InvalidClaimError);
        }

        for round in rounds {
            if !round.input_numerators_claim.is_one() {
                return Err(InvalidClaimError);
            }
        }

        Ok(())
    }

    pub fn accumulate_mle_eval_iop_claims(&self, acc: &mut MleClaimAccumulator) {
        let Self { scheduler, rounds } = self;

        for round in rounds {
            acc.accumulate(round.input_n_variables, round.input_denominators_claim);
        }

        // TODO: Note `n_variables` is not verified. Probably fine since if the prover gives wrong
        // info they'll be caught. Can panic though if the n_variables is too high. Consider
        // checking the number of GKR layers in the verifier is less than
        // LOG_CIRCLE_ORDER-LOG_BLOWUP-LOG_EXPAND.
        acc.accumulate(
            scheduler.input_n_variables,
            scheduler.input_denominators_claim,
        );
    }
}

pub fn generate_lookup_instances(
    log_size: u32,
    lookup_data: BlakeSchedulerLookupData,
    round_lookup_elements: &RoundElements,
    blake_lookup_elements: &BlakeElements,
    collection_for_univariate_iop: &mut MleCollection<SimdBackend>,
) -> Vec<Layer<SimdBackend>> {
    let _span = span!(Level::INFO, "Generate scheduler interaction trace").entered();
    let size = 1 << log_size;
    let mut round_lookup_layers = Vec::new();

    for l0 in &lookup_data.round_lookups {
        let mut denominators = Mle::<SimdBackend, SecureField>::new(SecureColumn::zeros(size));
        for vec_row in 0..1 << (log_size - LOG_N_LANES) {
            let denom = round_lookup_elements.combine(&l0.each_ref().map(|l| l.data[vec_row]));
            denominators.data[vec_row] = denom;
        }
        collection_for_univariate_iop.push(denominators.clone());
        round_lookup_layers.push(Layer::LogUpSingles { denominators })
    }

    // Blake hash lookup.
    let blake_numers = Mle::<SimdBackend, BaseField>::new(BaseColumn::zeros(size));
    let mut blake_denoms = Mle::<SimdBackend, SecureField>::new(SecureColumn::zeros(size));
    for vec_row in 0..1 << (log_size - LOG_N_LANES) {
        let blake_denom: PackedSecureField = blake_lookup_elements.combine(
            &lookup_data
                .blake_lookups
                .each_ref()
                .map(|l| l.data[vec_row]),
        );
        blake_denoms.data[vec_row] = blake_denom;
    }
    collection_for_univariate_iop.push(blake_denoms.clone());
    round_lookup_layers.push(Layer::LogUpMultiplicities {
        numerators: blake_numers,
        denominators: blake_denoms,
    });

    round_lookup_layers
}
