use std::collections::{BTreeMap, BTreeSet};

use itertools::Itertools;
use num_traits::Zero;
use once_cell::sync::OnceCell;

use crate::core::air::accumulation::PointEvaluationAccumulator;
use crate::core::air::Component;
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::circle::{CirclePoint, M31_CIRCLE_LOG_ORDER};
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr_verifier::{
    partially_verify_batch, Gate, GkrArtifact, GkrBatchProof,
};
use crate::core::pcs::TreeVec;
use crate::core::prover::{BASE_TRACE, INTERACTION_TRACE};
use crate::core::{ColumnVec, InteractionElements, LookupValues};
use crate::examples::xor::mle_eval_component::verifier::MleEvalVerifierComponent;

// TODO(andrew): Docs.
pub const MAX_MULTILINEAR_N_VARIABLES: u32 = M31_CIRCLE_LOG_ORDER - 1;

pub trait GkrLookupComponent: Component {
    /// Returns the config for each lookup instance used by this component.
    fn lookup_config(&self) -> Vec<LookupInstanceConfig>;

    /// Returns the number of variables in all multilinear columns involved in the univariate IOP
    /// for multilinear eval at point.
    fn mle_n_variables_for_univariate_iop(&self) -> BTreeSet<u32>;

    /// Validates GKR lookup column claims that the verifier has succinct multilinear polynomial of.
    fn validate_succinct_mle_claims(
        &self,
        ood_point: &[SecureField],
        mle_claims_by_instance: &[Vec<SecureField>],
        interaction_elements: &InteractionElements,
    ) -> bool;

    /// Accumulates GKR lookup column claims that the verifier must verify by a univariate IOP for
    /// multilinear eval at point.
    fn accumulate_mle_claims_for_univariate_iop(
        &self,
        mle_claims_by_instance: &[Vec<SecureField>],
        accumulator: &mut MleClaimAccumulator,
    );

    /// TODO
    fn evaluate_lookup_columns_for_univariate_iop_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &ColumnVec<Vec<SecureField>>,
        accumulator: &mut UnivariateClaimAccumulator,
        interaction_elements: &InteractionElements,
    );
}

pub struct GkrLookupVerifierComponent<'a> {
    gkr_proof: GkrBatchProof,
    mle_eval_components: OnceCell<Vec<MleEvalVerifierComponent>>,
    sub_components: Vec<&'a dyn GkrLookupComponent>,
}

impl<'a> GkrLookupVerifierComponent<'a> {
    pub fn new(gkr_proof: GkrBatchProof, sub_components: Vec<&'a dyn GkrLookupComponent>) -> Self {
        Self {
            gkr_proof,
            sub_components,
            mle_eval_components: OnceCell::new(),
        }
    }

    pub fn verify_gkr_and_generate_mle_eval_components(
        &self,
        channel: &mut Blake2sChannel,
        elements: &InteractionElements,
    ) {
        let mut gates = Vec::new();

        for component in &self.sub_components {
            for lookup_config in component.lookup_config() {
                gates.push(lookup_config.variant);
            }
        }

        // TODO(andrew): need to figure out how to handle errors here.
        let GkrArtifact {
            ood_point,
            claims_to_verify_by_instance,
            n_variables_by_instance,
        } = partially_verify_batch(gates, &self.gkr_proof, channel).unwrap();

        let lookup_claims = &mut claims_to_verify_by_instance.into_iter();
        let gkr_combination_coeff = channel.draw_felt();
        let mut mle_claim_accumulator = MleClaimAccumulator::new(gkr_combination_coeff);

        for component in &self.sub_components {
            let n_lookup_instances = component.lookup_config().len();
            let claims = lookup_claims.take(n_lookup_instances).collect_vec();

            if !component.validate_succinct_mle_claims(&ood_point, &claims, elements) {
                // TODO(andrew): Throw error.
                todo!()
            }

            component.accumulate_mle_claims_for_univariate_iop(&claims, &mut mle_claim_accumulator)
        }

        let all_multilinear_n_variables = BTreeSet::from_iter(n_variables_by_instance);
        let mut mle_eval_components = Vec::new();

        for (n_variables, claim) in mle_claim_accumulator.into_accumulations() {
            assert!(all_multilinear_n_variables.contains(&(n_variables as usize)));
            let eval_point = &ood_point[ood_point.len() - n_variables as usize..];
            let mle_eval_component = MleEvalVerifierComponent::new(eval_point, claim);
            mle_eval_components.push(mle_eval_component);
        }

        self.mle_eval_components.set(mle_eval_components).unwrap();
    }

    /// Returns (1) all user specified sub components and (2) components for GKR verification.
    ///
    /// Components for GKR verification are specifically the components that perform a univariate
    /// IOP for multilinear eval at point <https://eprint.iacr.org/2023/1284.pdf> (section 5.2).
    pub fn all_components(&self) -> impl Iterator<Item = &'_ dyn Component> {
        let sub_components = self
            .sub_components
            .iter()
            .copied()
            .map(|c| c as &dyn Component);

        let mle_eval_components = self
            .mle_eval_components
            .get()
            .unwrap()
            .iter()
            .map(|c| c as &dyn Component);

        sub_components.chain(mle_eval_components)
    }
}

impl<'a> Component for GkrLookupVerifierComponent<'a> {
    // TODO: Bad this is code duplication of GkrLookupProverComponent.
    fn n_constraints(&self) -> usize {
        self.all_components().fold(0, |n_constraints, component| {
            n_constraints + component.n_constraints()
        })
    }

    // TODO: Bad this is code duplication of GkrLookupProverComponent.
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.all_components()
            .map(|c| c.max_constraint_log_degree_bound())
            .max()
            .unwrap()
    }

    // TODO: Bad this is code duplication of GkrLookupProverComponent.
    fn n_interaction_phases(&self) -> u32 {
        2
    }

    // TODO: Bad this is code duplication of GkrLookupProverComponent.
    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        // TODO(andrew): Currently don't facilitate sub components writing interaction trace so
        // any interaction trace degree bound would cause an error. Consider how to allow
        // subcomponents to write an interaction trace. Do we even need this?
        let mut base_trace_log_bounds = Vec::new();
        let mut interaction_trace_log_bounds = Vec::new();

        for component in self.all_components() {
            let log_degree_bounds = component.trace_log_degree_bounds();
            base_trace_log_bounds.extend(log_degree_bounds[BASE_TRACE].clone());
            interaction_trace_log_bounds.extend(log_degree_bounds[INTERACTION_TRACE].clone());
        }

        TreeVec::new(vec![base_trace_log_bounds, interaction_trace_log_bounds])
    }

    // TODO: Bad this is code duplication of GkrLookupProverComponent.
    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let mut base_trace_mask_points = Vec::new();
        let mut interaction_trace_mask_points = Vec::new();

        for component in self.all_components() {
            let mask_points = component.mask_points(point);
            base_trace_mask_points.extend(mask_points[BASE_TRACE].clone());
            interaction_trace_mask_points.extend(mask_points[INTERACTION_TRACE].clone());
        }

        TreeVec::new(vec![base_trace_mask_points, interaction_trace_mask_points])
    }

    // TODO: Bad this is code duplication of GkrLookupProverComponent.
    fn interaction_element_ids(&self) -> Vec<String> {
        self.all_components()
            .flat_map(|c| c.interaction_element_ids())
            .collect()
    }

    // TODO: Bad this is code duplication of GkrLookupProverComponent.
    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &ColumnVec<Vec<SecureField>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        interaction_elements: &InteractionElements,
        lookup_values: &LookupValues,
    ) {
        let trace_log_degree_bounds = self.trace_log_degree_bounds();
        let n_base_trace_cols = trace_log_degree_bounds[BASE_TRACE].len();

        let (base_trace_mask, interaction_trace_mask) = mask.split_at(n_base_trace_cols);

        let base_trace_mask = &mut base_trace_mask.iter().cloned();
        let interaction_trace_mask = &mut interaction_trace_mask.iter().cloned();

        for component in self.all_components() {
            // TODO: Check if mask points always returns same number of values as
            // trace_log_degree_bounds.
            let mask_points = self.mask_points(point);
            let n_base_trace_mask_cols = mask_points[BASE_TRACE].len();
            let n_interaction_trace_mask_cols = mask_points[INTERACTION_TRACE].len();

            let mask = base_trace_mask
                .take(n_base_trace_mask_cols)
                .chain(interaction_trace_mask.take(n_interaction_trace_mask_cols))
                .collect();

            component.evaluate_constraint_quotients_at_point(
                point,
                &mask,
                evaluation_accumulator,
                interaction_elements,
                lookup_values,
            )
        }
    }
}

pub struct LookupInstanceConfig {
    pub variant: Gate,
    pub is_table: bool,
    pub table_id: String,
}

/// Accumulates claims of multilinear polynomials with the same number of variables.
pub struct MleClaimAccumulator {
    acc_coeff: SecureField,
    acc_by_n_variables: Vec<Option<SecureField>>,
}

impl MleClaimAccumulator {
    pub fn new(acc_coeff: SecureField) -> Self {
        Self {
            acc_coeff,
            acc_by_n_variables: vec![None; M31_CIRCLE_LOG_ORDER as usize + 1],
        }
    }

    pub fn accumulate(&mut self, log_size: u32, evaluation: SecureField) {
        let acc = self.acc_by_n_variables[log_size as usize].get_or_insert_with(SecureField::zero);
        *acc = *acc * self.acc_coeff + evaluation;
    }

    pub fn into_accumulations(self) -> BTreeMap<u32, SecureField> {
        let mut res = BTreeMap::new();

        for (n_variables, claim) in self.acc_by_n_variables.into_iter().enumerate() {
            if let Some(claim) = claim {
                res.insert(n_variables as u32, claim);
            }
        }

        res
    }
}

/// Accumulates claims of univariate polynomials with the same bounded degree.
// TODO(andrew): Identical to `MleClaimAccumulator`. Consider unifying.
pub struct UnivariateClaimAccumulator {
    acc_coeff: SecureField,
    acc_by_log_degree_bound: Vec<Option<SecureField>>,
}

impl UnivariateClaimAccumulator {
    pub fn new(acc_coeff: SecureField) -> Self {
        Self {
            acc_coeff,
            acc_by_log_degree_bound: vec![None; M31_CIRCLE_LOG_ORDER as usize + 1],
        }
    }

    pub fn accumulate(&mut self, log_size: u32, evaluation: SecureField) {
        let acc =
            self.acc_by_log_degree_bound[log_size as usize].get_or_insert_with(SecureField::zero);
        *acc = *acc * self.acc_coeff + evaluation;
    }

    pub fn into_accumulations(self) -> Vec<Option<SecureField>> {
        self.acc_by_log_degree_bound
    }
}
