use std::collections::BTreeMap;
use std::iter::zip;

use itertools::{zip_eq, Itertools};
use once_cell::sync::OnceCell;

use super::accumulation::{MleAccumulator, MleClaimAccumulator};
use super::mle_eval_component::MleEvalComponent;
use super::GkrLookupComponentProver;
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentProver, ComponentTrace};
use crate::core::backend::{Backend, CpuBackend};
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr_prover::prove_batch;
use crate::core::lookups::gkr_verifier::{partially_verify_batch, GkrArtifact, GkrBatchProof};
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::prover::{BASE_TRACE, INTERACTION_TRACE};
use crate::core::{ColumnVec, InteractionElements, LookupValues};

// TODO: This may be better intended and implemented as a sub air rather than a component.

pub struct GkrLookupWrapperComponent<'a, B: Backend> {
    gkr_combination_coeff: OnceCell<SecureField>,
    gkr_proof: OnceCell<GkrBatchProof>,
    mle_eval_components: Vec<MleEvalComponent<B>>,
    sub_components: Vec<&'a dyn GkrLookupComponentProver<B>>,
}

impl<'a> GkrLookupWrapperComponent<'a, CpuBackend> {
    /// # Panics
    ///
    /// Panics if no components are passed.
    pub fn new_verifier(
        gkr_proof: GkrBatchProof,
        sub_components: Vec<&'a dyn GkrLookupComponentProver<CpuBackend>>,
    ) -> Self {
        assert!(!sub_components.is_empty());
        Self {
            gkr_combination_coeff: OnceCell::new(),
            gkr_proof: OnceCell::with_value(gkr_proof),
            mle_eval_components: gen_mle_eval_components(&sub_components),
            sub_components,
        }
    }

    /// # Panics
    ///
    /// Panics if no components are passed.
    pub fn new_prover(sub_components: Vec<&'a dyn GkrLookupComponentProver<CpuBackend>>) -> Self {
        assert!(!sub_components.is_empty());
        Self {
            gkr_combination_coeff: OnceCell::new(),
            gkr_proof: OnceCell::new(),
            mle_eval_components: gen_mle_eval_components(&sub_components),
            sub_components,
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
        } = partially_verify_batch(gates, self.gkr_proof.get().unwrap(), channel).unwrap();

        let lookup_claims = &mut claims_to_verify_by_instance.into_iter();
        let lookup_n_variables = &mut n_variables_by_instance.into_iter();
        let gkr_combination_coeff = channel.draw_felt();
        self.gkr_combination_coeff
            .set(gkr_combination_coeff)
            .unwrap();
        let mut mle_claim_accumulator = MleClaimAccumulator::new(gkr_combination_coeff);

        for component in &self.sub_components {
            let n_lookup_instances = component.lookup_config().len();
            let claims = lookup_claims.take(n_lookup_instances).collect_vec();
            let eval_points = lookup_n_variables
                .take(n_lookup_instances)
                .map(|n_vars| &ood_point[ood_point.len() - n_vars..])
                .collect_vec();
            // TODO: Error not panic.
            assert!(component.validate_succinct_mle_claims(&eval_points, &claims, elements));
            component.accumulate_mle_claims_for_univariate_iop(&claims, &mut mle_claim_accumulator)
        }

        let mle_claim_accumulations = mle_claim_accumulator.into_accumulations();

        for mle_eval_component in &self.mle_eval_components {
            let n_variables = mle_eval_component.n_variables;
            let claim = mle_claim_accumulations[&n_variables];
            mle_eval_component.claim.set(claim).unwrap();
            let eval_point = ood_point[ood_point.len() - n_variables as usize..].to_vec();
            mle_eval_component.eval_point.set(eval_point).unwrap();
        }
    }

    fn prove_gkr_and_generate_mle_eval_components(
        &self,
        channel: &mut Blake2sChannel,
        trace: &ColumnVec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
        elements: &InteractionElements,
    ) {
        let trace_columns_iter = &mut trace.iter().copied();

        let mut lookup_instances = Vec::new();

        for component in &self.sub_components {
            let n_trace_columns = component.trace_log_degree_bounds()[BASE_TRACE].len();
            let component_trace = trace_columns_iter.take(n_trace_columns).collect();
            lookup_instances.extend(component.write_lookup_instances(component_trace, elements));
        }

        let (gkr_proof, gkr_artifact) = prove_batch(channel, lookup_instances.clone());
        self.gkr_proof.set(gkr_proof).unwrap();

        let lookup_instances = &mut lookup_instances.into_iter();
        let lookup_claims = &mut gkr_artifact.claims_to_verify_by_instance.into_iter();
        let gkr_combination_coeff = channel.draw_felt();
        self.gkr_combination_coeff
            .set(gkr_combination_coeff)
            .unwrap();
        let mut mle_claim_accumulator = MleClaimAccumulator::new(gkr_combination_coeff);
        let mut mle_accumulator = MleAccumulator::new(gkr_combination_coeff);

        for component in &self.sub_components {
            let n_lookup_instances = component.lookup_config().len();
            let claims = lookup_claims.take(n_lookup_instances).collect_vec();
            component.accumulate_mle_claims_for_univariate_iop(&claims, &mut mle_claim_accumulator);
            let instances = lookup_instances.take(n_lookup_instances).collect();
            component.accumulate_mle_for_univariate_iop(instances, &mut mle_accumulator);
        }

        let mle_claim_accumulations = mle_claim_accumulator.into_accumulations();
        println!("yoo: {:?}", mle_claim_accumulations);
        let mle_accumulations = mle_accumulator.into_accumulations();
        let ood_point = gkr_artifact.ood_point;

        for (mle_eval_component, mle) in zip_eq(&self.mle_eval_components, mle_accumulations) {
            assert_eq!(mle_eval_component.n_variables as usize, mle.n_variables());
            let mle_eval_point = ood_point[ood_point.len() - mle.n_variables()..].to_vec();
            let mle_claim = mle_claim_accumulations[&(mle.n_variables() as u32)];

            // Sanity check.
            #[cfg(test)]
            assert_eq!(mle.eval_at_point(&mle_eval_point), mle_claim);

            mle_eval_component.eval_point.set(mle_eval_point).unwrap();
            mle_eval_component.mle.set(mle).unwrap();
            mle_eval_component.claim.set(mle_claim).unwrap();
        }
    }

    pub fn write_interaction_trace(
        &self,
        channel: &mut Blake2sChannel,
        trace: &ColumnVec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
        elements: &InteractionElements,
    ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        self.prove_gkr_and_generate_mle_eval_components(channel, trace, elements);

        // TODO(andrew): Currently this component doesn't support sub components writing
        // their own interaction trace. Do we even need/want this?
        let mut interaction_trace = Vec::new();

        for mle_eval_component in &self.mle_eval_components {
            let component_interaction_trace = mle_eval_component.write_interaction_trace();
            interaction_trace.extend(component_interaction_trace);
        }

        interaction_trace
    }

    /// Returns (1) all user specified sub components and (2) components for GKR verification.
    ///
    /// Components for GKR verification are specifically the components that perform a univariate
    /// IOP for multilinear eval at point <https://eprint.iacr.org/2023/1284.pdf> (section 5.2).
    pub fn all_components(&self) -> impl Iterator<Item = &'_ dyn Component> {
        let sub_components = self.sub_components.iter().map(|c| *c as &dyn Component);
        let mle_eval_components = self.mle_eval_components.iter().map(|c| c as &dyn Component);
        sub_components.chain(mle_eval_components)
    }

    pub fn try_into_gkr_proof(self) -> Result<GkrBatchProof, GkrProofNotConstructedError> {
        self.gkr_proof
            .into_inner()
            .ok_or(GkrProofNotConstructedError)
    }
}

#[derive(Debug)]
pub struct GkrProofNotConstructedError;

// TODO(andrew): Huge amount of code duplication in the GKR lookup verifier.
impl Component for GkrLookupWrapperComponent<'_, CpuBackend> {
    fn n_constraints(&self) -> usize {
        self.all_components().map(Component::n_constraints).sum()
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.all_components()
            .map(Component::max_constraint_log_degree_bound)
            .max()
            .unwrap()
    }

    fn n_interaction_phases(&self) -> u32 {
        2
    }

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

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        interaction_elements: &InteractionElements,
        lookup_values: &LookupValues,
    ) {
        let mut mask_iter = mask.as_ref().map(|trace| trace.iter().cloned()).0;

        for component in self.all_components() {
            let component_mask_points = &*component.mask_points(point);
            let mut component_mask = TreeVec::new(vec![]);

            for (trace_mask_points, trace_mask_iter) in zip(component_mask_points, &mut mask_iter) {
                let n_trace_cols = trace_mask_points.len();
                component_mask.push(trace_mask_iter.take(n_trace_cols).collect_vec());
            }

            component.evaluate_constraint_quotients_at_point(
                point,
                &component_mask,
                evaluation_accumulator,
                interaction_elements,
                lookup_values,
            )
        }
    }
}

impl<'a> ComponentProver<CpuBackend> for GkrLookupWrapperComponent<'a, CpuBackend> {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        _trace: &ComponentTrace<'_, CpuBackend>,
        _evaluation_accumulator: &mut DomainEvaluationAccumulator<CpuBackend>,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        todo!()
    }

    fn lookup_values(&self, _trace: &ComponentTrace<'_, CpuBackend>) -> LookupValues {
        LookupValues::default()
    }
}

fn gen_mle_eval_components<B: Backend>(
    components: &[&dyn GkrLookupComponentProver<B>],
) -> Vec<MleEvalComponent<B>> {
    let mut mle_eval_components = BTreeMap::new();

    for component in components.iter() {
        for n_variables in component.mle_n_variables_for_univariate_iop() {
            mle_eval_components
                .entry(n_variables)
                .or_insert_with(|| MleEvalComponent::new(n_variables));
        }
    }

    mle_eval_components.into_values().collect()
}
