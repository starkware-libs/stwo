use num_traits::Zero;
use once_cell::sync::OnceCell;

use super::verifier::{GkrLookupComponent, MAX_MULTILINEAR_N_VARIABLES};
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentProver, ComponentTrace};
use crate::core::backend::{Backend, CpuBackend};
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr_prover::{prove_batch, Layer};
use crate::core::lookups::gkr_verifier::GkrBatchProof;
use crate::core::lookups::mle::Mle;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::prover::{BASE_TRACE, INTERACTION_TRACE};
use crate::core::{ColumnVec, InteractionElements, LookupValues};
use crate::examples::xor::mle_eval_component::prover::MleEvalProverComponent;

pub trait GkrLookupComponentProver<B: Backend>: ComponentProver<B> + GkrLookupComponent {
    fn accumulate_mle_for_univariate_iop(
        &self,
        lookup_instances: Vec<Layer<B>>,
        mle_accumulator: &mut MleAccumulator<B>,
    );

    fn write_lookup_instances(
        &self,
        trace: ColumnVec<&CircleEvaluation<B, BaseField, BitReversedOrder>>,
        interaction_elements: &InteractionElements,
    ) -> Vec<Layer<B>>;
}

// TODO: This may be better intended and implemented as a sub air rather than a component.
pub struct GkrLookupProverComponent<'a, B: Backend> {
    gkr_proof: OnceCell<GkrBatchProof>,
    mle_eval_components: OnceCell<Vec<MleEvalProverComponent<B>>>,
    sub_components: Vec<&'a dyn GkrLookupComponentProver<B>>,
}

impl<'a> GkrLookupProverComponent<'a, CpuBackend> {
    /// # Panics
    ///
    /// Panics if no components are passed.
    pub fn new(components: Vec<&'a dyn GkrLookupComponentProver<CpuBackend>>) -> Self {
        assert!(!components.is_empty());

        Self {
            gkr_proof: OnceCell::new(),
            mle_eval_components: OnceCell::new(),
            sub_components: components,
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

        let (gkr_proof, _gkr_artifact) = prove_batch(channel, lookup_instances.clone());
        self.gkr_proof.set(gkr_proof).unwrap();

        let lookup_instances = &mut lookup_instances.into_iter();
        let mut mle_accumulator = MleAccumulator::new(channel.draw_felt());

        for component in &self.sub_components {
            let n_lookup_instances = component.lookup_config().len();
            let instances = lookup_instances.take(n_lookup_instances).collect();
            component.accumulate_mle_for_univariate_iop(instances, &mut mle_accumulator);
        }

        let mle_eval_components = mle_accumulator
            .into_accumulations()
            .into_iter()
            .map(MleEvalProverComponent::new)
            .collect();

        self.mle_eval_components.set(mle_eval_components).unwrap();
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

        for mle_eval_component in self.mle_eval_components.get().unwrap() {
            let component_interaction_trace = mle_eval_component.write_interaction_trace();
            interaction_trace.extend(component_interaction_trace);
        }

        interaction_trace
    }

    pub fn try_into_gkr_proof(self) -> Result<GkrBatchProof, GkrProofNotConstructedError> {
        self.gkr_proof
            .into_inner()
            .ok_or(GkrProofNotConstructedError)
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

#[derive(Debug)]
pub struct GkrProofNotConstructedError;

// TODO(andrew): Huge amount of code duplication in the GKR lookup verifier.
impl Component for GkrLookupProverComponent<'_, CpuBackend> {
    fn n_constraints(&self) -> usize {
        self.all_components().fold(0, |n_constraints, component| {
            n_constraints + component.n_constraints()
        })
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.all_components()
            .map(|c| c.max_constraint_log_degree_bound())
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

    fn interaction_element_ids(&self) -> Vec<String> {
        self.all_components()
            .flat_map(|c| c.interaction_element_ids())
            .collect()
    }

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

impl<'a> ComponentProver<CpuBackend> for GkrLookupProverComponent<'a, CpuBackend> {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        _trace: &ComponentTrace<'_, CpuBackend>,
        _evaluation_accumulator: &mut DomainEvaluationAccumulator<CpuBackend>,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        todo!()
    }
}

/// Accumulates multilinear polynomials with the same number of variables.
pub struct MleAccumulator<B: Backend> {
    acc_coeff: SecureField,
    acc_by_n_variables: Vec<Option<Mle<B, SecureField>>>,
}

impl<B: Backend> MleAccumulator<B> {
    pub fn new(acc_coeff: SecureField) -> Self {
        Self {
            acc_coeff,
            acc_by_n_variables: vec![None; MAX_MULTILINEAR_N_VARIABLES as usize + 1],
        }
    }

    pub fn accumulation_coeff(&self) -> SecureField {
        self.acc_coeff
    }

    pub fn column(&mut self, n_variables: u32) -> &mut Mle<B, SecureField> {
        self.acc_by_n_variables[n_variables as usize].get_or_insert_with(|| {
            // TODO(andrew): Very inefficient.
            Mle::new((0..1 << n_variables).map(|_| SecureField::zero()).collect())
        })
    }

    /// Returns the accumulated [`Mle`]s in ascending order of their number of variables.
    pub fn into_accumulations(self) -> Vec<Mle<B, SecureField>> {
        self.acc_by_n_variables.into_iter().flatten().collect()
    }
}
