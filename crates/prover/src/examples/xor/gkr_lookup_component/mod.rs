use std::collections::BTreeSet;

use accumulation::{MleAccumulator, MleClaimAccumulator, UnivariateClaimAccumulator};

use crate::core::air::{Component, ComponentProver};
use crate::core::backend::Backend;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr_prover::Layer;
use crate::core::lookups::gkr_verifier::Gate;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::{ColumnVec, InteractionElements};

pub mod accumulation;
pub mod mle_eval_component;
pub mod wrapper_component;

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

pub trait GkrLookupComponent: Component {
    /// Returns the config for each lookup instance used by this component.
    fn lookup_config(&self) -> Vec<LookupInstanceConfig>;

    /// Returns the number of variables in all multilinear columns involved in the univariate IOP
    /// for multilinear eval at point.
    fn mle_n_variables_for_univariate_iop(&self) -> BTreeSet<u32>;

    /// Validates GKR lookup column claims that the verifier has succinct multilinear polynomial of.
    // TODO: This way of passing claims and eval points is kind of janky
    fn validate_succinct_mle_claims(
        &self,
        eval_point_by_instance: &[&[SecureField]],
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

pub struct LookupInstanceConfig {
    pub variant: Gate,
    pub is_table: bool,
    pub table_id: String,
}
