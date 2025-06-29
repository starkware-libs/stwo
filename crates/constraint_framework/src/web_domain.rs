use std::ops::Mul;

use stwo_prover::core::backend::simd::column::VeryPackedSecureColumnByCoords;
use stwo_prover::core::backend::simd::very_packed_m31::{
    VeryPackedBaseField, VeryPackedSecureField,
};
use stwo_prover::core::backend::web::WebBackend;
use stwo_prover::core::fields::m31::{BaseField, M31};
use stwo_prover::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo_prover::core::lookups::utils::Fraction;
use stwo_prover::core::pcs::TreeVec;
use stwo_prover::core::poly::circle::{CircleDomain, CircleEvaluation, CirclePoly};
use stwo_prover::core::poly::BitReversedOrder;

use super::logup::LogupAtRow;
use super::{EvalAtRow, INTERACTION_TRACE_IDX};

/// Dummy evaluator for WebGPU.
pub struct WebDomainEvaluator<'a> {
    pub trace_poly: &'a TreeVec<Vec<&'a CirclePoly<WebBackend>>>,
    pub trace_eval: &'a TreeVec<Vec<&'a CircleEvaluation<WebBackend, BaseField, BitReversedOrder>>>,
    pub needs_to_extend: bool,
    pub col: &'a mut VeryPackedSecureColumnByCoords,
    pub random_coeff_powers: Vec<SecureField>,
    pub eval_domain: CircleDomain,
    pub trace_domain_log_size: u32,
    pub denom_inv: Vec<M31>,
    pub claimed_sum: SecureField,
    pub log_size: u32,
    pub logup: LogupAtRow<Self>,
}

impl<'a> WebDomainEvaluator<'a> {
    pub fn new(
        trace_poly: &'a TreeVec<Vec<&'a CirclePoly<WebBackend>>>,
        trace_eval: &'a TreeVec<Vec<&'a CircleEvaluation<WebBackend, BaseField, BitReversedOrder>>>,
        needs_to_extend: bool,
        col: &'a mut VeryPackedSecureColumnByCoords,
        random_coeff_powers: Vec<SecureField>,
        eval_domain: CircleDomain,
        trace_domain_log_size: u32,
        denom_inv: Vec<M31>,
        log_size: u32,
        claimed_sum: SecureField,
    ) -> Self {
        Self {
            trace_poly,
            trace_eval,
            needs_to_extend,
            col,
            random_coeff_powers,
            eval_domain,
            trace_domain_log_size,
            denom_inv,
            claimed_sum,
            log_size,
            logup: LogupAtRow::new(INTERACTION_TRACE_IDX, claimed_sum, log_size),
        }
    }
}

/// Dummy implementation for WebGPU. These methods will be implemented as WGSL code, so they don't
/// need to be implemented here.
#[allow(unused_variables)]
impl EvalAtRow for WebDomainEvaluator<'_> {
    type F = VeryPackedBaseField;
    type EF = VeryPackedSecureField;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        unimplemented!()
    }
    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF> + From<G>,
    {
        unimplemented!()
    }

    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        unimplemented!()
    }

    super::logup_proxy!();
}
