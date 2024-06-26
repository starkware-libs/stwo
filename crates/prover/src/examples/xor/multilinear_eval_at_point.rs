use std::collections::BTreeMap;

use crate::core::channel::Channel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::utils::horner_eval;

pub struct BatchMultilinearEvalIopVerfier {
    eval_claims_by_n_variables: BTreeMap<u32, Vec<SecureField>>,
    aggregation_coeff: SecureField,
}

impl BatchMultilinearEvalIopVerfier {
    pub fn new(
        channel: &mut impl Channel,
        eval_claims_by_n_variables: BTreeMap<u32, Vec<SecureField>>,
    ) -> Self {
        Self {
            eval_claims_by_n_variables,
            aggregation_coeff: channel.draw_felt(),
        }
    }

    fn univariate_sumcheck_constant_coeff_claim_by_log_size(&self) -> BTreeMap<u32, SecureField> {
        self.eval_claims_by_n_variables
            .iter()
            .map(|(log_size, eval_claims)| {
                let n_claims = BaseField::from(eval_claims.len());
                let constant_coeff_claim =
                    horner_eval(eval_claims, self.aggregation_coeff) / n_claims;
                (log_size, constant_coeff_claim)
            })
            .collect()
    }
}
