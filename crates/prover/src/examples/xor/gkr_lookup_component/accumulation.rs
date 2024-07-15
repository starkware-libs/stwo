use std::collections::BTreeMap;

use num_traits::Zero;

use crate::core::backend::Backend;
use crate::core::circle::M31_CIRCLE_LOG_ORDER;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::mle::Mle;

/// Max number of variables for multilinear polynomials that get compiled into a univariate IOP.
pub const MAX_MULTILINEAR_N_VARIABLES: u32 = M31_CIRCLE_LOG_ORDER - 1;

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
