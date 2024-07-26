use std::ops::{Add, Mul};

use stwo_prover::examples::poseidon::N_STATE;

#[derive(Debug, Clone, Copy)]
pub struct CostCount {
    pub base_mul_base: usize,
    pub base_add_base: usize,
    pub ext_add_base: usize,
    pub ext_add_ext: usize,
    pub ext_mul_ext: usize,
    pub ext_mul_base: usize,
    pub hash_compression: usize,
}

impl CostCount {
    pub fn zero() -> Self {
        Self {
            base_mul_base: 0,
            base_add_base: 0,
            ext_add_base: 0,
            ext_add_ext: 0,
            ext_mul_ext: 0,
            ext_mul_base: 0,
            hash_compression: 0,
        }
    }

    pub fn fft_butterfly() -> Self {
        let mut res = Self::zero();
        res.base_add_base += 2;
        res.base_mul_base += 1;
        res
    }

    pub fn fft(log_size: u32) -> Self {
        let n_butterflies = (log_size as usize - 1) << (log_size - 1);
        let butterfly_cost = Self::fft_butterfly();
        butterfly_cost * n_butterflies
    }

    pub fn base_mul_base(n: usize) -> Self {
        let mut res = Self::zero();
        res.base_mul_base += n;
        res
    }

    pub fn _ext_mul_base(n: usize) -> Self {
        let mut res = Self::zero();
        res.ext_mul_base += n;
        res
    }

    pub fn base_eval_ext(n_coeffs: usize) -> Self {
        let mut res = Self::zero();
        res.ext_mul_base += n_coeffs;
        res.ext_add_ext += n_coeffs;
        res
    }

    pub fn ext_eval_ext(n_coeffs: usize) -> Self {
        let mut res = Self::zero();
        res.ext_mul_base += n_coeffs;
        res.ext_add_ext += n_coeffs;
        res
    }

    pub fn _base_quotient_ext(_n: usize) -> Self {
        todo!()
    }

    pub fn ext_quotient_ext(n: usize) -> Self {
        let denom_inv = Self::batch_inverse_ext_values(n);
        let numer_mul_denom_inv = Self::ext_mul_ext(n);
        denom_inv + numer_mul_denom_inv
    }

    pub fn ext_mul_ext(n: usize) -> Self {
        let mut res = Self::zero();
        res.ext_mul_ext += n;
        res
    }

    pub fn batch_inverse_base_values(n_values: usize) -> Self {
        let mut res = Self::zero();
        // 2 mults per item.
        res.base_mul_base += 2 * n_values;
        res
    }

    pub fn batch_inverse_ext_values(n_values: usize) -> Self {
        let mut res = Self::zero();
        // 2 mults per item.
        res.ext_mul_ext += 2 * n_values;
        res
    }

    pub fn commitment(log_degree_bound: u32, log_blowup: u32, n_base_columns: usize) -> Self {
        let interpolation_cost = Self::fft(log_degree_bound) * n_base_columns;
        let log_col_n_lde_evals = log_degree_bound + log_blowup;
        let lde_evaluation_cost = Self::fft(log_col_n_lde_evals) * n_base_columns;
        let merkle_tree_cost = Self::merkle(log_col_n_lde_evals, n_base_columns);
        interpolation_cost + lde_evaluation_cost + merkle_tree_cost
    }

    pub fn merkle(col_log_size: u32, n_base_columns: usize) -> Self {
        let col_size = 1 << col_log_size;
        let mut res = CostCount::zero();
        // 8 M31 field elements per 256 byte hash.
        let n_leaf_compressions = n_base_columns * col_size / 8;
        // All columns compressed into single leaf hash.
        let n_leaf = col_size;
        let n_nodes = n_leaf - 1;
        res.hash_compression += n_nodes + n_leaf_compressions;
        res
    }

    pub fn _random_linear_combination_ext_values(n_values: usize) -> Self {
        let mut res = Self::zero();
        res.ext_add_ext += n_values;
        res.ext_mul_ext += n_values;
        res
    }

    pub fn random_linear_combination_base_values(n_values: usize) -> Self {
        let mut res = Self::zero();
        res.ext_add_ext += n_values;
        res.ext_mul_base += n_values;
        res
    }

    pub fn gen_eq_evals(n_variables: u32) -> Self {
        let mut res = Self::zero();
        res.ext_mul_ext += (1 << n_variables) / 2;
        res.ext_add_ext += (1 << n_variables) / 2;
        res
    }

    pub fn poseidon_external_round_constants() -> Self {
        let mut res = Self::zero();
        res.base_add_base += N_STATE;
        res
    }

    pub fn poseidon_internal_round_constants() -> Self {
        let mut res = Self::zero();
        res.base_add_base += 1;
        res
    }

    pub fn poseidon_gkr_partial_round(log_n_poseidon_instances: u32) -> Self {
        let n_instances = 1 << log_n_poseidon_instances;
        let n_variables = log_n_poseidon_instances;
        let eq_evals_cost = Self::gen_eq_evals(n_variables.saturating_sub(1));

        // First sumcheck round.
        let n_terms = n_instances / 2;
        let mut single_point_sumcheck_first_round_cost = Self::poseidon_partial_round() * n_terms;
        // eq((r,0,x),y) * p((r,0,x))
        single_point_sumcheck_first_round_cost.ext_mul_base += n_terms;
        // Need to evaluate at 4 points (0, 2, 3, 4)
        let first_round_sumcheck_cost = single_point_sumcheck_first_round_cost * 4;

        // TODO: Remaining round.

        eq_evals_cost + first_round_sumcheck_cost
    }

    pub fn poseidon_gkr_full_round(log_n_poseidon_instances: u32) -> Self {
        let n_instances = 1 << log_n_poseidon_instances;
        let n_variables = log_n_poseidon_instances;
        let eq_evals_cost = Self::gen_eq_evals(n_variables.saturating_sub(1));

        // First sumcheck round.
        let n_terms = n_instances / 2;
        let mut single_point_sumcheck_first_round_cost = Self::poseidon_full_round() * n_terms;
        // eq((r,0,x),y) * p((r,0,x))
        single_point_sumcheck_first_round_cost.ext_mul_base += n_terms;
        // Need to evaluate at 4 points (0, 2, 3, 4)
        let first_round_sumcheck_cost = single_point_sumcheck_first_round_cost * 4;

        // TODO: Remaining round.

        eq_evals_cost + first_round_sumcheck_cost
    }

    pub fn poseidon_partial_round() -> Self {
        let round_const_cost = Self::poseidon_internal_round_constants();
        let int_matrix_cost = Self::poseidon_apply_internal_round_matrix();
        round_const_cost + int_matrix_cost + Self::base_pow5()
    }

    pub fn poseidon_full_round() -> Self {
        let round_const_cost = CostCount::poseidon_external_round_constants();
        let ext_matrix_cost = CostCount::poseidon_apply_external_round_matrix();
        round_const_cost + ext_matrix_cost + Self::base_pow5() * N_STATE
    }

    pub fn base_pow5() -> Self {
        let mut res = Self::zero();
        res.base_mul_base = 3;
        res
    }

    pub fn poseidon_apply_external_round_matrix() -> Self {
        let mut res = Self::poseidon_apply_m4() * 4;
        res.base_add_base += (3 + 4) * 4;
        res
    }

    pub fn poseidon_apply_internal_round_matrix() -> Self {
        let mut res = Self::zero();
        res.base_add_base += N_STATE - 1;
        res.base_mul_base += N_STATE;
        res.base_add_base += N_STATE;
        res
    }

    pub fn poseidon_apply_m4() -> Self {
        let mut res = Self::zero();
        res.base_add_base += 14;
        res
    }
}

impl Mul<usize> for CostCount {
    type Output = Self;

    fn mul(self, rhs: usize) -> Self::Output {
        let Self {
            base_mul_base,
            base_add_base,
            ext_add_base,
            ext_add_ext,
            ext_mul_ext,
            ext_mul_base,
            hash_compression,
        } = self;
        Self {
            base_mul_base: base_mul_base * rhs,
            base_add_base: base_add_base * rhs,
            ext_add_base: ext_add_base * rhs,
            ext_add_ext: ext_add_ext * rhs,
            ext_mul_ext: ext_mul_ext * rhs,
            ext_mul_base: ext_mul_base * rhs,
            hash_compression: hash_compression * rhs,
        }
    }
}

impl Add for CostCount {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            base_mul_base: self.base_mul_base + rhs.base_mul_base,
            base_add_base: self.base_add_base + rhs.base_add_base,
            ext_add_base: self.ext_add_base + rhs.ext_add_base,
            ext_add_ext: self.ext_add_ext + rhs.ext_add_ext,
            ext_mul_ext: self.ext_mul_ext + rhs.ext_mul_ext,
            ext_mul_base: self.ext_mul_base + rhs.ext_mul_base,
            hash_compression: self.hash_compression + rhs.hash_compression,
        }
    }
}
