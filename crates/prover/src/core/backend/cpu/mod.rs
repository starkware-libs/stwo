mod accumulation;
mod blake2s;
mod circle;
mod fri;
pub mod lookups;
pub mod quotients;

use std::fmt::Debug;
use std::iter::zip;

use num_traits::{One, Zero};

use super::{Backend, Col, Column, ColumnOps, FieldOps, MultilinearEvalAtPointIopOps};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;
use crate::core::lookups::gkr_prover::GkrOps;
use crate::core::lookups::mle::Mle;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::bit_reverse;
use crate::core::ColumnVec;
use crate::examples::xor::multilinear_eval_at_point::BatchMultilinearEvalIopProver;

#[derive(Copy, Clone, Debug)]
pub struct CpuBackend;

impl Backend for CpuBackend {}

// TODO: Remove.
impl MultilinearEvalAtPointIopOps for CpuBackend {
    fn random_linear_combination(
        columns: Vec<Vec<SecureField>>,
        random_coeff: SecureField,
    ) -> Col<Self, SecureField> {
        let mut columns = columns.into_iter().rev();

        let mut acc = columns.next().unwrap();

        for col in columns {
            zip(&mut acc, col).for_each(|(acc_v, col_v)| *acc_v *= random_coeff + col_v);
        }

        acc
    }

    fn write_interaction_trace(
        prover: &BatchMultilinearEvalIopProver<Self>,
    ) -> ColumnVec<CircleEvaluation<Self, BaseField, BitReversedOrder>> {
        let mut interaction_trace_columns = Vec::new();

        for (&n_variables, mle) in &prover.poly_by_n_variables {
            let point_len = prover.multilinear_eval_point.len();
            let y = &prover.multilinear_eval_point[point_len - n_variables as usize..];
            let eq_evals = Self::gen_eq_evals(y, SecureField::one());

            // Split into base field columns
            let mut eq_evals_col0 = Vec::new();
            let mut eq_evals_col1 = Vec::new();
            let mut eq_evals_col2 = Vec::new();
            let mut eq_evals_col3 = Vec::new();

            for eq_eval in eq_evals.into_evals() {
                let [e0, e1, e2, e3] = eq_eval.to_m31_array();
                eq_evals_col0.push(e0);
                eq_evals_col1.push(e1);
                eq_evals_col2.push(e2);
                eq_evals_col3.push(e3);
            }

            // Univariate sumcheck g columns.
            let mut g_col0 = Vec::new();
            let mut g_col1 = Vec::new();
            let mut g_col2 = Vec::new();
            let mut g_col3 = Vec::new();

            for g_eval in &**mle {
                let [e0, e1, e2, e3] = g_eval.to_m31_array();
                g_col0.push(e0);
                g_col1.push(e1);
                g_col2.push(e2);
                g_col3.push(e3);
            }

            // Univariate sumcheck h columns.
            // TODO: Fix later.
            // TODO: Let comopnets return the evaluations and polynomials. This is because g needs
            // to be interpolated to compute h and prevents the interpolation happening again in
            // another part of the protocol.
            let h_col0 = vec![BaseField::zero(); 1 << n_variables];
            let h_col1 = vec![BaseField::zero(); 1 << n_variables];
            let h_col2 = vec![BaseField::zero(); 1 << n_variables];
            let h_col3 = vec![BaseField::zero(); 1 << n_variables];

            let domain = CanonicCoset::new(n_variables).circle_domain();

            interaction_trace_columns.push(CircleEvaluation::new(domain, eq_evals_col0));
            interaction_trace_columns.push(CircleEvaluation::new(domain, eq_evals_col1));
            interaction_trace_columns.push(CircleEvaluation::new(domain, eq_evals_col2));
            interaction_trace_columns.push(CircleEvaluation::new(domain, eq_evals_col3));

            interaction_trace_columns.push(CircleEvaluation::new(domain, g_col0));
            interaction_trace_columns.push(CircleEvaluation::new(domain, g_col1));
            interaction_trace_columns.push(CircleEvaluation::new(domain, g_col2));
            interaction_trace_columns.push(CircleEvaluation::new(domain, g_col3));

            interaction_trace_columns.push(CircleEvaluation::new(domain, h_col0));
            interaction_trace_columns.push(CircleEvaluation::new(domain, h_col1));
            interaction_trace_columns.push(CircleEvaluation::new(domain, h_col2));
            interaction_trace_columns.push(CircleEvaluation::new(domain, h_col3));
        }

        interaction_trace_columns
    }
}

impl<T: Debug + Clone + Default> ColumnOps<T> for CpuBackend {
    type Column = Vec<T>;

    fn bit_reverse_column(column: &mut Self::Column) {
        bit_reverse(column)
    }
}

impl<F: Field> FieldOps<F> for CpuBackend {
    /// Batch inversion using the Montgomery's trick.
    // TODO(Ohad): Benchmark this function.
    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
        F::batch_inverse(column, &mut dst[..]);
    }
}

impl<T: Debug + Clone + Default> Column<T> for Vec<T> {
    fn zeros(len: usize) -> Self {
        vec![T::default(); len]
    }
    fn to_cpu(&self) -> Vec<T> {
        self.clone()
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn at(&self, index: usize) -> T {
        self[index].clone()
    }
}

pub type CpuCirclePoly = CirclePoly<CpuBackend>;
pub type CpuCircleEvaluation<F, EvalOrder> = CircleEvaluation<CpuBackend, F, EvalOrder>;
pub type CpuMle<F> = Mle<CpuBackend, F>;

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::prelude::*;
    use rand::rngs::SmallRng;

    use crate::core::backend::{Column, CpuBackend, FieldOps};
    use crate::core::fields::qm31::QM31;
    use crate::core::fields::FieldExpOps;

    #[test]
    fn batch_inverse_test() {
        let mut rng = SmallRng::seed_from_u64(0);
        let column = rng.gen::<[QM31; 16]>().to_vec();
        let expected = column.iter().map(|e| e.inverse()).collect_vec();
        let mut dst = Column::zeros(column.len());

        CpuBackend::batch_inverse(&column, &mut dst);

        assert_eq!(expected, dst);
    }
}
