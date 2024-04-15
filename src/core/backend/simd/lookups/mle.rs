use core::slice::SlicePattern;
use std::array;
use std::iter::zip;

use itertools::Itertools;
use num_traits::{One, Zero};

use crate::core::backend::cpu::lookups::gkr::gen_eq_evals;
use crate::core::backend::cpu::lookups::mle::eval_mle_at_point;
use crate::core::backend::simd::column::SecureFieldVec;
use crate::core::backend::simd::m31::N_LANES;
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{CPUBackend, Column};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::mle::{Mle, MleOps};
use crate::core::lookups::sumcheck::SumcheckOracle;
use crate::core::lookups::utils::UnivariatePolynomial;

impl MleOps<BaseField> for SimdBackend {
    fn eval_at_point(mle: &Mle<Self, BaseField>, point: &[BaseField]) -> BaseField {
        // TODO: SIMD implementation.
        eval_mle_at_point(mle.as_slice(), point)
    }

    fn fix_first(mle: Mle<Self, BaseField>, assignment: SecureField) -> Mle<Self, SecureField> {
        let midpoint = mle.len() / 2;

        // Offload small values to CPU implementation.
        if midpoint < N_LANES {
            let cpu_mle = Mle::<CPUBackend, BaseField>::new(mle.to_vec());
            let cpu_res = cpu_mle.fix_first(assignment);
            return Mle::new(cpu_res.into_evals().into_iter().collect());
        }

        let assignment = PackedSecureField::broadcast(assignment);
        let packed_midpoint = midpoint / N_LANES;
        let (lhs_evals, rhs_evals) = mle.data.split_at(packed_midpoint);

        let res = zip(lhs_evals, rhs_evals)
            // Equivalent to `eq(0, assignment) * lhs_eval + eq(1, assignment) * rhs_eval`.
            .map(|(&lhs_eval, &rhs_eval)| assignment * (rhs_eval - lhs_eval) + lhs_eval)
            .collect();

        Mle::new(res)
    }
}

impl MleOps<SecureField> for SimdBackend {
    fn eval_at_point(mle: &Mle<Self, SecureField>, point: &[SecureField]) -> SecureField {
        fn simd_eval_mle_at_point(
            secure_mle: &[PackedSecureField],
            p: &[PackedSecureField],
        ) -> SecureField {
            match p {
                [] => unreachable!("WAAAAT"),
                &[p_1] => (p_1 * secure_mle[0]).pointwise_sum(),
                &[p_i, ref p @ ..] => {
                    let chunks = secure_mle.chunks_exact(secure_mle.len() / N_LANES);
                    let mut evals = chunks.map(|chunk| simd_eval_mle_at_point(chunk, p));
                    let evals_array = array::from_fn(|_| evals.next().unwrap());
                    (p_i * PackedSecureField::from_array(evals_array)).pointwise_sum()
                }
            }
        }

        if mle.len() < N_LANES {
            return eval_mle_at_point(&mle.to_vec(), point);
        }

        assert_eq!(point.len(), mle.num_variables());
        let point_chunks = point.rchunks_exact(N_LANES.ilog2() as usize);
        let point_remainder = point_chunks.remainder().as_slice();

        let packed_point = point_chunks
            .rev()
            .map(|point_chunk| {
                let eq_evals = gen_eq_evals(point_chunk, One::one());
                PackedSecureField::from_array(eq_evals.try_into().unwrap())
            })
            .collect_vec();

        let sub_evals = mle
            .data
            .chunks_exact(N_LANES.pow(packed_point.len() as u32 - 1))
            .map(|mle_chunk| simd_eval_mle_at_point(mle_chunk, &packed_point))
            .collect_vec();

        eval_mle_at_point(&sub_evals, point_remainder)
    }

    fn fix_first(mle: Mle<Self, SecureField>, assignment: SecureField) -> Mle<Self, SecureField> {
        let midpoint = mle.len() / 2;

        // Offload small instances to CPU implementation.
        if midpoint < N_LANES {
            let cpu_mle = Mle::<CPUBackend, SecureField>::new(mle.to_vec());
            let cpu_res = cpu_mle.fix_first(assignment);
            return Mle::new(cpu_res.into_evals().into_iter().collect());
        }

        let packed_midpoint = midpoint / N_LANES;
        let assignment = PackedSecureField::broadcast(assignment);
        let mut packed_evals = mle.into_evals().data;

        for i in 0..packed_midpoint {
            let lhs_eval = packed_evals[i];
            let rhs_eval = packed_evals[i + packed_midpoint];
            // Equivalent to `eq(0, assignment) * lhs_eval + eq(1, assignment) * rhs_eval`.
            packed_evals[i] += assignment * (rhs_eval - lhs_eval);
        }

        packed_evals.truncate(packed_midpoint);

        let length = packed_evals.len() * N_LANES;
        let data = packed_evals;

        Mle::new(SecureFieldVec { data, length })
    }
}

impl SumcheckOracle for Mle<SimdBackend, SecureField> {
    fn num_variables(&self) -> usize {
        self.num_variables()
    }

    fn univariate_sum(&self, claim: SecureField) -> UnivariatePolynomial<SecureField> {
        let x0 = SecureField::zero();
        let x1 = SecureField::one();

        let midpoint = self.len() / 2;

        // Offload small instances to CPU implementation.
        if midpoint < N_LANES {
            let cpu_mle = Mle::<CPUBackend, SecureField>::new(self.to_vec());
            return cpu_mle.univariate_sum(claim);
        }

        let packed_midpoint = midpoint / N_LANES;

        let y0 = self.data[0..packed_midpoint]
            .iter()
            .sum::<PackedSecureField>()
            .pointwise_sum();
        let y1 = claim - y0;

        UnivariatePolynomial::interpolate_lagrange(&[x0, x1], &[y0, y1])
    }

    fn fix_first(self, challenge: SecureField) -> Self {
        self.fix_first(challenge)
    }
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::blake2_hash::Blake2sHasher;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::CPUBackend;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::mle::Mle;

    #[test]
    fn simd_eval_matches_cpu_eval() {
        const NUM_VARIABLES: usize = 8;
        let values = test_channel().draw_felts(1 << NUM_VARIABLES);
        let point = test_channel().draw_felts(NUM_VARIABLES);
        let cpu_mle = Mle::<CPUBackend, SecureField>::new(values.clone());
        let simd_mle = Mle::<SimdBackend, SecureField>::new(values.into_iter().collect());

        assert_eq!(
            cpu_mle.eval_at_point(&point),
            simd_mle.eval_at_point(&point)
        );
    }

    fn test_channel() -> Blake2sChannel {
        let seed = Blake2sHasher::hash(&[]);
        Blake2sChannel::new(seed)
    }
}
