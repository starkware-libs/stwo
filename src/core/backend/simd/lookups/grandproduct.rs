use itertools::Itertools;
use num_traits::Zero;

use crate::core::backend::simd::column::SecureFieldVec;
use crate::core::backend::simd::m31::N_LANES;
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{CPUBackend, Column};
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr::{EqEvals, GkrLayer};
use crate::core::lookups::mle::Mle;
use crate::core::lookups::sumcheck::SumcheckOracle;
use crate::core::lookups::{
    GrandProductOps, GrandProductOracle, GrandProductTrace, UnivariateEvals,
};

impl GrandProductOps for SimdBackend {
    fn next_layer(layer: &GrandProductTrace<Self>) -> GrandProductTrace<Self> {
        let half_len = layer.len() / 2;

        if half_len < N_LANES {
            let cpu_layer = GrandProductTrace::<CPUBackend>::new(Mle::new(layer.to_vec()));
            let cpu_next_layer = cpu_layer.next().unwrap().to_vec();
            return GrandProductTrace::new(Mle::new(cpu_next_layer.into_iter().collect()));
        }

        let data = layer
            .data
            .array_chunks()
            .map(|&[a, b]| {
                let (evens, odds) = a.deinterleave(b);
                evens * odds
            })
            .collect_vec();

        GrandProductTrace::new(Mle::new(SecureFieldVec {
            data,
            length: half_len,
        }))
    }

    fn univariate_sum_evals(oracle: &GrandProductOracle<'_, Self>) -> UnivariateEvals {
        let num_terms = 1 << (oracle.num_variables() - 1);
        let eq_evals = oracle.eq_evals();
        let trace = oracle.trace();

        // Offload small instances to CPU backend to avoid
        // complexity with packed SIMD types.
        if num_terms < 2 * N_LANES {
            // TODO(andrew): This is a bit poor. Could be improved.
            let eq_evals = EqEvals {
                evals: Mle::new((0..N_LANES).map(|i| eq_evals.at(i)).collect_vec()),
                y: eq_evals.y().to_vec(),
            };

            let trace = GrandProductTrace::<CPUBackend>::new(Mle::new(trace.to_vec()));
            let oracle = trace.into_sumcheck_oracle(SecureField::zero(), 0, &eq_evals);
            return CPUBackend::univariate_sum_evals(&oracle);
        }

        let packed_num_terms = num_terms / N_LANES;
        let (lhs_data, rhs_data) = trace.data.split_at(trace.data.len() / 2);

        let mut eval_at_0 = PackedSecureField::zero();
        let mut eval_at_2 = PackedSecureField::zero();

        #[allow(clippy::needless_range_loop)]
        for i in 0..packed_num_terms {
            let (lhs0, lhs1) = lhs_data[i * 2].deinterleave(lhs_data[i * 2 + 1]);
            let (rhs0, rhs1) = rhs_data[i * 2].deinterleave(rhs_data[i * 2 + 1]);

            let product2 = (rhs0.double() - lhs0) * (rhs1.double() - lhs1);
            let product0 = lhs0 * lhs1;

            let eq_eval = eq_evals.data[i];
            eval_at_0 += eq_eval * product0;
            eval_at_2 += eq_eval * product2;
        }

        let eval_at_0 = eval_at_0.pointwise_sum();
        let eval_at_2 = eval_at_2.pointwise_sum();

        UnivariateEvals {
            eval_at_0,
            eval_at_2,
        }
    }
}
