use bytemuck::Zeroable;
use num_traits::{One, Zero};

use crate::core::air::evaluation::SecureColumn;
use crate::core::backend::avx512::cm31::PackedCM31;
use crate::core::backend::avx512::qm31::PackedQM31;
use crate::core::backend::avx512::{AVX512Backend, BaseFieldVec, K_BLOCK_SIZE};
use crate::core::backend::cpu::CpuMle;
use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr::{GkrLayer, GkrOps};
use crate::core::lookups::grand_product::{GrandProductOps, GrandProductOracle, GrandProductTrace};
use crate::core::lookups::mle::{ColumnV2, Mle};
use crate::core::lookups::sumcheck::SumcheckOracle;
use crate::core::lookups::utils::{eq, Polynomial};

impl GrandProductOps for AVX512Backend {
    fn next_layer(layer: &GrandProductTrace<Self>) -> GrandProductTrace<Self> {
        let mut col0 = Vec::new();
        let mut col1 = Vec::new();
        let mut col2 = Vec::new();
        let mut col3 = Vec::new();

        if layer.len() < 2 * K_BLOCK_SIZE {
            let layer_values = layer.to_vec();
            let next_layer = CPUBackend::next_layer(&GrandProductTrace::new(CpuMle::new(
                layer_values.into_iter().collect(),
            )));
            return GrandProductTrace::new(Mle::new(next_layer.to_vec().into_iter().collect()));
        }

        let packed_midpoint = layer.cols[0].data.len() / 2;

        for i in 0..packed_midpoint {
            let (c0_evens, c0_odds) =
                layer.cols[0].data[i * 2].deinterleave_with(layer.cols[0].data[i * 2 + 1]);

            let (c1_evens, c1_odds) =
                layer.cols[1].data[i * 2].deinterleave_with(layer.cols[1].data[i * 2 + 1]);

            let (c2_evens, c2_odds) =
                layer.cols[2].data[i * 2].deinterleave_with(layer.cols[2].data[i * 2 + 1]);

            let (c3_evens, c3_odds) =
                layer.cols[3].data[i * 2].deinterleave_with(layer.cols[3].data[i * 2 + 1]);

            let evens = PackedQM31([
                PackedCM31([c0_evens, c1_evens]),
                PackedCM31([c2_evens, c3_evens]),
            ]);

            let odds = PackedQM31([
                PackedCM31([c0_odds, c1_odds]),
                PackedCM31([c2_odds, c3_odds]),
            ]);

            let PackedQM31([PackedCM31([c0, c1]), PackedCM31([c2, c3])]) = evens * odds;

            col0.push(c0);
            col1.push(c1);
            col2.push(c2);
            col3.push(c3);
        }

        let length = layer.len() / 2;

        GrandProductTrace::new(Mle::new(SecureColumn {
            cols: [
                BaseFieldVec { data: col0, length },
                BaseFieldVec { data: col1, length },
                BaseFieldVec { data: col2, length },
                BaseFieldVec { data: col3, length },
            ],
        }))
    }

    fn univariate_sum(
        oracle: &GrandProductOracle<'_, Self>,
        claim: SecureField,
    ) -> Polynomial<SecureField> {
        let num_terms = 1 << (oracle.num_variables() - 1);
        let eq_evals = oracle.eq_evals();
        let trace = oracle.trace();
        let z = oracle.z();
        let r = oracle.r();

        if num_terms < 2 * K_BLOCK_SIZE {
            let trace_values = trace.to_vec();
            let trace = GrandProductTrace::new(CpuMle::new(trace_values.into_iter().collect()));
            let eq_evals = CPUBackend::gen_eq_evals(&z[0..trace.num_variables() - 2]);
            let oracle = trace.into_sumcheck_oracle(SecureField::zero(), z, &eq_evals);
            return oracle.univariate_sum(claim);
        }

        let mut packed_eval_at_0 = PackedQM31::zeroed();
        let mut packed_eval_at_2 = PackedQM31::zeroed();

        let num_packed_terms = num_terms / K_BLOCK_SIZE;

        for i in 0..num_packed_terms {
            let (c0_lhs_evens, c0_lhs_odds) =
                trace.cols[0].data[i * 2].deinterleave_with(trace.cols[0].data[i * 2 + 1]);

            let (c1_lhs_evens, c1_lhs_odds) =
                trace.cols[1].data[i * 2].deinterleave_with(trace.cols[1].data[i * 2 + 1]);

            let (c2_lhs_evens, c2_lhs_odds) =
                trace.cols[2].data[i * 2].deinterleave_with(trace.cols[2].data[i * 2 + 1]);

            let (c3_lhs_evens, c3_lhs_odds) =
                trace.cols[3].data[i * 2].deinterleave_with(trace.cols[3].data[i * 2 + 1]);

            let lhs_evens = PackedQM31([
                PackedCM31([c0_lhs_evens, c1_lhs_evens]),
                PackedCM31([c2_lhs_evens, c3_lhs_evens]),
            ]);

            let lhs_odds = PackedQM31([
                PackedCM31([c0_lhs_odds, c1_lhs_odds]),
                PackedCM31([c2_lhs_odds, c3_lhs_odds]),
            ]);

            let product0 = lhs_evens * lhs_odds;

            let (c0_rhs_evens, c0_rhs_odds) = trace.cols[0].data[(num_packed_terms + i) * 2]
                .deinterleave_with(trace.cols[0].data[(num_packed_terms + i) * 2 + 1]);

            let (c1_rhs_evens, c1_rhs_odds) = trace.cols[1].data[(num_packed_terms + i) * 2]
                .deinterleave_with(trace.cols[1].data[(num_packed_terms + i) * 2 + 1]);

            let (c2_rhs_evens, c2_rhs_odds) = trace.cols[2].data[(num_packed_terms + i) * 2]
                .deinterleave_with(trace.cols[2].data[(num_packed_terms + i) * 2 + 1]);

            let (c3_rhs_evens, c3_rhs_odds) = trace.cols[3].data[(num_packed_terms + i) * 2]
                .deinterleave_with(trace.cols[3].data[(num_packed_terms + i) * 2 + 1]);

            let rhs_evens = PackedQM31([
                PackedCM31([c0_rhs_evens, c1_rhs_evens]),
                PackedCM31([c2_rhs_evens, c3_rhs_evens]),
            ]);

            let rhs_odds = PackedQM31([
                PackedCM31([c0_rhs_odds, c1_rhs_odds]),
                PackedCM31([c2_rhs_odds, c3_rhs_odds]),
            ]);

            let product2 = (rhs_evens.double() - lhs_evens) * (rhs_odds.double() - lhs_odds);

            let eq_eval = PackedQM31([
                PackedCM31([eq_evals.cols[0].data[i], eq_evals.cols[1].data[i]]),
                PackedCM31([eq_evals.cols[2].data[i], eq_evals.cols[3].data[i]]),
            ]);

            packed_eval_at_0 += eq_eval * product0;
            packed_eval_at_2 += eq_eval * product2;
        }

        let mut eval_at_0 = packed_eval_at_0.to_array().into_iter().sum::<SecureField>();
        let mut eval_at_2 = packed_eval_at_2.to_array().into_iter().sum::<SecureField>();

        // We wanted to compute a sum of a multivariate polynomial
        // `eq((0^(k-1), x_k, .., x_n), (z_1, .., z_n)) * (..)` over
        // all `(x_k, ..., x_n)` in `{0, 1}^(n-k)`. Instead we computes a sum over
        // `eq((0^(k-2), x_k, .., x_n), (z_2, .., z_n)) * (..)`. The two multivariate sums differs
        // by a constant factor `eq((0), (z_1))` which is added back in here.
        //
        // The reason the factor is left out originally is for performance reasons. In the naive
        // version we want to precompute the evaluations of `eq((x_1, .., x_n), (z_1, .., z_n))`
        // ahead of time for all `(x_1, .., x_n)` in `{0, 1}^n`. Notice we only use half of
        // these evaluations (specifically those where `x_1` is zero). Each the term of the sum gets
        // multiplied by one of these evaluations. Notice all the terms of the sum contain a
        // constant factor `eq((x_1), (z_1))` (since x_1 equals zero). In the optimized
        // version we precompute the evaluations of `eq((x_2, .., x_n), (z_2, .., z_n))` which is
        // half the size (and takes half the work) of the original precomputation. We then add the
        // missing `eq((x_1), (z_1))` factor back here.
        //
        // TODO: Doc is a bit wordy it's not great have to explain all this but the optimization
        // is worthwhile. Consider modifying `gen_eq_evals()` so that it only returns the first
        // half. Would be just as optimized but prevent having to explain things here.
        eval_at_0 *= eq(&[SecureField::zero()], &[z[0]]);
        eval_at_2 *= eq(&[SecureField::zero()], &[z[0]]);

        // The evaluations on `0` and `2` are invalid. They were obtained by summing over the poly
        // `eq((0^(k-1), x_k, .., x_n), (z_1, .., z_n)) * (..)` but we require the sum to be taken
        // on `eq((r_1, ..., r_{k-1}, x_k, .., x_n), (z_1, .., z_n)) * (..)`. Conveniently
        // `eq((0^(k-1), x_k, .., x_n), (z_1, .., z_n))` and `eq((r_1, ..., r_{k-1}, x_k, .., x_n),
        // (z_1, .., z_n))` differ only by a constant factor `eq((r_1, ..., r_{k-1}), (z_1, ..,
        // z_{k-1})) / eq((0^(k-1)), (z_1, .., z_{k-1}))` for all values of `x`.
        // TODO: explain
        let k = r.len();
        let eq_correction_factor = eq(r, &z[0..k]) / eq(&vec![SecureField::zero(); k], &z[0..k]);

        // Our goal is to compute the sum of `eq((x_k, .., x_n), (z_k, .., z_n)) * h(x_k, .., x_n)`
        // over all possible values `(x_{k+1}, .., x_n)` in `{0, 1}^{n-1}`, effectively reducing the
        // sum to a univariate polynomial in `x_k`. Let this univariate polynomial be `f`. Our
        // method to is to evaluate `f` in `deg(f) + 1` points (which can be done efficiently) to
        // obtain the coefficient representation of `f` via interpolation.
        //
        // Although evaluating `f` is efficient, the runtime of the sumcheck prover is proportional
        // to how many points `f` needs to be evaluated on. To reduce the number of evaluations the
        // prover must perform we can reduce the degree of of the polynomial we need to interpolate.
        // This can be done by instead computing the sum over `eq((0, .., x_n), (z_k, .., z_n)) *
        // h(x_k, .., x_n)` denoted `simplified_sum` which has degree `deg(f) - 1`. We interpolate,
        // our lower degree polynomial, `simplified_sum` with one less evaluation and multiply it
        // afterwards by `eq((x_k), (z_k)) / eq((0), (z_k))` to obtain the original `f`. This idea
        // and algorithm is from <https://eprint.iacr.org/2024/108.pdf> (Section 3.2).
        let correction_factor_at = |x| eq(&[x], &[z[k]]) / eq(&[SecureField::zero()], &[z[k]]);

        let x0: SecureField = BaseField::zero().into();
        let x1 = BaseField::one().into();
        let x2 = BaseField::from(2).into();

        let mut y0 = eq_correction_factor * eval_at_0;
        let mut y1 = (claim - y0) / correction_factor_at(x1);
        let mut y2 = eq_correction_factor * eval_at_2;

        // We are interpolating a degree 2 function so need three evaluations.
        let simplified_univariate_sum =
            Polynomial::interpolate_lagrange(&[x0, x1, x2], &[y0, y1, y2]);

        let x3 = BaseField::from(3).into();
        let mut y3 = simplified_univariate_sum.eval(x3);

        // Correct all the evaluations (see comment above).
        y0 *= correction_factor_at(x0); // `y0 *= 1`
        y1 *= correction_factor_at(x1);
        y2 *= correction_factor_at(x2);
        y3 *= correction_factor_at(x3);

        Polynomial::interpolate_lagrange(&[x0, x1, x2, x3], &[y0, y1, y2, y3])
    }
}
