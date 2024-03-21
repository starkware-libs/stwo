use std::borrow::Cow;

use bytemuck::Zeroable;
use num_traits::Zero;

use crate::core::air::evaluation::SecureColumn;
use crate::core::backend::avx512::cm31::PackedCM31;
use crate::core::backend::avx512::qm31::PackedQM31;
use crate::core::backend::avx512::{AVX512Backend, BaseFieldVec, K_BLOCK_SIZE};
use crate::core::backend::cpu::CpuMle;
use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::grand_product::{GrandProductOps, GrandProductOracle, GrandProductTrace};
use crate::core::lookups::mle::{ColumnV2, Mle};
use crate::core::lookups::sumcheck::{SumcheckOracle, UnivariateEvals};

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

    fn univariate_sum_evals(oracle: &GrandProductOracle<'_, Self>) -> UnivariateEvals {
        let num_terms = 1 << (oracle.num_variables() - 1);
        let eq_evals = oracle.eq_evals();
        let trace = oracle.trace();

        println!("Num terms: {num_terms}");

        let (col0, col1, col2, col3) = if num_terms >= 2 * K_BLOCK_SIZE {
            (
                Cow::Borrowed(&trace.cols[0]),
                Cow::Borrowed(&trace.cols[1]),
                Cow::Borrowed(&trace.cols[2]),
                Cow::Borrowed(&trace.cols[3]),
            )
        } else {
            let mut col0_lhs = [BaseField::zero(); 2 * K_BLOCK_SIZE];
            let mut col0_rhs = [BaseField::zero(); 2 * K_BLOCK_SIZE];
            let mut col1_lhs = [BaseField::zero(); 2 * K_BLOCK_SIZE];
            let mut col1_rhs = [BaseField::zero(); 2 * K_BLOCK_SIZE];
            let mut col2_lhs = [BaseField::zero(); 2 * K_BLOCK_SIZE];
            let mut col2_rhs = [BaseField::zero(); 2 * K_BLOCK_SIZE];
            let mut col3_lhs = [BaseField::zero(); 2 * K_BLOCK_SIZE];
            let mut col3_rhs = [BaseField::zero(); 2 * K_BLOCK_SIZE];

            let midpoint = trace.len() / 2;
            println!("MID {midpoint}");
            col0_lhs[0..midpoint].copy_from_slice(&trace.cols[0].as_slice()[..midpoint]);
            col0_rhs[0..midpoint].copy_from_slice(&trace.cols[0].as_slice()[midpoint..]);

            col1_lhs[0..midpoint].copy_from_slice(&trace.cols[1].as_slice()[..midpoint]);
            col1_rhs[0..midpoint].copy_from_slice(&trace.cols[1].as_slice()[midpoint..]);

            col2_lhs[0..midpoint].copy_from_slice(&trace.cols[2].as_slice()[..midpoint]);
            col2_rhs[0..midpoint].copy_from_slice(&trace.cols[2].as_slice()[midpoint..]);

            col3_lhs[0..midpoint].copy_from_slice(&trace.cols[3].as_slice()[..midpoint]);
            col3_rhs[0..midpoint].copy_from_slice(&trace.cols[3].as_slice()[midpoint..]);

            (
                Cow::Owned([col0_lhs, col0_rhs].concat().into_iter().collect()),
                Cow::Owned([col1_lhs, col1_rhs].concat().into_iter().collect()),
                Cow::Owned([col2_lhs, col2_rhs].concat().into_iter().collect()),
                Cow::Owned([col3_lhs, col3_rhs].concat().into_iter().collect()),
            )
        };

        let (col0, col1, col2, col3) = (col0.as_ref(), col1.as_ref(), col2.as_ref(), col3.as_ref());

        let mut packed_eval_at_0 = PackedQM31::zeroed();
        let mut packed_eval_at_2 = PackedQM31::zeroed();

        let num_packed_terms = num_terms / K_BLOCK_SIZE;

        for i in 0..num_packed_terms {
            let (c0_lhs_evens, c0_lhs_odds) =
                col0.data[i * 2].deinterleave_with(col0.data[i * 2 + 1]);

            let (c1_lhs_evens, c1_lhs_odds) =
                col1.data[i * 2].deinterleave_with(col1.data[i * 2 + 1]);

            let (c2_lhs_evens, c2_lhs_odds) =
                col2.data[i * 2].deinterleave_with(col2.data[i * 2 + 1]);

            let (c3_lhs_evens, c3_lhs_odds) =
                col3.data[i * 2].deinterleave_with(col3.data[i * 2 + 1]);

            let lhs_evens = PackedQM31([
                PackedCM31([c0_lhs_evens, c1_lhs_evens]),
                PackedCM31([c2_lhs_evens, c3_lhs_evens]),
            ]);

            let lhs_odds = PackedQM31([
                PackedCM31([c0_lhs_odds, c1_lhs_odds]),
                PackedCM31([c2_lhs_odds, c3_lhs_odds]),
            ]);

            let product0 = lhs_evens * lhs_odds;

            let (c0_rhs_evens, c0_rhs_odds) = col0.data[(num_packed_terms + i) * 2]
                .deinterleave_with(col0.data[(num_packed_terms + i) * 2 + 1]);

            let (c1_rhs_evens, c1_rhs_odds) = col1.data[(num_packed_terms + i) * 2]
                .deinterleave_with(col1.data[(num_packed_terms + i) * 2 + 1]);

            let (c2_rhs_evens, c2_rhs_odds) = col2.data[(num_packed_terms + i) * 2]
                .deinterleave_with(col2.data[(num_packed_terms + i) * 2 + 1]);

            let (c3_rhs_evens, c3_rhs_odds) = col3.data[(num_packed_terms + i) * 2]
                .deinterleave_with(col3.data[(num_packed_terms + i) * 2 + 1]);

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

        let eval_at_0 = packed_eval_at_0.to_array().into_iter().sum::<SecureField>();
        let eval_at_2 = packed_eval_at_2.to_array().into_iter().sum::<SecureField>();

        UnivariateEvals {
            eval_at_0,
            eval_at_2,
        }
    }
}
