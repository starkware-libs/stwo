use std::collections::BTreeSet;

use itertools::{chain, izip, Itertools};
use num_traits::One;

use super::xor_table_component::{XOR_ALPHA_ID, XOR_Z_ID};
use crate::core::air::{
    Air, AirProver, AirTraceVerifier, AirTraceWriter, Component, ComponentProver,
};
use crate::core::backend::cpu::CpuCirclePoly;
use crate::core::backend::CpuBackend;
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;
use crate::core::fields::FieldExpOps;
use crate::core::lookups::gkr_prover::{prove_batch, GkrOps, Layer};
use crate::core::lookups::gkr_verifier::{GkrArtifact, GkrBatchProof};
use crate::core::lookups::mle::Mle;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly};
use crate::core::poly::BitReversedOrder;
use crate::core::{ColumnVec, InteractionElements};
use crate::examples::xor2::unordered_xor_component::UnorderedXorComponent;
use crate::examples::xor2::xor_table_component::XorTableComponent;
use crate::examples::xor2::{LOG_TRACE_LEN, TRACE_LEN};

pub struct XorAir;

impl Air for XorAir {
    fn components(&self) -> Vec<&dyn Component> {
        vec![&UnorderedXorComponent, &XorTableComponent]
    }
}

impl AirTraceVerifier for XorAir {
    fn interaction_elements(&self, channel: &mut Blake2sChannel) -> InteractionElements {
        let ids = self
            .components()
            .iter()
            .flat_map(|c| c.interaction_element_ids())
            .collect::<BTreeSet<String>>();
        InteractionElements::new(
            ids.into_iter()
                .map(|id| (id, channel.draw_felt()))
                .collect(),
        )
    }
}

impl AirTraceWriter<CpuBackend> for XorAir {
    fn interact(
        &self,
        channel: &mut Blake2sChannel,
        trace: &ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
        elements: &InteractionElements,
    ) -> (
        Vec<CirclePoly<CpuBackend>>,
        GkrBatchProof,
        GkrArtifact,
        SecureField,
    ) {
        let z = elements[XOR_Z_ID];
        let alpha = elements[XOR_ALPHA_ID];

        // Create lookup columns for GKR.
        let unordered_xor_denominators = {
            let xor_lhs_col = &**trace[0];
            let xor_rhs_col = &**trace[1];
            let xor_res_col = &**trace[2];

            izip!(xor_lhs_col, xor_rhs_col, xor_res_col)
                .map(|(&lhs, &rhs, &xor)| z - lhs - alpha * rhs - alpha.square() * xor)
                .collect_vec()
        };

        let ordered_xor_numerators = {
            let xor_multiplicities_col = &trace[3];
            xor_multiplicities_col.to_vec()
        };

        let ordered_xor_denominators = (0..256 * 256)
            .map(|i| {
                let lhs = i & 0xFF;
                let rhs = i >> 8;
                let res = lhs ^ rhs;
                z - BaseField::from(lhs)
                    - alpha * BaseField::from(rhs)
                    - alpha.square() * BaseField::from(res)
            })
            .collect_vec();

        let unordered_xor_logup = Layer::<CpuBackend>::LogUpSingles {
            denominators: Mle::new(
                unordered_xor_denominators
                    .iter()
                    .map(|&v| v.into())
                    .collect(),
            ),
        };

        let ordered_xor_logup = Layer::<CpuBackend>::LogUpMultiplicities {
            numerators: Mle::new(ordered_xor_numerators.iter().map(|&v| v.into()).collect()),
            denominators: Mle::new(ordered_xor_denominators.iter().map(|&v| v.into()).collect()),
        };

        let (gkr_proof, gkr_artifact) =
            prove_batch(channel, vec![unordered_xor_logup, ordered_xor_logup]);

        // Columns of the same size can be batched for the univariate IOP for multilinear eval at
        // point. Assume ordered and unordered logup columns have the same size and can be
        // batched together.
        assert_eq!(TRACE_LEN, 256 * 256);
        let acc_alpha = channel.draw_felt();

        // Create `eq(r, x)` column involved in univariate IOP for multivariate eval-at-point.
        let r = &gkr_artifact.ood_point;
        let r_rev = r.iter().copied().rev().collect_vec();
        let bit_rev_eq_evals_col =
            CpuBackend::gen_eq_evals(&r_rev, SecureField::one()).into_evals();

        // Perform a random linear combination of all GKR top layer columns that don't
        // have a succinct multilinear representation and multiply by the eq column.
        let acc_column = (0..TRACE_LEN)
            .map(|i| {
                let eq_eval = bit_rev_eq_evals_col[i];
                let ordered_numerator = ordered_xor_numerators[i];
                let unordered_denominator = unordered_xor_denominators[i];

                eq_eval * (ordered_numerator + acc_alpha * unordered_denominator)
            })
            .collect_vec();

        // TODO: Generate univariate sumcheck polynomial `h`.

        let bit_rev_eq_evals_col: SecureColumn<CpuBackend> =
            bit_rev_eq_evals_col.into_iter().collect();
        let acc_column: SecureColumn<CpuBackend> = acc_column.into_iter().collect();

        let sumcheck_beta_plus_g_column = acc_column.clone();
        let sumcheck_h_column = acc_column;

        let interaction_trace_columns = chain!(
            bit_rev_eq_evals_col.columns,
            sumcheck_beta_plus_g_column.columns,
            sumcheck_h_column.columns
        )
        .map(|col| {
            let domain = CanonicCoset::new(LOG_TRACE_LEN).circle_domain();
            let col_evals =
                CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(domain, col);
            col_evals.interpolate()
        })
        .collect();

        (
            interaction_trace_columns,
            gkr_proof,
            gkr_artifact,
            acc_alpha,
        )
    }

    fn to_air_prover(&self) -> &impl AirProver<CpuBackend> {
        self
    }
}

impl AirProver<CpuBackend> for XorAir {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<CpuBackend>> {
        vec![]
    }
}

/// Represents polynomial `f` as constituent parts needed for performing univariate sumcheck.
///
/// Let `f(x, y) = beta + g(x, y) + Z_H(x) * h(x, y)` where `g(0, 0) = 0`, `deg(g) < |H| - 1` and
/// `Z_H` is the vanishing polynomial of sum domain `H`.
///
/// See https://eprint.iacr.org/2018/828.pdf (section 5)
pub struct UnivariateSumcheckPoly {
    _g: CpuCirclePoly,
    _h: CpuCirclePoly,
    _beta: BaseField,
    _sum_domain: CanonicCoset,
}

// impl UnivariateSumcheckPoly {
//     /// Decomposes circle polynomial `f` into the constituents required for univariate sum-check.
//     pub fn decompose(f: CpuCirclePoly, sum_domain: CanonicCoset) -> Self {
//         if f.log_size() > sum_domain.log_size() {
//             let eval_domain = CanonicCoset::new(f.log_size()).circle_domain();

//             let g_plus_beta_evals = CpuCircleEvaluation::<BaseField, NaturalOrder>::new(
//                 sum_domain.circle_domain(),
//                 sum_domain
//                     .circle_domain()
//                     .iter()
//                     .map(|p| f.eval_at_point(p.into_ef()).0 .0)
//                     .collect(),
//             );

//             let g_plus_beta = g_plus_beta_evals.bit_reverse().interpolate();

//             let mut g_plus_beta_evals = g_plus_beta.evaluate(eval_domain).values;
//             CpuBackend::bit_reverse_column(&mut g_plus_beta_evals);

//             let mut f_evals = f.evaluate(eval_domain).values;
//             CpuBackend::bit_reverse_column(&mut f_evals);

//             let h_evals = izip!(f_evals, g_plus_beta_evals, eval_domain)
//                 .map(|(f_eval, g_plus_beta_eval, p)| {
//                     (f_eval - g_plus_beta_eval) / coset_vanishing(sum_domain.coset(), p)
//                 })
//                 .collect();

//             let h = CpuCircleEvaluation::<BaseField, NaturalOrder>::new(eval_domain, h_evals)
//                 .bit_reverse()
//                 .interpolate();

//             let p = SECURE_FIELD_CIRCLE_GEN;
//             assert_eq!(
//                 g_plus_beta.eval_at_point(p)
//                     + coset_vanishing(sum_domain.coset(), p) * h.eval_at_point(p),
//                 f.eval_at_point(p),
//             );

//             let (g0, g1, beta) = decompose_g_plus_beta(g_plus_beta);

//             Self {
//                 g0,
//                 g1,
//                 h,
//                 beta,
//                 sum_domain,
//             }
//         } else {
//             // Note `f(x, y) = beta + g(x, y) + Z_H(x) * 0`.
//             let h = CpuCirclePoly::new(vec![BaseField::zero()]);
//             let (g0, g1, beta) = decompose_g_plus_beta(f);

//             Self {
//                 g0,
//                 g1,
//                 h,
//                 beta,
//                 sum_domain,
//             }
//         }
//     }

//     pub fn eval_at_point(&self, p: CirclePoint<SecureField>) -> SecureField {
//         let Self {
//             g0,
//             g1,
//             h,
//             beta,
//             sum_domain,
//         } = self;

//         let g_eval = p.x * g0.eval_at_point(p.x) + p.y * g1.eval_at_point(p.x);

//         g_eval + coset_vanishing(sum_domain.coset(), p) * h.eval_at_point(p) + *beta
//     }
// }
