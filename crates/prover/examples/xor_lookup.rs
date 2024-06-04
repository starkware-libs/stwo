use std::array;
use std::iter::zip;

use itertools::{izip, Itertools};
use num_traits::{One, Zero};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo_prover::core::backend::{ColumnOps, CpuBackend};
use stwo_prover::core::channel::{Blake2sChannel, Channel};
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::FieldExpOps;
use stwo_prover::core::lookups::gkr::{
    partially_verify_batch, prove_batch, GkrArtifact, GkrBatchProof, GkrOps,
};
use stwo_prover::core::lookups::logup::{LogupGate, LogupTrace};
use stwo_prover::core::lookups::mle::Mle;
use stwo_prover::core::lookups::utils::horner_eval;
use stwo_prover::core::vcs::blake2_hash::Blake2sHash;

const TRACE_LOG_LEN: u32 = 16;
const TRACE_LEN: usize = 1 << TRACE_LOG_LEN;

/// Rectangular trace.
struct BaseTrace {
    /// 8-bit XOR LHS operands
    xor_lhs_column: Vec<BaseField>,
    /// 8-bit XOR RHS operands
    xor_rhs_column: Vec<BaseField>,
    /// 8-bit XOR results
    xor_res_column: Vec<BaseField>,
    /// Multiplicity of each 8-bit XOR.
    ///
    /// Index `i` stores the multiplicity of `(i & 0xFF) ^ ((i >> 8) & 0xFF)`.
    xor_multiplicities: Vec<BaseField>,
}

/// Rectangular trace.
struct InteractionTrace {
    /// Evals of `eq(r, x)` on `x` in the boolean hypercube and random `r` sampled by the verifier.
    eq_column: Vec<SecureField>,
    /// Random linear combination of GKR top layers (ones that don't have a succinct multilinear
    /// representation) multiplied pointwise by `eq_column`.
    acc_column: Vec<SecureField>,
}

fn main() {
    let (base_trace, extension_trace, proof) = prove();
    assert!(verify(base_trace, extension_trace, proof));
}

fn prove() -> (BaseTrace, InteractionTrace, GkrBatchProof) {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut channel = test_channel();

    let mut xor_lhs_column = Vec::new();
    let mut xor_rhs_column = Vec::new();
    let mut xor_res_column = Vec::new();
    let mut xor_multiplicities = vec![BaseField::zero(); 256 * 256];

    // Fill trace with random XOR instances.
    for _ in 0..TRACE_LEN {
        let lhs = rng.gen::<u8>() as usize;
        let rhs = rng.gen::<u8>() as usize;
        let res = lhs ^ rhs;

        xor_lhs_column.push(BaseField::from(lhs));
        xor_rhs_column.push(BaseField::from(rhs));
        xor_res_column.push(BaseField::from(res));

        xor_multiplicities[lhs + (rhs << 8)] += BaseField::one();
    }

    // Commit to the trace.
    let trace = BaseTrace {
        xor_lhs_column,
        xor_rhs_column,
        xor_res_column,
        xor_multiplicities,
    };

    // Draw LogUp lookup interaction elements.
    let z = channel.draw_felt();
    let alpha = channel.draw_felt();

    // Create lookup columns for GKR.
    let unordered_xor_denominators = izip!(
        &trace.xor_lhs_column,
        &trace.xor_rhs_column,
        &trace.xor_res_column
    )
    .map(|(&lhs, &rhs, &xor)| z - lhs - alpha * rhs - alpha.square() * xor)
    .collect_vec();

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

    // The GKR implementation operates on columns stored in bit-reversed order.
    let mut unordered_xor_denominators_bit_rev = unordered_xor_denominators.clone();
    CpuBackend::bit_reverse_column(&mut unordered_xor_denominators_bit_rev);
    let mut xor_multiplicities_bit_rev = trace.xor_multiplicities.clone();
    CpuBackend::bit_reverse_column(&mut xor_multiplicities_bit_rev);
    let mut ordered_xor_denominators_bit_rev = ordered_xor_denominators.clone();
    CpuBackend::bit_reverse_column(&mut ordered_xor_denominators_bit_rev);

    // Create GKR components
    let unordered_xor_logup = LogupTrace::<CpuBackend>::Singles {
        denominators: Mle::new(unordered_xor_denominators_bit_rev),
    };
    let ordered_xor_logup = LogupTrace::<CpuBackend>::Multiplicities {
        numerators: Mle::new(xor_multiplicities_bit_rev),
        denominators: Mle::new(ordered_xor_denominators_bit_rev),
    };
    let (gkr_proof, gkr_artifact) =
        prove_batch(&mut channel, vec![unordered_xor_logup, ordered_xor_logup]);

    // Columns of the same size can be batched for the univariate IOP for multilinear eval at point.
    // Assume ordered and unordered logup columns have the same size and can be batched together.
    assert_eq!(TRACE_LEN, 256 * 256);
    let acc_alpha = channel.draw_felt();

    // Create `eq(r, x)` column involved in univariate IOP for multivariate eval-at-point.
    let r = gkr_artifact.ood_point;
    let mut eq_column = CpuBackend::gen_eq_evals(&r, SecureField::one()).into_evals();
    CpuBackend::bit_reverse_column(&mut eq_column);

    // Perform a random linear combination of all GKR top layer columns that don't
    // have a succinct multilinear representation and multiply by the eq column.
    let mut acc_column = unordered_xor_denominators.clone();
    zip(&mut acc_column, &trace.xor_multiplicities).for_each(|(acc, &v)| *acc += v * acc_alpha);
    zip(&mut acc_column, &eq_column).for_each(|(acc, &eq_eval)| *acc *= eq_eval);

    // Commit to interaction trace.
    // TODO: Use univariate sum-check to do an eval-at-point (inner product of eq_col and acc_col).
    // TODO: Commit to univariate sum-check polynomials
    let interaction_trace = InteractionTrace {
        eq_column,
        acc_column,
    };

    // TODO: Constraint evaluation etc.

    (trace, interaction_trace, gkr_proof)
}

fn verify(
    base_trace: BaseTrace,
    interaction_trace: InteractionTrace,
    gkr_proof: GkrBatchProof,
) -> bool {
    let mut channel = test_channel();

    // Commit to the trace.
    _ = &base_trace;

    // Draw lookup interaction elements.
    let z = channel.draw_felt();
    let alpha = channel.draw_felt();

    // Check lookup claims match.
    let unordered_logup_output_claim = {
        let gkr_output = &gkr_proof.output_claims_by_component[0];
        let numerator = gkr_output[0];
        let denominator = gkr_output[1];
        numerator / denominator
    };

    let ordered_logup_output_claim = {
        let gkr_output = &gkr_proof.output_claims_by_component[1];
        let numerator = gkr_output[0];
        let denominator = gkr_output[1];
        numerator / denominator
    };

    assert_eq!(unordered_logup_output_claim, ordered_logup_output_claim);

    // Verify GKR proof (Convert the claim about the circuit output into a claim about evaluations
    // at random point `r` in the top layers).
    let GkrArtifact {
        ood_point: r,
        claims_to_verify_by_component,
        n_variables_by_component: _,
    } = partially_verify_batch(vec![&LogupGate; 2], &gkr_proof, &mut channel).unwrap();

    let acc_alpha = channel.draw_felt();

    // Commit to extension columns.
    _ = &interaction_trace;

    // Verify GKR top layer column claims.
    let unordered_logup_numerators_claim = claims_to_verify_by_component[0][0];
    let unordered_logup_denominators_claim = claims_to_verify_by_component[0][1];
    // Numerators for unordered LogUp are all one.
    assert!(unordered_logup_numerators_claim.is_one());

    let ordered_logup_numerators_claim = claims_to_verify_by_component[1][0];
    let ordered_logup_denominators_claim = claims_to_verify_by_component[1][1];
    // The denominator for ordered LogUp lookup table has a succinct multilinear representation so
    // the verifier can evaluate it directly.
    assert_eq!(
        ordered_logup_denominators_claim,
        eval_xor_denominator_multilinear(&r, z, alpha)
    );

    // The claims left to validate are the ordered multiplicities (`ordered_logup_numerators_claim`)
    // and the unordered denominators (`unordered_logup_denominators_claim`).
    let expected_inner_product =
        unordered_logup_denominators_claim + acc_alpha * ordered_logup_numerators_claim;

    // TODO: Use univariate sum-check protocol for this.
    let actual_inner_product = interaction_trace.acc_column.iter().sum::<SecureField>();
    assert_eq!(actual_inner_product, expected_inner_product);

    // Check the interaction trace columns are constructed correctly.
    // TODO: Once using univariate sum-check we will have to check the univariate sumcheck
    // polynomials are valid. Requires evaluating trace column involved in lookups at a
    // random circle point and checking those against the committed sum-check polynomials.

    // TODO: Check `eq(r, x)` column was constructed correctly using the AIR constraints
    // outlined in https://eprint.iacr.org/2023/1284.pdf.
    let r_rev = r.iter().rev().copied().collect_vec();
    let expected_eq_evals = CpuBackend::gen_eq_evals(&r_rev, SecureField::one());
    assert_eq!(interaction_trace.eq_column, *expected_eq_evals);

    // TODO: Check AIR constraints.

    true
}

fn eval_xor_denominator_multilinear(
    gkr_ood_point: &[SecureField],
    z: SecureField,
    alpha: SecureField,
) -> SecureField {
    match gkr_ood_point {
        // `li`: left operand bit `i`, `ri`: right operand bit `i`
        &[ref _unused @ .., l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7] => {
            let lhs_assignment = [l0, l1, l2, l3, l4, l5, l6, l7];
            let rhs_assignment = [r0, r1, r2, r3, r4, r5, r6, r7];

            let xor_assignment: [SecureField; 8] = array::from_fn(|i| {
                let a = lhs_assignment[i];
                let b = rhs_assignment[i];

                // Note `a ^ b = 1 - a * b - (1 - a)(1 - b)` for all `a, b` in `{0, 1}`.
                SecureField::one() - a * b - (SecureField::one() - a) * (SecureField::one() - b)
            });

            let two = BaseField::from(2).into();
            let lhs = horner_eval(&lhs_assignment, two);
            let rhs = horner_eval(&rhs_assignment, two);
            let xor = horner_eval(&xor_assignment, two);

            z - lhs - rhs * alpha - xor * alpha.square()
        }
        _ => panic!(),
    }
}

fn test_channel() -> Blake2sChannel {
    let seed = Blake2sHash::from(vec![0; 32]);
    Blake2sChannel::new(seed)
}
