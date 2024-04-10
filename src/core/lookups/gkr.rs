//! GKR protocol implementation designed to prove and verify lookup arguments.
use std::iter::successors;
use std::ops::Deref;

// use num_traits::Zero;
use thiserror::Error;

use super::mle::{Mle, MleOps, MleTrace};
use super::sumcheck::{SumcheckError, SumcheckOracle, SumcheckProof};
use super::utils::eq;
use super::GrandProductTrace;
use crate::core::backend::{CPUBackend, Col, Column, ColumnOps};
use crate::core::channel::Channel;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::utils::horner_eval;
use crate::core::lookups::{sumcheck, GrandProductCircuit};

pub trait GkrOps: MleOps<SecureField> {
    /// Returns the evaluations of [`eq(x, y)`] for all values of `x = (x_1, ..., x_n)` where
    /// `x_1 = 0` and `x_2, ..., x_n` in `{0, 1}`.
    ///
    /// [`eq(x, y)`]: crate::core::lookups::utils::eq
    fn gen_eq_evals(y: &[SecureField]) -> Mle<Self, SecureField>;
}

/// Stores evaluations of [`eq(x, y)`] for all values of `x = (x_1, ..., x_n)` where `x_1 = 0` and
/// `x_2, ..., x_n` in `{0, 1}`.
///
/// Evaluations are stored in bit-reversed order i.e. `evals[0] = eq((0, ..., 0, 0), y)`,
/// `evals[1] = eq((0, ..., 0, 1), y)`, etc.
///
/// [`eq(x, y)`]: crate::core::lookups::utils::eq
pub struct EqEvals<B: ColumnOps<SecureField>> {
    y: Vec<SecureField>,
    evals: Mle<B, SecureField>,
}

impl<B: GkrOps> EqEvals<B> {
    pub fn new(y: &[SecureField]) -> Self {
        let y = y.to_vec();
        let evals = B::gen_eq_evals(&y);
        assert_eq!(evals.len(), 1 << y.len().saturating_sub(1));
        Self { evals, y }
    }

    pub fn y(&self) -> &[SecureField] {
        &self.y
    }
}

impl<B: ColumnOps<SecureField>> Deref for EqEvals<B> {
    type Target = Col<B, SecureField>;

    fn deref(&self) -> &Col<B, SecureField> {
        &self.evals
    }
}

pub trait GkrLayer: Sized {
    type Backend: GkrOps + MleOps<SecureField>;
    type SumcheckOracle<'a>: GkrSumcheckOracle<Backend = Self::Backend>;

    /// Produces the next GKR layer from the current layer.
    ///
    /// Returns [`None`] if the current layer is the output layer.
    fn next(&self) -> Option<Self>;

    /// Transforms layer `l+1` (current layer) into a sumcheck oracle for layer `l` (next layer).
    fn into_sumcheck_oracle(
        self,
        lambda: SecureField,
        eq_evals: &EqEvals<Self::Backend>,
    ) -> Self::SumcheckOracle<'_>;

    /// Returns this layer as a [`MleTrace`].
    ///
    /// Currently used to obtain the output values.
    fn into_trace(self) -> MleTrace<Self::Backend, SecureField>;
}

// TODO: Merge `GkrSumcheckOracle` and `SumcheckOracle`
pub trait GkrSumcheckOracle: SumcheckOracle {
    type Backend: GkrOps;

    /// Returns the multilinear extensions in layer `l+1` that define the current `g` for layer `l`.
    // TODO: Document better. The sumcheck oracle gets transformed due to the fixing of variables.
    // The multilinear extensions returned by this function have the same fixed variables.
    fn into_inputs(self) -> MleTrace<Self::Backend, SecureField> {
        todo!()
    }
}

/// Defines a circuit with a binary tree structure.
///
/// Defines how the circuit operates locally on pairs of neighboring input rows to produce a
/// single output row. This local 2-to-1 constraint is what gives the whole circuit its "binary
/// tree" structure. Circuit examples: [`LogupCircuit`], [`GrandProductCircuit`].
///
/// A binary tree circuit has a highly "regular" wiring pattern that fits the structure of the
/// circuits defined in [Thaler13] which allows for an efficient linear time (linear in size of the
/// circuit) implementation of a GKR prover.
///
/// [Thaler13]: https://eprint.iacr.org/2013/351.pdf
pub trait BinaryTreeCircuit {
    /// Returns the output row after applying the circuit to the provided neighboring input rows.
    fn eval(even_row: &[SecureField], odd_row: &[SecureField]) -> Vec<SecureField>;
}

// <https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf> (page 65)
pub fn prove(channel: &mut impl Channel, top_layer: GrandProductTrace<CPUBackend>) -> GkrProof {
    let layers = successors(Some(top_layer), |layer| layer.next()).collect::<Vec<_>>();
    let mut layers = layers.into_iter().rev();

    let output_trace = layers.next().unwrap().into_trace();
    assert_eq!(output_trace.num_variables(), 0);
    let output_claims = output_trace.eval_at_point(&[]);
    println!("prod {}", output_claims[0]);

    let mut ood_point = vec![];
    let mut claims_to_verify = output_claims.clone();

    let layer_proofs = layers
        .map(|layer| {
            channel.mix_felts(&claims_to_verify);
            let lambda = channel.draw_felt();
            let eq_evals = EqEvals::new(&ood_point);
            let layer_copy = layer.clone();
            let sumcheck_oracle = layer.into_sumcheck_oracle(lambda, &eq_evals);
            let sumcheck_claim = horner_eval(&claims_to_verify, lambda);
            let (sumcheck_proof, sumcheck_ood_point, oracle, sumcheck_eval) =
                sumcheck::prove(sumcheck_claim, sumcheck_oracle, channel);
            // println!("prover assignment: {:?}", sumcheck_ood_point);

            let input_mle_trace = oracle.into_inputs();
            let input_mask = GkrMask::try_from(input_mle_trace.clone()).unwrap();
            channel.mix_felts(input_mask.columns().flatten());

            println!("yo {:?}", sumcheck_eval);
            let [even_row, odd_row] = input_mask.to_rows();
            println!(
                "=bro {:?}",
                eq(&ood_point, &sumcheck_ood_point)
                    * GrandProductCircuit::eval(&even_row, &odd_row)[0]
            );

            let r_star = channel.draw_felt();
            claims_to_verify = input_mle_trace.eval_at_point(&[r_star]);
            ood_point = sumcheck_ood_point;
            ood_point.push(r_star);
            println!("1tyo: {}", layer_copy.eval_at_point(&ood_point));
            println!("2tyo: {}", claims_to_verify[0]);
            println!("3tyo: {}", {
                let mut layer_copy = layer_copy.0;
                for point in &ood_point {
                    layer_copy = layer_copy.fix_first(*point)
                }
                layer_copy[0]
            });

            GkrLayerProof {
                sumcheck_proof,
                input_mask,
            }
        })
        .collect();

    GkrProof {
        layer_proofs,
        output_claims,
    }
}

/// Partially verifies a GKR proof.
///
/// On successful verification the function Returns a [`GkrVerificationArtifact`] which stores the
/// out-of-domain point and claimed evaluations in the top layer's columns at the OOD point. These
/// claimed evaluations are not checked by this function - hence partial verification.
pub fn partially_verify<C: BinaryTreeCircuit>(
    proof: &GkrProof,
    channel: &mut impl Channel,
) -> Result<GkrVerificationArtifact, GkrError> {
    let GkrProof {
        layer_proofs,
        output_claims,
    } = proof;

    let mut ood_point = vec![];
    let mut claims_to_verify = output_claims.to_vec();

    for (layer, layer_proof) in layer_proofs.iter().enumerate() {
        let GkrLayerProof {
            sumcheck_proof,
            input_mask,
        } = layer_proof;

        channel.mix_felts(&claims_to_verify);
        let lambda = channel.draw_felt();
        let sumcheck_claim = horner_eval(&claims_to_verify, lambda);
        let (sumcheck_ood_point, sumcheck_eval) =
            sumcheck::partially_verify(sumcheck_claim, sumcheck_proof, channel)
                .map_err(|source| GkrError::InvalidSumcheck { layer, source })?;

        println!("verifier assignment: {:?}", sumcheck_ood_point);

        let [input_row_0, input_row_1] = input_mask.to_rows();
        let circuit_output = C::eval(&input_row_0, &input_row_1);
        let folded_output = horner_eval(&circuit_output, lambda);
        let layer_eval = eq(&ood_point, &sumcheck_ood_point) * folded_output;

        if sumcheck_eval != layer_eval {
            return Err(GkrError::CircuitCheckFailure {
                claim: sumcheck_eval,
                output: layer_eval,
                layer,
            });
        }

        println!("made it");

        // v0 [p0, p1]
        // v0, v1 [p0, p1, p2, p3]
        // v0, v1, v2, v3 [p0, p1, p2, p3, p4, p5, p6, p7]

        channel.mix_felts(input_mask.columns().flatten());
        let r_star = channel.draw_felt();
        ood_point = sumcheck_ood_point;
        ood_point.push(r_star);

        claims_to_verify = input_mask.to_mle_trace().eval_at_point(&[r_star]);
    }

    Ok(GkrVerificationArtifact {
        ood_point,
        claims_to_verify,
    })
}

/// Error encountered during GKR protocol verification.
///
/// Layer 0 corresponds to the output layer.
#[derive(Error, Debug)]
pub enum GkrError {
    #[error("sum-check invalid in layer {layer}: {source}")]
    InvalidSumcheck { layer: usize, source: SumcheckError },
    #[error("circuit check failed in layer {layer} (calculated {output}, claim {claim})")]
    CircuitCheckFailure {
        claim: SecureField,
        output: SecureField,
        layer: usize,
    },
}

#[derive(Debug, Clone)]
pub struct GkrProof {
    layer_proofs: Vec<GkrLayerProof>,
    output_claims: Vec<SecureField>,
}

/// GKR partial verification artifact.
pub struct GkrVerificationArtifact {
    /// Out-of-domain (OOD) point for columns in the top layer.
    pub ood_point: Vec<SecureField>,
    /// The claimed evaluation at `ood_point` for each column in the top layer.
    pub claims_to_verify: Vec<SecureField>,
}

/// GKR layer input mask.
///
/// Stores two rows.
#[derive(Debug, Clone)]
struct GkrMask {
    columns: Vec<[SecureField; 2]>,
}

impl GkrMask {
    fn to_rows(&self) -> [Vec<SecureField>; 2] {
        self.columns.iter().map(|[a, b]| (a, b)).unzip().into()
    }

    fn columns(&self) -> &[[SecureField; 2]] {
        &self.columns
    }

    fn to_mle_trace(&self) -> MleTrace<CPUBackend, SecureField> {
        let columns = self
            .columns
            .iter()
            .map(|&column| Mle::new(column.into_iter().collect()))
            .collect();
        MleTrace::new(columns)
    }
}

impl<B: MleOps<SecureField>> TryFrom<MleTrace<B, SecureField>> for GkrMask {
    type Error = InvalidNumRowsError;

    fn try_from(trace: MleTrace<B, SecureField>) -> Result<Self, InvalidNumRowsError> {
        let num_rows = 1 << trace.num_variables();

        if num_rows != 2 {
            return Err(InvalidNumRowsError { num_rows });
        }

        Ok(Self {
            columns: trace
                .into_columns()
                .into_iter()
                .map(|column| column.to_vec().try_into().unwrap())
                .collect(),
        })
    }
}

#[derive(Debug, Error)]
#[error("trace has an invalid number of rows (given {num_rows}, expected 2)")]
struct InvalidNumRowsError {
    num_rows: usize,
}

// TODO(Andrew): Remove generic on proof structs. Consider using `Vec` instead of `MleTrace`.
#[derive(Debug, Clone)]
struct GkrLayerProof {
    sumcheck_proof: SumcheckProof,
    input_mask: GkrMask,
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::{partially_verify, prove, GkrError};
    use crate::commitment_scheme::blake2_hash::Blake2sHash;
    use crate::core::backend::CPUBackend;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::lookups::gkr::GkrVerificationArtifact;
    use crate::core::lookups::mle::Mle;
    use crate::core::lookups::{GrandProductCircuit, GrandProductTrace};

    // #[test]
    // fn avx_eq_evals_matches_cpu_eq_evals() {
    //     const LOG_SIZE: usize = 8;
    //     let mut rng = test_channel();
    //     let assignment: [SecureField; LOG_SIZE] = array::from_fn(|_| rng.draw_felt());
    //     let cpu_evals = CPUBackend::gen_eq_evals(&assignment);

    //     let avx_evals = AVX512Backend::gen_eq_evals(&assignment);

    //     assert_eq!(avx_evals.to_vec(), cpu_evals);
    // }

    // sum on g = eq(x1, ..., xn, z1, ..., zn) * p(0, x1, ..., xn) * p(1, x1, ..., xn)
    // sum on g = eq(x1, ..., xn, z1, ..., zn) * p(x1, ..., xn, 0) * p(1, x1, ..., xn)

    #[test]
    fn cpu_grand_product_works() -> Result<(), GkrError> {
        const N: usize = 1 << 8;
        let values = test_channel().draw_felts(N);
        let product = values.iter().product();
        println!("prod {product}");
        let top_layer = GrandProductTrace::<CPUBackend>::new(Mle::new(values));
        let now = Instant::now();
        let proof = prove(&mut test_channel(), top_layer.clone());
        println!("CPU took: {:?}", now.elapsed());

        let GkrVerificationArtifact {
            ood_point,
            claims_to_verify,
        } = partially_verify::<GrandProductCircuit>(&proof, &mut test_channel())?;

        assert_eq!(proof.output_claims, &[product]);
        assert_eq!(claims_to_verify, &[top_layer.eval_at_point(&ood_point)]);
        Ok(())
    }

    // #[test]
    // fn avx_grand_product_works() -> Result<(), GkrError> {
    //     const N: usize = 1 << 26;
    //     let values = test_channel().draw_felts(N);
    //     let product = values.iter().product();
    //     let top_layer =
    //         GrandProductTrace::new(AvxMle::<SecureField>::new(values.into_iter().collect()));
    //     let now = Instant::now();
    //     let proof = prove(&mut test_channel(), top_layer.clone().into());
    //     println!("AVX took: {:?}", now.elapsed());

    //     let GkrVerificationArtifact {
    //         ood_point,
    //         claims_to_verify,
    //     } = partially_verify(
    //         GkrCircuitInstance::GrandProduct,
    //         &proof,
    //         &mut test_channel(),
    //     )?;

    //     assert_eq!(proof.output_claims, &[product]);
    //     assert_eq!(claims_to_verify, &[top_layer.eval_at_point(&ood_point)]);
    //     Ok(())
    // }

    // #[test]
    // fn cpu_logup_works() -> Result<(), GkrError> {
    //     const N: usize = 1 << 22;
    //     let two = BaseField::from(2).into();
    //     let numerator_values = repeat(two).take(N).collect::<Vec<SecureField>>();
    //     let denominator_values = test_channel().draw_felts(N);
    //     let sum = zip(&numerator_values, &denominator_values)
    //         .map(|(&n, &d)| Fraction::new(n, d))
    //         .sum::<Fraction<SecureField>>();
    //     let numerators = CpuMle::<SecureField>::new(numerator_values.into_iter().collect());
    //     let denominators = CpuMle::<SecureField>::new(denominator_values.into_iter().collect());
    //     let top_layer = LogupTrace::Generic {
    //         numerators: numerators.clone(),
    //         denominators: denominators.clone(),
    //     };
    //     let proof = prove(&mut test_channel(), top_layer.into());

    //     let GkrVerificationArtifact {
    //         ood_point,
    //         claims_to_verify,
    //     } = partially_verify(GkrCircuitInstance::Logup, &proof, &mut test_channel())?;

    //     // TODO: `eva_claim` and `circuit_output_row` being an MleTrace (doesn't explain
    // structure)     // means there is a loss of context on the verification outputs (don't
    // know what order the     // numerator or denominator come in).
    //     assert_eq!(claims_to_verify.len(), 2);
    //     assert_eq!(claims_to_verify[0], numerators.eval_at_point(&ood_point));
    //     assert_eq!(claims_to_verify[1], denominators.eval_at_point(&ood_point));
    //     assert_eq!(proof.output_claims, &[sum.numerator, sum.denominator]);
    //     Ok(())
    // }

    // #[test]
    // fn avx_logup_works() -> Result<(), GkrError> {
    //     const N: usize = 1 << 26;
    //     let two = BaseField::from(2).into();
    //     let numerator_values = repeat(two).take(N).collect::<Vec<SecureField>>();
    //     let denominator_values = test_channel().draw_felts(N);
    //     let sum = zip(&numerator_values, &denominator_values)
    //         .map(|(&n, &d)| Fraction::new(n, d))
    //         .sum::<Fraction<SecureField>>();
    //     let numerators = AvxMle::<SecureField>::new(numerator_values.into_iter().collect());
    //     let denominators = AvxMle::<SecureField>::new(denominator_values.into_iter().collect());
    //     let top_layer = LogupTrace::Generic {
    //         numerators: numerators.clone(),
    //         denominators: denominators.clone(),
    //     };
    //     let proof = prove(&mut test_channel(), top_layer.into());

    //     let GkrVerificationArtifact {
    //         ood_point,
    //         claims_to_verify,
    //     } = partially_verify(GkrCircuitInstance::Logup, &proof, &mut test_channel())?;

    //     // TODO: `eva_claim` and `circuit_output_row` being an MleTrace, an nondescriptive type
    //     // means there is a loss of context on the verification outputs (don't know what the
    //     // numerator or denominator is).
    //     assert_eq!(claims_to_verify.len(), 2);
    //     assert_eq!(claims_to_verify[0], numerators.eval_at_point(&ood_point));
    //     assert_eq!(claims_to_verify[1], denominators.eval_at_point(&ood_point));
    //     assert_eq!(proof.output_claims, &[sum.numerator, sum.denominator]);
    //     Ok(())
    // }

    // #[test]
    // fn avx_logup_singles_works() -> Result<(), GkrError> {
    //     const N: usize = 1 << 26;
    //     let denominator_values = test_channel().draw_felts(N);
    //     let sum = denominator_values
    //         .iter()
    //         .map(|&d| Fraction::new(SecureField::one(), d))
    //         .sum::<Fraction<SecureField>>();
    //     let denominators = AvxMle::<SecureField>::new(denominator_values.into_iter().collect());
    //     let top_layer = LogupTrace::Singles {
    //         denominators: denominators.clone(),
    //     };
    //     let proof = prove(&mut test_channel(), top_layer.into());

    //     let GkrVerificationArtifact {
    //         ood_point,
    //         claims_to_verify,
    //     } = partially_verify(GkrCircuitInstance::Logup, &proof, &mut test_channel())?;

    //     // TODO: `eva_claim` and `circuit_output_row` being an MleTrace, an nondescriptive type
    //     // means there is a loss of context on the verification outputs (don't know what the
    //     // numerator or denominator is).
    //     assert_eq!(claims_to_verify.len(), 2);
    //     assert_eq!(claims_to_verify[0], SecureField::one());
    //     assert_eq!(claims_to_verify[1], denominators.eval_at_point(&ood_point));
    //     assert_eq!(proof.output_claims, &[sum.numerator, sum.denominator]);
    //     Ok(())
    // }

    fn test_channel() -> Blake2sChannel {
        let seed = Blake2sHash::from(vec![0; 32]);
        Blake2sChannel::new(seed)
    }
}
