use std::iter::successors;
use std::time::Instant;

use itertools::Itertools;
use num_traits::Zero;
use thiserror::Error;

use super::grand_product::{
    GrandProductCircuit, GrandProductOps, GrandProductOracle, GrandProductTrace,
};
use super::logup::{LogupCircuit, LogupOps, LogupOracle, LogupTrace};
use super::mle::{ColumnOpsV2, Mle, MleOps, MleTrace};
use super::sumcheck::{self, SumcheckError, SumcheckOracle, SumcheckProof};
use super::utils::{eq, horner_eval, Polynomial};
use crate::core::backend::CPUBackend;
use crate::core::channel::Channel;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::mle::ColumnV2;

pub trait GkrOps: ColumnOpsV2<SecureField> {
    type EqEvals: ToOwned + 'static;

    /// Returns the evaluations of [`eq(x, y)`] for all values of `x` in `{0, 1}^n`.
    ///
    /// `y` has length `n`.
    fn gen_eq_evals(y: &[SecureField]) -> Self::EqEvals;
}

pub trait GkrLayer: Sized {
    type Backend: GkrOps;
    type SumcheckOracle<'a>: GkrSumcheckOracle<Backend = Self::Backend>;

    /// Generates the next GKR layer from the current one.
    ///
    /// Returns [`None`] if the current layer is the output layer.
    fn next(&self) -> Option<Self>;

    /// Create an oracle for sumcheck
    fn into_sumcheck_oracle<'a>(
        self,
        lambda: SecureField,
        layer_assignment: &[SecureField],
        eq_evals: &'a <Self::Backend as GkrOps>::EqEvals,
    ) -> Self::SumcheckOracle<'a>;

    /// Returns this layer as a [`MleTrace`].
    ///
    /// Note that this function is only called by [`prove()`] on the GKR output layer. The output
    /// layer needs to be stored in the proof. To prevent making proofs generic, all output layers
    /// are stored in the proof as a [`MleTrace`].
    // TODO: Also note it's used for the output of each layer.
    fn into_trace(self) -> MleTrace<Self::Backend, SecureField>;
}

// TODO: Merge `GkrSumcheckOracle` and `SumcheckOracle`
pub trait GkrSumcheckOracle: SumcheckOracle {
    type Backend: GkrOps;

    /// Returns the multi-linear extensions in layer `l+1` that define `g` in layer `l`.
    ///
    /// Let `g(x_1, ..., x_n)` represent the multivariate polynomial that the sum-check protocol is
    /// applied to in GKR layer `l`. In GKR `g` is defined by the values in layer `l+1`. In the
    /// final round of sumcheck the verifier is left with a claimed evaluation of `g` at a random
    /// point `p`. The prover sends the values in layer `l+1` so the verifier is able evaluate `g`.
    /// This function returns these values in layer l+1 that the verifier needs.
    // TODO: Document better. Also two rows are always required to be returned for the "binary tree"
    // structured proofs that `prove` and `partially_verify` work with so returning a MleTrace is
    // overkill/misleading. This function should enforce the binary tree structure and only allow
    // returning two rows.
    fn into_inputs(self) -> MleTrace<Self::Backend, SecureField> {
        todo!()
    }
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

// TODO(Andrew): Remove generic on proof structs. Consider using `Vec` instead of `MleTrace`.
#[derive(Debug, Clone)]
pub struct GkrProof {
    layer_proofs: Vec<GkrLayerProof>,
    output_claims: Vec<SecureField>,
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

    fn to_mle_trace(&self) -> MleTrace<CPUBackend, SecureField> {
        let columns = self
            .columns
            .iter()
            .map(|&column| Mle::new(column.into_iter().collect()))
            .collect_vec();
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

pub enum GkrTraceInstance<B: LogupOps + GrandProductOps> {
    Logup(LogupTrace<B>),
    GrandProduct(GrandProductTrace<B>),
}

impl<B: LogupOps + GrandProductOps> From<LogupTrace<B>> for GkrTraceInstance<B> {
    fn from(trace: LogupTrace<B>) -> Self {
        Self::Logup(trace)
    }
}

impl<B: LogupOps + GrandProductOps> From<GrandProductTrace<B>> for GkrTraceInstance<B> {
    fn from(trace: GrandProductTrace<B>) -> Self {
        Self::GrandProduct(trace)
    }
}

impl<B: LogupOps + GrandProductOps> GkrLayer for GkrTraceInstance<B> {
    type Backend = B;
    type SumcheckOracle<'a> = GkrOracleInstance<'a, B>;

    fn next(&self) -> Option<Self> {
        Some(match self {
            Self::Logup(trace) => Self::Logup(trace.next()?),
            Self::GrandProduct(trace) => Self::GrandProduct(trace.next()?),
        })
    }

    fn into_sumcheck_oracle<'a>(
        self,
        lambda: SecureField,
        layer_assignment: &[SecureField],
        eq_evals: &'a B::EqEvals,
    ) -> GkrOracleInstance<'a, B> {
        use GkrOracleInstance::*;
        match self {
            Self::Logup(t) => Logup(t.into_sumcheck_oracle(lambda, layer_assignment, eq_evals)),
            Self::GrandProduct(t) => {
                GrandProduct(t.into_sumcheck_oracle(lambda, layer_assignment, eq_evals))
            }
        }
    }

    fn into_trace(self) -> MleTrace<B, SecureField> {
        match self {
            Self::Logup(t) => t.into_trace(),
            Self::GrandProduct(t) => t.into_trace(),
        }
    }
}

pub enum GkrOracleInstance<'a, B: LogupOps + GrandProductOps> {
    Logup(LogupOracle<'a, B>),
    GrandProduct(GrandProductOracle<'a, B>),
}

impl<'a, B: LogupOps + GrandProductOps> SumcheckOracle for GkrOracleInstance<'a, B> {
    fn num_variables(&self) -> usize {
        // TODO: could have a map!(...) macro to help squash all these matches.
        match self {
            Self::Logup(oracle) => oracle.num_variables(),
            Self::GrandProduct(oracle) => oracle.num_variables(),
        }
    }

    fn univariate_sum(&self, claim: SecureField) -> Polynomial<SecureField> {
        match self {
            Self::Logup(oracle) => oracle.univariate_sum(claim),
            Self::GrandProduct(oracle) => oracle.univariate_sum(claim),
        }
    }

    fn fix_first(self, challenge: SecureField) -> Self {
        match self {
            Self::Logup(oracle) => Self::Logup(oracle.fix_first(challenge)),
            Self::GrandProduct(oracle) => Self::GrandProduct(oracle.fix_first(challenge)),
        }
    }
}

impl<'a, B: LogupOps + GrandProductOps> GkrSumcheckOracle for GkrOracleInstance<'a, B> {
    type Backend = B;

    fn into_inputs(self) -> MleTrace<B, SecureField> {
        match self {
            Self::Logup(oracle) => oracle.into_inputs(),
            Self::GrandProduct(oracle) => oracle.into_inputs(),
        }
    }
}

pub enum GkrCircuitInstance {
    Logup,
    GrandProduct,
}

impl BinaryTreeCircuit for GkrCircuitInstance {
    fn eval(&self, even_row: &[SecureField], odd_row: &[SecureField]) -> Vec<SecureField> {
        match self {
            Self::Logup => LogupCircuit.eval(even_row, odd_row),
            Self::GrandProduct => GrandProductCircuit.eval(even_row, odd_row),
        }
    }
}

// <https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf> (page 65)
pub fn prove<B: LogupOps + GrandProductOps>(
    channel: &mut impl Channel,
    top_layer: GkrTraceInstance<B>,
) -> GkrProof {
    let layers = successors(Some(top_layer), |layer| layer.next()).collect::<Vec<_>>();
    let mut layers = layers.into_iter().rev();

    let output_trace = layers.next().unwrap().into_trace();
    assert_eq!(output_trace.num_variables(), 0);
    let output_claims = output_trace.eval_at_point(&[]);

    // TODO: First layer OOD point should be empty not zero.
    let mut ood_point = vec![SecureField::zero()];
    let mut claims_to_verify = output_claims.clone();

    let now = Instant::now();
    let layer_proofs = layers
        .map(|layer| {
            channel.mix_felts(&claims_to_verify);
            let lambda = channel.draw_felt();
            let eq_evals = B::gen_eq_evals(&ood_point[1..]);
            let sumcheck_oracle = layer.into_sumcheck_oracle(lambda, &ood_point, &eq_evals);
            let sumcheck_claim = horner_eval(&claims_to_verify, lambda);
            let (sumcheck_proof, sumcheck_ood_point, oracle, sumcheck_eval) =
                sumcheck::prove(sumcheck_claim, sumcheck_oracle, channel);

            channel.mix_felts(&[sumcheck_eval]);
            let r_star = channel.draw_felt();
            ood_point = sumcheck_ood_point;
            ood_point.push(r_star);

            let input_mle_trace = oracle.into_inputs();
            claims_to_verify = input_mle_trace.eval_at_point(&[r_star]);
            let input_mask = input_mle_trace.try_into().unwrap();

            GkrLayerProof {
                sumcheck_proof,
                input_mask,
            }
        })
        .collect();
    println!("proof gen took: {:?}", now.elapsed());

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
pub fn partially_verify(
    circuit: GkrCircuitInstance,
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

        // Say the output of the logup circuit is p=1, q=2
        // The sumcheck claim for the next layer is

        channel.mix_felts(&claims_to_verify);
        let lambda = channel.draw_felt();
        let sumcheck_claim = horner_eval(&claims_to_verify, lambda);
        let (sumcheck_ood_point, sumcheck_eval) =
            sumcheck::partially_verify(sumcheck_claim, sumcheck_proof, channel)
                .map_err(|source| GkrError::InvalidSumcheck { layer, source })?;

        let [input_row_0, input_row_1] = input_mask.to_rows();
        let circuit_output = circuit.eval(&input_row_0, &input_row_1);
        let folded_output = horner_eval(&circuit_output, lambda);
        let layer_eval = eq(&ood_point, &sumcheck_ood_point) * folded_output;

        if sumcheck_eval != layer_eval {
            return Err(GkrError::CircuitCheckFailure {
                claim: sumcheck_eval,
                output: layer_eval,
                layer,
            });
        }

        // TODO: Is seeting the channel with the sumche_eval ok? Or does it need to be reseeded with
        // all inputs? i.e. `input_encoding.iter().for_each(|c| channel.mix_felts(&c.to_vec()));`
        channel.mix_felts(&[sumcheck_eval]);
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

/// GKR partial verification artifact.
pub struct GkrVerificationArtifact {
    /// Out-of-domain (OOD) point for columns in the top layer.
    pub ood_point: Vec<SecureField>,
    /// The claimed evaluation at `ood_point` for each column in the top layer.
    pub claims_to_verify: Vec<SecureField>,
}

/// Defines a circuit with a binary tree structure.
///
/// These circuits define how they operate locally on pairs of neighboring input rows to produce a
/// single output row. This local 2-to-1 constraint is what gives the whole circuit its "binary
/// tree" structure. Circuit examples: [`LogupCircuit`], [`GrandProductCircuit`].
///
/// A binary tree circuit has a highly "regular" wiring pattern. It fits the structure of the
/// circuits defined in [Thaler13] which allows for an efficient linear time (linear in size of the
/// circuit) implementation of a GKR prover.
///
/// [Thaler13]: https://eprint.iacr.org/2013/351.pdf
pub trait BinaryTreeCircuit {
    /// Returns the output row after applying the circuit to the provided neighboring input rows.
    fn eval(
        &self,
        // layer: usize,
        even_row: &[SecureField],
        odd_row: &[SecureField],
    ) -> Vec<SecureField>;
}

#[cfg(test)]
mod tests {
    use std::array;
    use std::iter::{repeat, zip};
    use std::time::Instant;

    use num_traits::One;

    use super::{partially_verify, prove, GkrCircuitInstance, GkrError};
    use crate::commitment_scheme::blake2_hash::Blake2sHash;
    use crate::core::backend::avx512::{AVX512Backend, AvxMle};
    use crate::core::backend::cpu::CpuMle;
    use crate::core::backend::CPUBackend;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::gkr::{GkrOps, GkrVerificationArtifact};
    use crate::core::lookups::grand_product::GrandProductTrace;
    use crate::core::lookups::logup::{Fraction, LogupTrace};
    use crate::core::lookups::mle::ColumnV2;

    #[test]
    fn avx_eq_evals_matches_cpu_eq_evals() {
        const LOG_SIZE: usize = 8;
        let mut rng = test_channel();
        let assignment: [SecureField; LOG_SIZE] = array::from_fn(|_| rng.draw_felt());
        let cpu_evals = CPUBackend::gen_eq_evals(&assignment);

        let avx_evals = AVX512Backend::gen_eq_evals(&assignment);

        assert_eq!(avx_evals.to_vec(), cpu_evals);
    }

    #[test]
    fn cpu_grand_product_works() -> Result<(), GkrError> {
        const N: usize = 1 << 7;
        let values = test_channel().draw_felts(N);
        let product = values.iter().product();
        let top_layer =
            GrandProductTrace::new(CpuMle::<SecureField>::new(values.into_iter().collect()));
        let now = Instant::now();
        let proof = prove(&mut test_channel(), top_layer.clone().into());
        println!("CPU took: {:?}", now.elapsed());

        let GkrVerificationArtifact {
            ood_point,
            claims_to_verify,
        } = partially_verify(
            GkrCircuitInstance::GrandProduct,
            &proof,
            &mut test_channel(),
        )?;

        assert_eq!(proof.output_claims, &[product]);
        assert_eq!(claims_to_verify, &[top_layer.eval_at_point(&ood_point)]);
        Ok(())
    }

    #[test]
    fn avx_grand_product_works() -> Result<(), GkrError> {
        const N: usize = 1 << 26;
        let values = test_channel().draw_felts(N);
        let product = values.iter().product();
        let top_layer =
            GrandProductTrace::new(AvxMle::<SecureField>::new(values.into_iter().collect()));
        let now = Instant::now();
        let proof = prove(&mut test_channel(), top_layer.clone().into());
        println!("AVX took: {:?}", now.elapsed());

        let GkrVerificationArtifact {
            ood_point,
            claims_to_verify,
        } = partially_verify(
            GkrCircuitInstance::GrandProduct,
            &proof,
            &mut test_channel(),
        )?;

        assert_eq!(proof.output_claims, &[product]);
        assert_eq!(claims_to_verify, &[top_layer.eval_at_point(&ood_point)]);
        Ok(())
    }

    #[test]
    fn cpu_logup_works() -> Result<(), GkrError> {
        const N: usize = 1 << 22;
        let two = BaseField::from(2).into();
        let numerator_values = repeat(two).take(N).collect::<Vec<SecureField>>();
        let denominator_values = test_channel().draw_felts(N);
        let sum = zip(&numerator_values, &denominator_values)
            .map(|(&n, &d)| Fraction::new(n, d))
            .sum::<Fraction<SecureField>>();
        let numerators = CpuMle::<SecureField>::new(numerator_values.into_iter().collect());
        let denominators = CpuMle::<SecureField>::new(denominator_values.into_iter().collect());
        let top_layer = LogupTrace::Generic {
            numerators: numerators.clone(),
            denominators: denominators.clone(),
        };
        let proof = prove(&mut test_channel(), top_layer.into());

        let GkrVerificationArtifact {
            ood_point,
            claims_to_verify,
        } = partially_verify(GkrCircuitInstance::Logup, &proof, &mut test_channel())?;

        // TODO: `eva_claim` and `circuit_output_row` being an MleTrace (doesn't explain structure)
        // means there is a loss of context on the verification outputs (don't know what order the
        // numerator or denominator come in).
        assert_eq!(claims_to_verify.len(), 2);
        assert_eq!(claims_to_verify[0], numerators.eval_at_point(&ood_point));
        assert_eq!(claims_to_verify[1], denominators.eval_at_point(&ood_point));
        assert_eq!(proof.output_claims, &[sum.numerator, sum.denominator]);
        Ok(())
    }

    #[test]
    fn avx_logup_works() -> Result<(), GkrError> {
        const N: usize = 1 << 26;
        let two = BaseField::from(2).into();
        let numerator_values = repeat(two).take(N).collect::<Vec<SecureField>>();
        let denominator_values = test_channel().draw_felts(N);
        let sum = zip(&numerator_values, &denominator_values)
            .map(|(&n, &d)| Fraction::new(n, d))
            .sum::<Fraction<SecureField>>();
        let numerators = AvxMle::<SecureField>::new(numerator_values.into_iter().collect());
        let denominators = AvxMle::<SecureField>::new(denominator_values.into_iter().collect());
        let top_layer = LogupTrace::Generic {
            numerators: numerators.clone(),
            denominators: denominators.clone(),
        };
        let proof = prove(&mut test_channel(), top_layer.into());

        let GkrVerificationArtifact {
            ood_point,
            claims_to_verify,
        } = partially_verify(GkrCircuitInstance::Logup, &proof, &mut test_channel())?;

        // TODO: `eva_claim` and `circuit_output_row` being an MleTrace, an nondescriptive type
        // means there is a loss of context on the verification outputs (don't know what the
        // numerator or denominator is).
        assert_eq!(claims_to_verify.len(), 2);
        assert_eq!(claims_to_verify[0], numerators.eval_at_point(&ood_point));
        assert_eq!(claims_to_verify[1], denominators.eval_at_point(&ood_point));
        assert_eq!(proof.output_claims, &[sum.numerator, sum.denominator]);
        Ok(())
    }

    #[test]
    fn avx_logup_singles_works() -> Result<(), GkrError> {
        const N: usize = 1 << 26;
        let denominator_values = test_channel().draw_felts(N);
        let sum = denominator_values
            .iter()
            .map(|&d| Fraction::new(SecureField::one(), d))
            .sum::<Fraction<SecureField>>();
        let denominators = AvxMle::<SecureField>::new(denominator_values.into_iter().collect());
        let top_layer = LogupTrace::Singles {
            denominators: denominators.clone(),
        };
        let proof = prove(&mut test_channel(), top_layer.into());

        let GkrVerificationArtifact {
            ood_point,
            claims_to_verify,
        } = partially_verify(GkrCircuitInstance::Logup, &proof, &mut test_channel())?;

        // TODO: `eva_claim` and `circuit_output_row` being an MleTrace, an nondescriptive type
        // means there is a loss of context on the verification outputs (don't know what the
        // numerator or denominator is).
        assert_eq!(claims_to_verify.len(), 2);
        assert_eq!(claims_to_verify[0], SecureField::one());
        assert_eq!(claims_to_verify[1], denominators.eval_at_point(&ood_point));
        assert_eq!(proof.output_claims, &[sum.numerator, sum.denominator]);
        Ok(())
    }

    fn test_channel() -> Blake2sChannel {
        let seed = Blake2sHash::from(vec![0; 32]);
        Blake2sChannel::new(seed)
    }
}
