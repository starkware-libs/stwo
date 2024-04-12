use std::iter::successors;
use std::time::Instant;

use num_traits::{One, Zero};
use thiserror::Error;

use super::grand_product::{
    GrandProductCircuit, GrandProductOps, GrandProductOracle, GrandProductTrace,
};
use super::logup::{LogupCircuit, LogupOps, LogupOracle, LogupTrace};
use super::mle::{ColumnOpsV2, MleOps, MleTrace};
use super::sumcheck::{self, SumcheckError, SumcheckOracle, SumcheckProof};
use super::utils::{eq, horner_eval, Polynomial};
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
pub struct GkrProof<B: ColumnOpsV2<SecureField>> {
    layer_proofs: Vec<GkrLayerProof<B>>,
    output_layer: MleTrace<B, SecureField>,
}

// TODO(Andrew): Remove generic on proof structs. Consider using `Vec` instead of `MleTrace`.
#[derive(Debug, Clone)]
struct GkrLayerProof<B: ColumnOpsV2<SecureField>> {
    sumcheck_proof: SumcheckProof,
    input_encoding: MleTrace<B, SecureField>,
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
// TODO: add a mix felts over multiple slices.
// `channel.mix_felt_chunks(output_layer.iter().map(|mle| &mle));`
pub fn prove<B: LogupOps + GrandProductOps>(
    channel: &mut impl Channel,
    top_layer: GkrTraceInstance<B>,
) -> GkrProof<B> {
    let layers = successors(Some(top_layer), |layer| layer.next()).collect::<Vec<_>>();
    let mut layers = layers.into_iter().rev();

    let now = Instant::now();
    let output_layer = layers.next().unwrap().into_trace();
    output_layer
        .iter()
        .for_each(|c| channel.mix_felts(&c.to_vec()));

    let mut claim_assignment = vec![];
    let mut claims_to_verify = layers.next().unwrap();

    let layer_proofs = layers
        .map(|layer| {
            let lambda = channel.draw_felt();
            let now = Instant::now();
            let eq_evals = B::gen_eq_evals(&layer_assignment[1..]);
            println!("gen eq took {:?}", now.elapsed());
            let sumcheck_oracle = layer.into_sumcheck_oracle(lambda, &layer_assignment, &eq_evals);
            let sumcheck_claim = horner_eval(&layer_evals, lambda);
            let (sumcheck_proof, sumcheck_assignment, oracle) =
                sumcheck::prove(sumcheck_claim, sumcheck_oracle, channel);

            let input_encoding = oracle.into_inputs();
            input_encoding
                .iter()
                .for_each(|c| channel.mix_felts(&c.to_vec()));

            assert_eq!(input_encoding.num_variables(), 1);
            let r_star = channel.draw_felt();
            layer_assignment = sumcheck_assignment;
            layer_assignment.push(r_star);

            layer_evals = input_encoding.eval_at_point(&[r_star]);

            GkrLayerProof {
                sumcheck_proof,
                input_encoding,
            }
        })
        .collect();
    println!("proof gen time: {:?}", now.elapsed());

    GkrProof {
        layer_proofs,
        output_layer,
    }
}

/// Partially verifies a GKR proof.
///
/// On successful verification the function Returns a [`GkrVerificationArtifact`] which stores the
/// variable assignment and claimed evaluations in the top layer's columns. These claimed
/// evaluations are not validated by this function - hence partial verification.
pub fn partially_verify<B: MleOps<SecureField>>(
    circuit: GkrCircuitInstance,
    proof: &GkrProof<B>,
    channel: &mut impl Channel,
) -> Result<GkrVerificationArtifact, GkrError> {
    let zero = SecureField::zero();
    let one = SecureField::one();

    let GkrProof {
        output_layer,
        layer_proofs,
    } = proof;

    if output_layer.num_variables() != 1 {
        todo!("Return error.")
    }

    output_layer
        .iter()
        .for_each(|c| channel.mix_felts(&c.to_vec()));

    let mut layer_assignment = channel.draw_felts(output_layer.num_variables());
    let mut layer_claim = output_layer.eval_at_point(&layer_assignment);

    for (layer, layer_proof) in layer_proofs.iter().enumerate() {
        let GkrLayerProof {
            sumcheck_proof,
            input_encoding,
        } = layer_proof;

        let lambda = channel.draw_felt();
        let sumcheck_claim = horner_eval(&layer_claim, lambda);
        let (sumcheck_assignment, sumcheck_eval) =
            sumcheck::partially_verify(sumcheck_claim, sumcheck_proof, channel)
                .map_err(|source| GkrError::InvalidSumcheck { layer, source })?;

        if input_encoding.num_variables() != 1 {
            todo!("Return error.")
        }

        // TODO: Not need to eval. Just first row (0) and second row (1)
        let input0 = input_encoding.eval_at_point(&[zero]);
        let input1 = input_encoding.eval_at_point(&[one]);
        let circuit_output = circuit.eval(&input0, &input1);
        let folded_output = horner_eval(&circuit_output, lambda);
        let layer_eval = eq(&layer_assignment, &sumcheck_assignment) * folded_output;

        if sumcheck_eval != layer_eval {
            return Err(GkrError::CircuitCheckFailure {
                claim: sumcheck_eval,
                output: layer_eval,
                layer,
            });
        }

        input_encoding
            .iter()
            .for_each(|c| channel.mix_felts(&c.to_vec()));

        let r_star = channel.draw_felt();
        layer_assignment = sumcheck_assignment;
        layer_assignment.push(r_star);

        layer_claim = input_encoding.eval_at_point(&[r_star]);
    }

    let row0 = output_layer.eval_at_point(&[zero]);
    let row1 = output_layer.eval_at_point(&[one]);
    let circuit_output_row = circuit.eval(&row0, &row1);

    // TODO: Consider naming "eval_point" "layer_assignment" more similar to be consistent. Same
    // with "circuit_output_claim".
    Ok(GkrVerificationArtifact {
        eval_point: layer_assignment,
        eval_claim: layer_claim,
        circuit_output_row,
    })
}

// <https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf> (page 65)
// TODO: add a mix felts over multiple slices.
// `channel.mix_felt_chunks(output_layer.iter().map(|mle| &mle));`
pub fn _prove_batch<B: LogupOps + GrandProductOps>(
    channel: &mut impl Channel,
    top_layers: Vec<GkrTraceInstance<B>>,
) -> GkrProof<B> {
    let num_instances = top_layers.len();
    let mut instance_layers = top_layers
        .into_iter()
        .map(|top_layer| successors(Some(top_layer), |layer| layer.next()).collect())
        .collect::<Vec<Vec<GkrTraceInstance<B>>>>();

    let mut instance_output_layers = vec![None; num_instances];

    let num_layers = instance_layers.iter().map(|l| l.len()).max().unwrap();

    for layer in (0..num_layers).rev() {
        for instance in (0..num_instances) {
            let layer = &instance_layers[instance][layer];
            match layer.num_variables() {
                1 => instance_output_layers[instance] = Some(),
                // Don't need the output values
                0 => {}
            }
        }
    }

    for layers in &mut all_instance_layers {
        _ = layers.pop();
    }

    let now = Instant::now();
    let output_layer = layers.next().unwrap().into_trace();
    output_layer
        .iter()
        .for_each(|c| channel.mix_felts(&c.to_vec()));

    let mut layer_assignment = channel.draw_felts(output_layer.num_variables());
    let mut layer_evals = output_layer.eval_at_point(&layer_assignment);

    let layer_proofs = layers
        .map(|layer| {
            let lambda = channel.draw_felt();
            let now = Instant::now();
            let eq_evals = B::gen_eq_evals(&layer_assignment[1..]);
            println!("gen eq took {:?}", now.elapsed());
            let sumcheck_oracle = layer.into_sumcheck_oracle(lambda, &layer_assignment, &eq_evals);
            let sumcheck_claim = horner_eval(&layer_evals, lambda);
            let (sumcheck_proof, sumcheck_assignment, oracle) =
                sumcheck::prove(sumcheck_claim, sumcheck_oracle, channel);

            let input_encoding = oracle.into_inputs();
            input_encoding
                .iter()
                .for_each(|c| channel.mix_felts(&c.to_vec()));

            assert_eq!(input_encoding.num_variables(), 1);
            let r_star = channel.draw_felt();
            layer_assignment = sumcheck_assignment;
            layer_assignment.push(r_star);

            layer_evals = input_encoding.eval_at_point(&[r_star]);

            GkrLayerProof {
                sumcheck_proof,
                input_encoding,
            }
        })
        .collect();
    println!("proof gen time: {:?}", now.elapsed());

    GkrProof {
        layer_proofs,
        output_layer,
    }
}

struct GkrBatchProof {
    sumcheck_proofs: Vec<SumcheckProof>,
    layer_masks: Vec<Vec<GkrMask>>,
}

struct GkrMask {
    columns: Vec<[SecureField; 2]>,
}

/// GKR partial verification artifact.
pub struct GkrVerificationArtifact {
    /// Variable assignment for columns in the top layer.
    pub eval_point: Vec<SecureField>,
    /// The claimed evaluation at `variable_assignment` for each column in the top layer.
    pub eval_claim: Vec<SecureField>,
    pub circuit_output_row: Vec<SecureField>,
}

// pub struct GkrLayerVerificationArtifact {
//     /// Input variable assignment for columns in the layer.
//     pub variable_assignment: Vec<SecureField>,
//     /// The claimed evaluation at `variable_assignment` for each column in the layer.
//     pub claimed_evals: Vec<SecureField>,
// }

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
        const LOG_SIZE: usize = 6;
        let mut rng = test_channel();
        let assignment: [SecureField; LOG_SIZE] = array::from_fn(|_| rng.draw_felt());
        let cpu_evals = CPUBackend::gen_eq_evals(&assignment);

        let avx_evals = AVX512Backend::gen_eq_evals(&assignment);

        assert_eq!(avx_evals.to_vec(), cpu_evals);
    }

    #[test]
    fn cpu_grand_product_works() -> Result<(), GkrError> {
        const N: usize = 1 << 3;
        let values = test_channel().draw_felts(N);
        let product = values.iter().product();
        let top_layer =
            GrandProductTrace::new(CpuMle::<SecureField>::new(values.into_iter().collect()));
        let now = Instant::now();
        let proof = prove(&mut test_channel(), top_layer.clone().into());
        println!("CPU took: {:?}", now.elapsed());

        let GkrVerificationArtifact {
            eval_point,
            eval_claim,
            circuit_output_row,
        } = partially_verify(
            GkrCircuitInstance::GrandProduct,
            &proof,
            &mut test_channel(),
        )?;

        assert_eq!(circuit_output_row, &[product]);
        assert_eq!(eval_claim, &[top_layer.eval_at_point(&eval_point)]);
        Ok(())
    }

    #[test]
    fn avx_grand_product_works() -> Result<(), GkrError> {
        const N: usize = 1 << 3;
        let values = test_channel().draw_felts(N);
        let product = values.iter().product();
        let top_layer =
            GrandProductTrace::new(AvxMle::<SecureField>::new(values.into_iter().collect()));
        let now = Instant::now();
        let proof = prove(&mut test_channel(), top_layer.clone().into());
        println!("AVX took: {:?}", now.elapsed());

        let GkrVerificationArtifact {
            eval_point,
            eval_claim,
            circuit_output_row,
        } = partially_verify(
            GkrCircuitInstance::GrandProduct,
            &proof,
            &mut test_channel(),
        )?;

        assert_eq!(circuit_output_row, &[product]);
        assert_eq!(eval_claim, &[top_layer.eval_at_point(&eval_point)]);
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
            eval_point,
            eval_claim,
            circuit_output_row,
        } = partially_verify(GkrCircuitInstance::Logup, &proof, &mut test_channel())?;

        // TODO: `eva_claim` and `circuit_output_row` being an MleTrace, an nondescriptive type
        // means there is a loss of context on the verification outputs (don't know what the
        // numerator or denominator is).
        assert_eq!(eval_claim.len(), 2);
        assert_eq!(eval_claim[0], numerators.eval_at_point(&eval_point));
        assert_eq!(eval_claim[1], denominators.eval_at_point(&eval_point));
        assert_eq!(circuit_output_row, &[sum.numerator, sum.denominator]);
        Ok(())
    }

    #[test]
    fn avx_logup_works() -> Result<(), GkrError> {
        const N: usize = 1 << 20;
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
            eval_point,
            eval_claim,
            circuit_output_row,
        } = partially_verify(GkrCircuitInstance::Logup, &proof, &mut test_channel())?;

        // TODO: `eva_claim` and `circuit_output_row` being an MleTrace, an nondescriptive type
        // means there is a loss of context on the verification outputs (don't know what the
        // numerator or denominator is).
        assert_eq!(eval_claim.len(), 2);
        assert_eq!(eval_claim[0], numerators.eval_at_point(&eval_point));
        assert_eq!(eval_claim[1], denominators.eval_at_point(&eval_point));
        assert_eq!(circuit_output_row, &[sum.numerator, sum.denominator]);
        Ok(())
    }

    fn test_channel() -> Blake2sChannel {
        let seed = Blake2sHash::from(vec![0; 32]);
        Blake2sChannel::new(seed)
    }
}
