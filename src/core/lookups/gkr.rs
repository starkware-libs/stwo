//! GKR protocol implementation designed to prove and verify lookup arguments.
use std::iter::{successors, zip};
use std::ops::Deref;

use itertools::Itertools;
use thiserror::Error;

use super::mle::{Mle, MleOps, MleTrace};
use super::sumcheck::{SumcheckError, SumcheckOracle, SumcheckProof};
use super::utils::eq;
use crate::core::backend::{CPUBackend, Col, Column, ColumnOps};
use crate::core::channel::Channel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::sumcheck;
use crate::core::lookups::utils::horner_eval;

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
    // TODO: Remove visibility.
    pub y: Vec<SecureField>,
    // TODO: Remove visibility.
    pub evals: Mle<B, SecureField>,
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

    fn num_variables(&self) -> usize;

    /// Produces the next GKR layer from the current layer.
    ///
    /// Returns [`None`] if the current layer is the output layer.
    fn next(&self) -> Option<Self>;

    /// Transforms layer `l+1` (current layer) into a sumcheck oracle for layer `l` (next layer).
    // TODO(andrew): Document `n_unused_variables`.
    fn into_sumcheck_oracle(
        self,
        lambda: SecureField,
        num_unused_variables: usize,
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
pub fn prove<L: GkrLayer>(channel: &mut impl Channel, top_layer: L) -> GkrProof {
    let mut layers = gen_layers(top_layer).into_iter().rev();

    let output_trace = layers.next().unwrap().into_trace();
    assert_eq!(output_trace.num_variables(), 0);
    let output_claims = output_trace.eval_at_point(&[]);

    let mut ood_point = vec![];
    let mut claims_to_verify = output_claims.clone();

    let layer_proofs = layers
        .map(|layer| {
            channel.mix_felts(&claims_to_verify);
            let lambda = channel.draw_felt();
            let eq_evals = EqEvals::new(&ood_point);
            let sumcheck_oracle = layer.into_sumcheck_oracle(lambda, 0, &eq_evals);
            let sumcheck_claim = horner_eval(&claims_to_verify, lambda);
            let (sumcheck_proof, sumcheck_ood_point, oracle, _) =
                sumcheck::prove(sumcheck_claim, sumcheck_oracle, channel);

            let input_mle_trace = oracle.into_inputs();
            let mask = GkrMask::try_from(input_mle_trace.clone()).unwrap();
            channel.mix_felts(mask.columns().flatten());

            let r_star = channel.draw_felt();
            claims_to_verify = input_mle_trace.eval_at_point(&[r_star]);
            ood_point = sumcheck_ood_point;
            ood_point.push(r_star);

            GkrLayerProof {
                sumcheck_proof,
                mask,
            }
        })
        .collect();

    GkrProof {
        layer_proofs,
        output_claims,
    }
}

// macro_rules! log {
//     ( $( $t:tt )* ) => {
//         web_sys::console::log_1(&format!( $( $t )* ).into());
//     }
// }

// pub struct Timer<'a> {
//     name: &'a str,
// }

// impl<'a> Timer<'a> {
//     pub fn new(name: &'a str) -> Timer<'a> {
//         web_sys::console::time_with_label(name);
//         Timer { name }
//     }
// }

// impl<'a> Drop for Timer<'a> {
//     fn drop(&mut self) {
//         web_sys::console::time_end_with_label(self.name);
//     }
// }

pub fn prove_batch<L: GkrLayer>(channel: &mut impl Channel, top_layers: Vec<L>) -> GkrBatchProof {
    let num_components = top_layers.len();
    let component_num_layers = top_layers.iter().map(|l| l.num_variables()).collect_vec();
    let num_layers = top_layers.iter().map(|l| l.num_variables()).max().unwrap();

    let mut components_layers = top_layers
        .into_iter()
        .map(|top_layer| gen_layers(top_layer).into_iter().rev().peekable())
        .collect::<Vec<_>>();

    let mut components_output_claims = vec![None; num_components];
    let mut components_layer_masks = (0..num_components).map(|_| Vec::new()).collect_vec();
    let mut sumcheck_proofs = Vec::new();

    let mut ood_point = Vec::new();
    let mut components_claims_to_verify = vec![None; num_components];

    for layer in 0..num_layers {
        // Check for output layers.
        for (component, layers) in components_layers.iter_mut().enumerate() {
            if component_num_layers[component] == num_layers - layer {
                let output_layer = layers.next().unwrap();
                assert_eq!(output_layer.num_variables(), 0);
                let output_claim = output_layer.into_trace().eval_at_point(&[]);
                components_output_claims[component] = Some(output_claim.clone());
                components_claims_to_verify[component] = Some(output_claim);
            }
        }

        // Seed channel with layer claims.
        for claims_to_verify in components_claims_to_verify.iter().flatten() {
            channel.mix_felts(claims_to_verify);
        }

        let eq_evals = EqEvals::new(&ood_point);
        let mut sumcheck_oracles = Vec::new();
        let mut sumcheck_claims = Vec::new();
        let mut sumcheck_components = Vec::new();

        // Create sumcheck oracles.
        for (component, claims_to_verify) in components_claims_to_verify.iter().enumerate() {
            if let Some(claims_to_verify) = claims_to_verify {
                let lambda = channel.draw_felt();
                let layer = components_layers[component].next().unwrap();
                let n_unused = num_layers - component_num_layers[component];
                let sumcheck_oracle = layer.into_sumcheck_oracle(lambda, n_unused, &eq_evals);
                sumcheck_oracles.push(sumcheck_oracle);
                let doubling_factor = BaseField::from(1 << n_unused);
                sumcheck_claims.push(horner_eval(claims_to_verify, lambda) * doubling_factor);
                sumcheck_components.push(component);
            }
        }

        let lambda = channel.draw_felt();
        let (sumcheck_proof, sumcheck_ood_point, sumcheck_oracles, _) =
            sumcheck::prove_batch(sumcheck_claims, sumcheck_oracles, lambda, channel);

        sumcheck_proofs.push(sumcheck_proof);

        let component_inputs = sumcheck_oracles
            .into_iter()
            .map(GkrSumcheckOracle::into_inputs)
            .collect_vec();

        // Seed channel TODO...
        for (&component, input) in zip(&sumcheck_components, &component_inputs) {
            let input_mask = GkrMask::try_from(input.clone()).unwrap();
            channel.mix_felts(input_mask.columns().flatten());
            components_layer_masks[component].push(input_mask);
        }

        let r_star = channel.draw_felt();
        ood_point = sumcheck_ood_point;
        ood_point.push(r_star);

        // Update the claims to verify.
        for (component, input) in zip(sumcheck_components, component_inputs) {
            components_claims_to_verify[component] = Some(input.eval_at_point(&[r_star]));
        }
    }

    let components_output_claims = components_output_claims
        .into_iter()
        .map(Option::unwrap)
        .collect();

    GkrBatchProof {
        sumcheck_proofs,
        components_layer_masks,
        components_output_claims,
    }
}

pub struct GkrBatchProof {
    pub sumcheck_proofs: Vec<SumcheckProof>,
    pub components_layer_masks: Vec<Vec<GkrMask>>,
    pub components_output_claims: Vec<Vec<SecureField>>,
}

/// Generates all GKR layers from the top layer.
fn gen_layers<L: GkrLayer>(top_layer: L) -> Vec<L> {
    successors(Some(top_layer), |layer| layer.next()).collect()
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
            mask: input_mask,
        } = layer_proof;

        channel.mix_felts(&claims_to_verify);
        let lambda = channel.draw_felt();
        let sumcheck_claim = horner_eval(&claims_to_verify, lambda);
        let (sumcheck_ood_point, sumcheck_eval) =
            sumcheck::partially_verify(sumcheck_claim, sumcheck_proof, channel)
                .map_err(|source| GkrError::InvalidSumcheck { layer, source })?;

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

/// Partially verifies a batch GKR proof.
///
/// On successful verification the function returns a [`GkrBatchVerificationArtifact`] which stores
/// the out-of-domain point and claimed evaluations in the top layer columns for each component at
/// the OOD point. These evaluations are not checked by this function - hence partial verification.
pub fn partially_verify_batch<C: BinaryTreeCircuit>(
    proof: &GkrBatchProof,
    channel: &mut impl Channel,
) -> Result<GkrBatchVerificationArtifact, GkrError> {
    let GkrBatchProof {
        sumcheck_proofs,
        components_layer_masks,
        components_output_claims,
    } = proof;

    let num_components = components_layer_masks.len();
    let component_num_layers = |component: usize| components_layer_masks[component].len();
    let num_layers = (0..num_components).map(component_num_layers).max().unwrap();

    // TODO: These should be errors.
    assert_eq!(components_output_claims.len(), components_layer_masks.len());
    assert_eq!(sumcheck_proofs.len(), num_layers);

    let mut ood_point = vec![];
    let mut components_claims_to_verify = vec![None; num_components];

    for (layer, sumcheck_proof) in sumcheck_proofs.iter().enumerate() {
        let num_remaining_layers = num_layers - layer;

        // Check for output layers.
        for component in 0..num_components {
            if component_num_layers(component) == num_remaining_layers {
                let output_claim = components_output_claims[component].clone();
                components_claims_to_verify[component] = Some(output_claim);
            }
        }

        // Seed channel with layer claims.
        for claims_to_verify in components_claims_to_verify.iter().flatten() {
            channel.mix_felts(claims_to_verify);
        }

        let mut sumcheck_lambdas = Vec::new();
        let mut sumcheck_claims = Vec::new();
        let mut sumcheck_components = Vec::new();

        // Prepare sumcheck claim.
        for (component, claims_to_verify) in components_claims_to_verify.iter().enumerate() {
            if let Some(claims_to_verify) = claims_to_verify {
                let lambda = channel.draw_felt();
                sumcheck_lambdas.push(lambda);
                let n_unused = num_layers - component_num_layers(component);
                let doubling_factor = BaseField::from(1 << n_unused);
                sumcheck_claims.push(horner_eval(claims_to_verify, lambda) * doubling_factor);
                sumcheck_components.push(component);
            }
        }

        let lambda = channel.draw_felt();
        let sumcheck_claim = horner_eval(&sumcheck_claims, lambda);
        let (sumcheck_ood_point, sumcheck_eval) =
            sumcheck::partially_verify(sumcheck_claim, sumcheck_proof, channel)
                .map_err(|source| GkrError::InvalidSumcheck { layer, source })?;

        let mut layer_evals = Vec::new();

        // Evaluate circuit at sumcheck OOD point.
        // TODO: Name challenges better. Shadow variable lambda confusing.
        for (&component, lambda) in zip(&sumcheck_components, sumcheck_lambdas) {
            let n_unused = num_layers - component_num_layers(component);
            let mask = &components_layer_masks[component][layer - n_unused];
            let [input_row_0, input_row_1] = mask.to_rows();
            let circuit_output = C::eval(&input_row_0, &input_row_1);
            let folded_output = horner_eval(&circuit_output, lambda);
            let eq_eval = eq(&ood_point[n_unused..], &sumcheck_ood_point[n_unused..]);
            layer_evals.push(eq_eval * folded_output);
        }

        let layer_eval = horner_eval(&layer_evals, lambda);

        if sumcheck_eval != layer_eval {
            return Err(GkrError::CircuitCheckFailure {
                claim: sumcheck_eval,
                output: layer_eval,
                layer,
            });
        }

        // Seed channel with masks.
        for &component in &sumcheck_components {
            let n_unused = num_layers - component_num_layers(component);
            let mask = &components_layer_masks[component][layer - n_unused];
            channel.mix_felts(mask.columns().flatten());
        }

        // Set evaluation point for next layer.
        let r_star = channel.draw_felt();
        ood_point = sumcheck_ood_point;
        ood_point.push(r_star);

        // Update the claims to verify.
        for component in sumcheck_components {
            let n_unused = num_layers - component_num_layers(component);
            let mask = &components_layer_masks[component][layer - n_unused];
            components_claims_to_verify[component] =
                Some(mask.to_mle_trace().eval_at_point(&[r_star]));
        }
    }

    // TODO: Double check unwrap fail is unreachable.
    let components_claims_to_verify = components_claims_to_verify
        .into_iter()
        .map(Option::unwrap)
        .collect();

    Ok(GkrBatchVerificationArtifact {
        ood_point,
        components_claims_to_verify,
    })
}

/// GKR partial verification artifact.
pub struct GkrBatchVerificationArtifact {
    /// Out-of-domain (OOD) point for columns in the top layer.
    pub ood_point: Vec<SecureField>,
    /// The claimed evaluation at `ood_point` for each column in the top layer of each component.
    pub components_claims_to_verify: Vec<Vec<SecureField>>,
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
pub struct GkrMask {
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
pub struct InvalidNumRowsError {
    num_rows: usize,
}

// TODO(Andrew): Remove generic on proof structs. Consider using `Vec` instead of `MleTrace`.
#[derive(Debug, Clone)]
struct GkrLayerProof {
    sumcheck_proof: SumcheckProof,
    mask: GkrMask,
}

#[cfg(test)]
mod tests {
    use super::{partially_verify, prove, GkrError};
    use crate::commitment_scheme::blake2_hash::Blake2sHash;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::CPUBackend;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::gkr::{
        partially_verify_batch, prove_batch, GkrBatchVerificationArtifact, GkrVerificationArtifact,
    };
    use crate::core::lookups::mle::Mle;
    use crate::core::lookups::{GrandProductCircuit, GrandProductTrace};

    type CpuGrandProductTrace = GrandProductTrace<CPUBackend>;

    #[test]
    fn cpu_grand_product_works() -> Result<(), GkrError> {
        const N: usize = 1 << 8;
        let values = test_channel().draw_felts(N);
        let product = values.iter().product();
        let top_layer = GrandProductTrace::<CPUBackend>::new(Mle::new(values));
        let proof = prove(&mut test_channel(), top_layer.clone());

        let GkrVerificationArtifact {
            ood_point,
            claims_to_verify,
        } = partially_verify::<GrandProductCircuit>(&proof, &mut test_channel())?;

        assert_eq!(proof.output_claims, &[product]);
        assert_eq!(claims_to_verify, &[top_layer.eval_at_point(&ood_point)]);
        Ok(())
    }

    #[test]
    fn cpu_prove_batch_works() -> Result<(), GkrError> {
        const N: usize = 1 << 10;
        let mut channel = test_channel();
        let col0 = GrandProductTrace::<CPUBackend>::new(Mle::new(channel.draw_felts(N)));
        let col1 = GrandProductTrace::<CPUBackend>::new(Mle::new(channel.draw_felts(N)));
        let product0 = col0.iter().product::<SecureField>();
        let product1 = col1.iter().product::<SecureField>();
        let top_layers = vec![col0.clone(), col1.clone()];
        let proof = prove_batch(&mut test_channel(), top_layers);

        let GkrBatchVerificationArtifact {
            ood_point,
            components_claims_to_verify,
        } = partially_verify_batch::<GrandProductCircuit>(&proof, &mut test_channel())?;

        assert_eq!(proof.components_output_claims[0], &[product0]);
        assert_eq!(proof.components_output_claims[1], &[product1]);
        let claim0 = &components_claims_to_verify[0];
        let claim1 = &components_claims_to_verify[1];
        assert_eq!(claim0, &[col0.eval_at_point(&ood_point)]);
        assert_eq!(claim1, &[col1.eval_at_point(&ood_point)]);
        Ok(())
    }

    #[test]
    fn cpu_prove_batch_works2() -> Result<(), GkrError> {
        const N: usize = 1 << 10;
        let mut channel = test_channel();
        let col0 = GrandProductTrace::<CPUBackend>::new(Mle::new(channel.draw_felts(N)));
        let col1 = GrandProductTrace::<CPUBackend>::new(Mle::new(channel.draw_felts(N / 2)));
        let product0 = col0.iter().product::<SecureField>();
        let product1 = col1.iter().product::<SecureField>();
        let top_layers = vec![col0.clone(), col1.clone()];
        let proof = prove_batch(&mut test_channel(), top_layers);

        let GkrBatchVerificationArtifact {
            ood_point,
            components_claims_to_verify,
        } = partially_verify_batch::<GrandProductCircuit>(&proof, &mut test_channel())?;

        assert_eq!(proof.components_output_claims[0], &[product0]);
        assert_eq!(proof.components_output_claims[1], &[product1]);
        let claim0 = &components_claims_to_verify[0];
        let claim1 = &components_claims_to_verify[1];
        assert_eq!(claim0, &[col0.eval_at_point(&ood_point)]);
        assert_eq!(claim1, &[col1.eval_at_point(&ood_point[1..])]);
        Ok(())
    }

    #[test]
    fn prove_batch_with_different_sizes_works() -> Result<(), GkrError> {
        const LOG_N0: usize = 12;
        const LOG_N1: usize = 14;
        let mut channel = test_channel();
        let col0 = GrandProductTrace::<CPUBackend>::new(Mle::new(channel.draw_felts(1 << LOG_N0)));
        let col1 = GrandProductTrace::<CPUBackend>::new(Mle::new(channel.draw_felts(1 << LOG_N1)));
        let product0 = col0.iter().product::<SecureField>();
        let product1 = col1.iter().product::<SecureField>();
        let top_layers = vec![col0.clone(), col1.clone()];
        let proof = prove_batch(&mut test_channel(), top_layers);

        let GkrBatchVerificationArtifact {
            ood_point,
            components_claims_to_verify,
        } = partially_verify_batch::<GrandProductCircuit>(&proof, &mut test_channel())?;

        assert_eq!(proof.components_output_claims[0], &[product0]);
        assert_eq!(proof.components_output_claims[1], &[product1]);
        let claim0 = &components_claims_to_verify[0];
        let claim1 = &components_claims_to_verify[1];
        let n_vars = ood_point.len();
        assert_eq!(claim0, &[col0.eval_at_point(&ood_point[n_vars - LOG_N0..])]);
        assert_eq!(claim1, &[col1.eval_at_point(&ood_point[n_vars - LOG_N1..])]);
        Ok(())
    }

    #[test]
    fn simd_prove_batch_works() -> Result<(), GkrError> {
        const N: usize = 1 << 12;
        let mut channel = test_channel();
        let values0 = channel.draw_felts(N);
        let values1 = channel.draw_felts(N);
        let product0 = values0.iter().product::<SecureField>();
        let product1 = values1.iter().product::<SecureField>();
        let col0 = GrandProductTrace::<SimdBackend>::new(Mle::new(values0.into_iter().collect()));
        let col1 = GrandProductTrace::<SimdBackend>::new(Mle::new(values1.into_iter().collect()));
        let top_layers = vec![col0.clone(), col1.clone()];
        let proof = prove_batch(&mut test_channel(), top_layers);

        let GkrBatchVerificationArtifact {
            ood_point,
            components_claims_to_verify,
        } = partially_verify_batch::<GrandProductCircuit>(&proof, &mut test_channel())?;

        assert_eq!(proof.components_output_claims[0], &[product0]);
        assert_eq!(proof.components_output_claims[1], &[product1]);
        let claim_0 = &components_claims_to_verify[0];
        let claim_1 = &components_claims_to_verify[1];
        assert_eq!(claim_0, &[col0.eval_at_point(&ood_point)]);
        assert_eq!(claim_1, &[col1.eval_at_point(&ood_point)]);
        Ok(())
    }

    #[test]
    fn prove_batch_with_different_sizes_works2() -> Result<(), GkrError> {
        const LOG_N0: usize = 8;
        const LOG_N1: usize = 5;
        const LOG_N2: usize = 7;
        let mut channel = test_channel();
        let col0 = CpuGrandProductTrace::new(Mle::new(channel.draw_felts(1 << LOG_N0)));
        let col1 = CpuGrandProductTrace::new(Mle::new(channel.draw_felts(1 << LOG_N1)));
        let col2 = CpuGrandProductTrace::new(Mle::new(channel.draw_felts(1 << LOG_N2)));
        let product0 = col0.iter().product::<SecureField>();
        let product1 = col1.iter().product::<SecureField>();
        let product2 = col2.iter().product::<SecureField>();
        let top_layers = vec![col0.clone(), col1.clone(), col2.clone()];
        let proof = prove_batch(&mut test_channel(), top_layers);

        let GkrBatchVerificationArtifact {
            ood_point,
            components_claims_to_verify,
        } = partially_verify_batch::<GrandProductCircuit>(&proof, &mut test_channel())?;

        assert_eq!(proof.components_output_claims[0], &[product0]);
        assert_eq!(proof.components_output_claims[1], &[product1]);
        assert_eq!(proof.components_output_claims[2], &[product2]);
        let claim0 = &components_claims_to_verify[0];
        let claim1 = &components_claims_to_verify[1];
        let claim2 = &components_claims_to_verify[2];
        let dim = ood_point.len();
        assert_eq!(claim0, &[col0.eval_at_point(&ood_point[dim - LOG_N0..])]);
        assert_eq!(claim1, &[col1.eval_at_point(&ood_point[dim - LOG_N1..])]);
        assert_eq!(claim2, &[col2.eval_at_point(&ood_point[dim - LOG_N2..])]);
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
