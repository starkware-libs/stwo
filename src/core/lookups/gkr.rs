//! GKR protocol implementation designed to prove and verify lookup arguments.
use std::iter::{successors, zip};
use std::ops::Deref;

use itertools::Itertools;
use num_traits::{One, Zero};
use thiserror::Error;

use super::mle::{Mle, MleOps};
use super::sumcheck::{MultivariatePolyOracle, SumcheckError, SumcheckProof};
use super::utils::eq;
use crate::core::backend::{Col, Column, ColumnOps};
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

/// Stores the gate values in a layer of a binary tree structured GKR circuit.
pub trait GkrLayer: Sized {
    type Backend: GkrOps + MleOps<SecureField>;
    type MultivariatePolyOracle<'a>: GkrMultivariatePolyOracle<Backend = Self::Backend>;

    fn n_variables(&self) -> usize;

    /// Produces the next GKR layer from the current layer.
    ///
    /// The next layer should be strictly half the size of the current layer.
    fn next(&self) -> Layer<Self>;

    /// Transforms the layer into a multivariate polynomial oracle that can be used with
    /// [`sumcheck::prove_batch()`] to prove the relationship between layer `l+1` (this layer)
    /// and `l` (next lower layer).
    ///
    /// `n_unused_variables` indicates how many leading variables go unused in the multivariate
    /// polynomial. This facilitate batching sum-check across multiple GKR components as there is a
    /// requirement all multivariate polynomials have the same number of variables.
    ///
    /// [`sumcheck::prove_batch()`]: crate::core::lookups::sumcheck::prove_batch
    fn into_sumcheck_oracle(
        self,
        lambda: SecureField,
        n_unused_variables: usize,
        eq_evals: &EqEvals<Self::Backend>,
    ) -> Self::MultivariatePolyOracle<'_>;
}

pub enum Layer<L: GkrLayer> {
    Output(Vec<SecureField>),
    Internal(L),
}

impl<L: GkrLayer> Layer<L> {
    /// Produces the next layer from the current layer.
    ///
    /// Returns [`None`] if the current layer is the output layer.
    fn next(&self) -> Option<Self> {
        match self {
            Self::Internal(layer) => Some(layer.next()),
            Self::Output(_) => None,
        }
    }
}

pub trait GkrMultivariatePolyOracle: MultivariatePolyOracle {
    type Backend: GkrOps;

    // TODO: Docs.
    fn try_into_mask(self) -> Option<GkrMask> {
        todo!()
    }
}

/// Defines how a circuit operates locally on pairs of input rows to produce a single output row.
/// This local 2-to-1 constraint is what gives the whole circuit its "binary tree" structure.
///
/// Binary tree structured circuit have a highly regular wiring pattern that fit the structure of
/// the circuits defined in [Thaler13] which allow for efficient linear time (linear in size of the
/// circuit) GKR prover implementations.
///
/// [Thaler13]: https://eprint.iacr.org/2013/351.pdf
pub trait GkrBinaryCircuitUnit {
    /// Returns the output row after applying the circuit to the provided input rows.
    fn eval(&self, row0: &[SecureField], row1: &[SecureField]) -> Vec<SecureField>;
}

// TODO
/// GKR algorithm: <https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf> (page 65)
pub fn prove_batch<L: GkrLayer>(channel: &mut impl Channel, top_layers: Vec<L>) -> GkrBatchProof {
    let n_components = top_layers.len();
    let n_layers_by_component = top_layers.iter().map(|l| l.n_variables()).collect_vec();
    let n_layers = top_layers.iter().map(|l| l.n_variables()).max().unwrap();

    // Evaluate all component circuits and collect the layer values.
    let mut layers_by_component = top_layers
        .into_iter()
        .map(|top_layer| gen_layers(top_layer).into_iter().rev().peekable())
        .collect::<Vec<_>>();

    let mut output_claims_by_component = vec![None; n_components];
    let mut layer_masks_by_component = (0..n_components).map(|_| Vec::new()).collect_vec();
    let mut sumcheck_proofs = Vec::new();

    let mut ood_point = Vec::new();
    let mut claims_to_verify_by_component = vec![None; n_components];

    for layer in 0..n_layers {
        let n_remaining_layers = n_layers - layer;

        // Check all the components for output layers.
        for (component, layers) in layers_by_component.iter_mut().enumerate() {
            if n_layers_by_component[component] == n_remaining_layers {
                let output_layer = match layers.next().unwrap() {
                    Layer::Output(layer) => layer,
                    Layer::Internal(_) => unreachable!(),
                };

                output_claims_by_component[component] = Some(output_layer.clone());
                claims_to_verify_by_component[component] = Some(output_layer);
            }
        }

        // Seed the channel with layer claims.
        for claims_to_verify in claims_to_verify_by_component.iter().flatten() {
            channel.mix_felts(claims_to_verify);
        }

        let eq_evals = EqEvals::new(&ood_point);
        let mut sumcheck_oracles = Vec::new();
        let mut sumcheck_claims = Vec::new();
        let mut sumcheck_components = Vec::new();

        // Create the multivariate polynomial oracles used with sumcheck.
        for (component, claims_to_verify) in claims_to_verify_by_component.iter().enumerate() {
            if let Some(claims_to_verify) = claims_to_verify {
                let layer = match layers_by_component[component].next().unwrap() {
                    Layer::Internal(layer) => layer,
                    Layer::Output(_) => unreachable!(),
                };

                let lambda = channel.draw_felt();
                let n_unused_vars = n_layers - n_layers_by_component[component];
                let sumcheck_oracle = layer.into_sumcheck_oracle(lambda, n_unused_vars, &eq_evals);
                sumcheck_oracles.push(sumcheck_oracle);
                let doubling_factor = BaseField::from(1 << n_unused_vars);
                sumcheck_claims.push(horner_eval(claims_to_verify, lambda) * doubling_factor);
                sumcheck_components.push(component);
            }
        }

        let lambda = channel.draw_felt();
        let (sumcheck_proof, sumcheck_ood_point, folded_sumcheck_oracles, _) =
            sumcheck::prove_batch(sumcheck_claims, sumcheck_oracles, lambda, channel);

        sumcheck_proofs.push(sumcheck_proof);

        let masks = folded_sumcheck_oracles
            .into_iter()
            .map(|oracle| oracle.try_into_mask().unwrap())
            .collect_vec();

        // Seed the channel with the layer masks.
        for (&component, mask) in zip(&sumcheck_components, &masks) {
            channel.mix_felts(mask.columns().flatten());
            layer_masks_by_component[component].push(mask.clone());
        }

        let r_star = channel.draw_felt();
        ood_point = sumcheck_ood_point;
        ood_point.push(r_star);

        // Set the claims to prove in the layer above.
        for (component, mask) in zip(sumcheck_components, masks) {
            claims_to_verify_by_component[component] = Some(mask.reduce_at_point(r_star));
        }
    }

    let output_claims_by_component = output_claims_by_component
        .into_iter()
        .map(Option::unwrap)
        .collect();

    GkrBatchProof {
        sumcheck_proofs,
        layer_masks_by_component,
        output_claims_by_component,
    }
}

/// Partially verifies a batch GKR proof.
///
/// On successful verification the function returns a [`GkrBatchVerificationArtifact`] which stores
/// the out-of-domain point and claimed evaluations in the top layer columns for each component at
/// the OOD point. These evaluations are not checked by this function - hence partial verification.
pub fn partially_verify_batch(
    circuit_unit_by_component: Vec<&dyn GkrBinaryCircuitUnit>,
    proof: &GkrBatchProof,
    channel: &mut impl Channel,
) -> Result<GkrBatchVerificationArtifact, GkrError> {
    let GkrBatchProof {
        sumcheck_proofs,
        layer_masks_by_component,
        output_claims_by_component,
    } = proof;

    let n_components = layer_masks_by_component.len();
    let component_n_layers = |component: usize| layer_masks_by_component[component].len();
    let n_layers = (0..n_components).map(component_n_layers).max().unwrap();

    // TODO(andrew): These should be errors.
    assert_eq!(output_claims_by_component.len(), n_components);
    assert_eq!(circuit_unit_by_component.len(), n_components);
    assert_eq!(sumcheck_proofs.len(), n_layers);

    let mut ood_point = vec![];
    let mut claims_to_verify_by_component = vec![None; n_components];

    for (layer, sumcheck_proof) in sumcheck_proofs.iter().enumerate() {
        let n_remaining_layers = n_layers - layer;

        // Check for output layers.
        for component in 0..n_components {
            if component_n_layers(component) == n_remaining_layers {
                let output_claims = output_claims_by_component[component].clone();
                claims_to_verify_by_component[component] = Some(output_claims);
            }
        }

        // Seed the channel with layer claims.
        for claims_to_verify in claims_to_verify_by_component.iter().flatten() {
            channel.mix_felts(claims_to_verify);
        }

        let mut sumcheck_comopnent_lambdas = Vec::new();
        let mut sumcheck_claims = Vec::new();
        let mut sumcheck_components = Vec::new();

        // Prepare the sumcheck claim.
        for (component, claims_to_verify) in claims_to_verify_by_component.iter().enumerate() {
            if let Some(claims_to_verify) = claims_to_verify {
                let lambda = channel.draw_felt();
                sumcheck_comopnent_lambdas.push(lambda);
                let n_unused = n_layers - component_n_layers(component);
                let doubling_factor = BaseField::from(1 << n_unused);
                sumcheck_claims.push(horner_eval(claims_to_verify, lambda) * doubling_factor);
                sumcheck_components.push(component);
            }
        }

        let sumcheck_lambda = channel.draw_felt();
        let sumcheck_claim = horner_eval(&sumcheck_claims, sumcheck_lambda);
        let (sumcheck_ood_point, sumcheck_eval) =
            sumcheck::partially_verify(sumcheck_claim, sumcheck_proof, channel)
                .map_err(|source| GkrError::InvalidSumcheck { layer, source })?;

        let mut layer_evals = Vec::new();

        // Evaluate the circuit locally at sumcheck OOD point.
        for (&component, lambda) in zip(&sumcheck_components, sumcheck_comopnent_lambdas) {
            let n_unused = n_layers - component_n_layers(component);
            let mask = &layer_masks_by_component[component][layer - n_unused];
            let [row0, row1] = mask.to_rows();
            let circuit_output = circuit_unit_by_component[component].eval(&row0, &row1);
            let folded_output = horner_eval(&circuit_output, lambda);
            let eq_eval = eq(&ood_point[n_unused..], &sumcheck_ood_point[n_unused..]);
            layer_evals.push(eq_eval * folded_output);
        }

        let layer_eval = horner_eval(&layer_evals, sumcheck_lambda);

        if sumcheck_eval != layer_eval {
            return Err(GkrError::CircuitCheckFailure {
                claim: sumcheck_eval,
                output: layer_eval,
                layer,
            });
        }

        // Seed the channel with the layer masks.
        for &component in &sumcheck_components {
            let n_unused = n_layers - component_n_layers(component);
            let mask = &layer_masks_by_component[component][layer - n_unused];
            channel.mix_felts(mask.columns().flatten());
        }

        // Set the OOD evaluation point for layer above.
        let r_star = channel.draw_felt();
        ood_point = sumcheck_ood_point;
        ood_point.push(r_star);

        // Set the claims to verify in the layer above.
        for component in sumcheck_components {
            let n_unused = n_layers - component_n_layers(component);
            let mask = &layer_masks_by_component[component][layer - n_unused];
            claims_to_verify_by_component[component] = Some(mask.reduce_at_point(r_star));
        }
    }

    let claims_to_verify_by_component = claims_to_verify_by_component
        .into_iter()
        .map(Option::unwrap)
        .collect();

    Ok(GkrBatchVerificationArtifact {
        ood_point,
        claims_to_verify_by_component,
    })
}

/// Evaluates the GKR circuit on the top layer and returns all the circuit's layers.
fn gen_layers<L: GkrLayer>(top_layer: L) -> Vec<Layer<L>> {
    let n_variables = top_layer.n_variables();
    let layers = successors(Some(Layer::Internal(top_layer)), |layer| layer.next()).collect_vec();
    assert_eq!(layers.len(), n_variables);
    layers
}

pub struct GkrBatchProof {
    pub sumcheck_proofs: Vec<SumcheckProof>,
    pub layer_masks_by_component: Vec<Vec<GkrMask>>,
    pub output_claims_by_component: Vec<Vec<SecureField>>,
}

pub struct GkrBatchVerificationArtifact {
    /// Out-of-domain (OOD) point for columns in the top layer.
    pub ood_point: Vec<SecureField>,
    /// The claimed evaluation at `ood_point` for each column in the top layer of each component.
    pub claims_to_verify_by_component: Vec<Vec<SecureField>>,
}

// TODO: Docs.
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

    // TODO: Docs.
    fn reduce_at_point(&self, p: SecureField) -> Vec<SecureField> {
        self.columns
            .iter()
            .map(|&[v0, v1]| {
                eq(&[SecureField::zero()], &[p]) * v0 + eq(&[SecureField::one()], &[p]) * v1
            })
            .collect()
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
