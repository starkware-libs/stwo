//! GKR protocol implementation designed to prove and verify lookup arguments.
use std::iter::{successors, zip};
use std::ops::Deref;

use itertools::Itertools;
use num_traits::{One, Zero};
use thiserror::Error;

use super::mle::{Mle, MleOps};
use super::sumcheck::{MultivariatePolyOracle, SumcheckError, SumcheckProof};
use super::utils::{eq, fold_mle_evals, horner_eval, UnivariatePoly};
use crate::core::backend::{Col, Column, ColumnOps};
use crate::core::channel::Channel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::lookups::sumcheck;

pub trait GkrOps: MleOps<SecureField> {
    /// Returns the evaluations of [`eq(x, y)`] on all boolean hypercube points of the form
    /// `x = (0, x_1, ..., x_{n-1})`.
    ///
    /// [`eq(x, y)`]: crate::core::lookups::utils::eq
    fn gen_eq_evals(y: &[SecureField]) -> Mle<Self, SecureField>;
}

/// Stores evaluations of [`eq(x, y)`] on all boolean hypercube points of the form
/// `x = (0, x_1, ..., x_{n-1})`.
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

    /// Returns the fixed vector `y` used to generate the evaluations.
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
///
/// Layers can contain multiple multilinear extension columns, for example [LogUp] which has
/// separate columns for numerators and denominators.
///
/// [LogUp]: https://eprint.iacr.org/2023/1284.pdf
pub trait GkrBinaryLayer: Sized {
    type Backend: GkrOps + MleOps<SecureField>;

    type MultivariatePolyOracle<'a>: GkrMultivariatePolyOracle<Backend = Self::Backend>;

    /// Returns the number of variables used to interpolate the layer's gate values.
    fn n_variables(&self) -> usize;

    /// Produces the next GKR layer from the current layer.
    ///
    /// The next layer should be strictly half the size of the current layer.
    fn next(&self) -> Layer<Self>;

    /// Represents the next GKR layer evaluation as a multivariate polynomial which uses this GKR
    /// layer as input.
    ///
    /// Layers can contain multiple columns `c_0, ..., c_{n-1}` with multivariate polynomial `g_i`
    /// representing[^note] `c_i` in the next layer. These polynomials must be combined with
    /// `lambda` into a single polynomial `h = g_0 + lambda * g_1 + ... + lambda^(n-1) *
    /// g_{n-1}`. The oracle for `h` should be returned.
    ///
    /// ## Optimization: precomputed `eq(x, y)` evals
    ///
    /// Let `y` be a fixed vector of length `m` and let `z` be a subvector comprising of the
    /// last `k` elements of `y`. `h(x)` **must** equal some multivariate polynomial of the form
    /// `eq(x, z) * p(x)`. A common operation will be computing the univariate polynomial `f(t) =
    /// sum_x h(t, x)` for `x` in the boolean hypercube `{0, 1}^(k-1)`.
    ///
    /// `eq_evals` stores evaluations of `eq((0, x), y)` for `x` in a potentially extended boolean
    /// hypercube `{0, 1}^{m-1}`. These evaluations, on the extended hypercube, can be used directly
    /// in computing the sums of `h(x)`, however a correction factor must be applied to the final
    /// sum which is handled by [`correct_sum_as_poly_in_first_variable`] in `O(m)`.
    ///
    /// Being able to compute sums of `h(x)` using `eq_evals` in this way leads to a more efficient
    /// implementation because the prover only has to generate `eq_evals` once for an entire batch
    /// of multiple GKR layer instances.
    ///
    /// [^note]: By "representing" we mean `g_i` agrees with the next layer's `c_i` on the boolean
    /// hypercube that interpolates `c_i`.
    fn into_multivariate_poly(
        self,
        lambda: SecureField,
        eq_evals: &EqEvals<Self::Backend>,
    ) -> Self::MultivariatePolyOracle<'_>;
}

/// Represents a layer in a GKR circuit.
pub enum Layer<L: GkrBinaryLayer> {
    /// The output layer. Each column of the output layer is represented by a single element.
    Output(Vec<SecureField>),
    /// Any layer that isn't the output layer.
    Internal(L),
}

impl<L: GkrBinaryLayer> Layer<L> {
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
    fn try_into_mask(self) -> Option<GkrMask>;
}

/// Defines how a circuit operates locally on pairs of input rows to produce a single output row.
/// This local 2-to-1 constraint is what gives the whole circuit its "binary tree" structure.
///
/// Binary tree structured circuits have a highly regular wiring pattern that fit the structure of
/// the circuits defined in [Thaler13] which allow for efficient linear time (linear in size of the
/// circuit) GKR prover implementations.
///
/// [Thaler13]: https://eprint.iacr.org/2013/351.pdf
pub trait GkrBinaryGate {
    /// Returns the output row after applying the gate to the provided input rows.
    fn eval(&self, row0: &[SecureField], row1: &[SecureField]) -> Vec<SecureField>;
}

// GKR algorithm: <https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf> (page 65)
pub fn prove_batch<L: GkrBinaryLayer>(
    channel: &mut impl Channel,
    top_layers: Vec<L>,
) -> GkrBatchProof {
    let n_components = top_layers.len();
    let n_layers_by_component = top_layers.iter().map(|l| l.n_variables()).collect_vec();
    let n_layers = top_layers.iter().map(|l| l.n_variables()).max().unwrap();

    // Evaluate all component circuits and collect the layer values.
    let mut layers_by_component = top_layers
        .into_iter()
        .map(|top_layer| gen_layers(top_layer).into_iter().rev().peekable())
        .collect_vec();

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
        let sumcheck_lambda = channel.draw_felt();
        let component_lambda = channel.draw_felt();

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

                sumcheck_oracles.push(layer.into_multivariate_poly(component_lambda, &eq_evals));
                sumcheck_claims.push(horner_eval(claims_to_verify, component_lambda));
                sumcheck_components.push(component);
            }
        }

        let (sumcheck_proof, sumcheck_ood_point, fixed_sumcheck_oracles, _) =
            sumcheck::prove_batch(sumcheck_claims, sumcheck_oracles, sumcheck_lambda, channel);

        sumcheck_proofs.push(sumcheck_proof);

        let masks = fixed_sumcheck_oracles
            .into_iter()
            .map(|oracle| oracle.try_into_mask().unwrap())
            .collect_vec();

        // Seed the channel with the layer masks.
        for (&component, mask) in zip(&sumcheck_components, &masks) {
            channel.mix_felts(mask.columns().flatten());
            layer_masks_by_component[component].push(mask.clone());
        }

        let challenge = channel.draw_felt();
        ood_point = sumcheck_ood_point;
        ood_point.push(challenge);

        // Set the claims to prove in the layer above.
        for (component, mask) in zip(sumcheck_components, masks) {
            claims_to_verify_by_component[component] = Some(mask.reduce_at_point(challenge));
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
/// the OOD point. These claimed evaluations are not checked in this function - hence partial
/// verification.
pub fn partially_verify_batch(
    gate_by_component: Vec<&dyn GkrBinaryGate>,
    proof: &GkrBatchProof,
    channel: &mut impl Channel,
) -> Result<GkrBatchVerificationArtifact, GkrError> {
    let GkrBatchProof {
        sumcheck_proofs,
        layer_masks_by_component,
        output_claims_by_component,
    } = proof;

    if layer_masks_by_component.len() != output_claims_by_component.len() {
        return Err(GkrError::MalformedProof);
    }

    let n_components = layer_masks_by_component.len();
    let component_n_layers = |component: usize| layer_masks_by_component[component].len();
    let n_layers = (0..n_components).map(component_n_layers).max().unwrap();

    if n_layers != sumcheck_proofs.len() {
        return Err(GkrError::MalformedProof);
    }

    if gate_by_component.len() != n_components {
        return Err(GkrError::NumComponentsMismatch {
            given: gate_by_component.len(),
            proof: n_components,
        });
    }

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

        let sumcheck_lambda = channel.draw_felt();
        let component_lambda = channel.draw_felt();

        let mut sumcheck_claims = Vec::new();
        let mut sumcheck_components = Vec::new();

        // Prepare the sumcheck claim.
        for (component, claims_to_verify) in claims_to_verify_by_component.iter().enumerate() {
            if let Some(claims_to_verify) = claims_to_verify {
                let n_unused_variables = n_layers - component_n_layers(component);
                let doubling_factor = BaseField::from(1 << n_unused_variables);
                let claim = horner_eval(claims_to_verify, component_lambda) * doubling_factor;
                sumcheck_claims.push(claim);
                sumcheck_components.push(component);
            }
        }

        let sumcheck_claim = horner_eval(&sumcheck_claims, sumcheck_lambda);
        let (sumcheck_ood_point, sumcheck_eval) =
            sumcheck::partially_verify(sumcheck_claim, sumcheck_proof, channel)
                .map_err(|source| GkrError::InvalidSumcheck { layer, source })?;

        let mut layer_evals = Vec::new();

        // Evaluate the circuit locally at sumcheck OOD point.
        for &component in &sumcheck_components {
            let n_unused = n_layers - component_n_layers(component);
            let mask = &layer_masks_by_component[component][layer - n_unused];
            let [row0, row1] = mask.to_rows();
            let gate_output = gate_by_component[component].eval(&row0, &row1);
            let folded_output = horner_eval(&gate_output, component_lambda);
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
        let challenge = channel.draw_felt();
        ood_point = sumcheck_ood_point;
        ood_point.push(challenge);

        // Set the claims to verify in the layer above.
        for component in sumcheck_components {
            let n_unused = n_layers - component_n_layers(component);
            let mask = &layer_masks_by_component[component][layer - n_unused];
            claims_to_verify_by_component[component] = Some(mask.reduce_at_point(challenge));
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
fn gen_layers<L: GkrBinaryLayer>(top_layer: L) -> Vec<Layer<L>> {
    let n_variables = top_layer.n_variables();
    let layers = successors(Some(Layer::Internal(top_layer)), |layer| layer.next()).collect_vec();
    assert_eq!(layers.len(), n_variables + 1);
    layers
}

/// Batch GKR proof.
pub struct GkrBatchProof {
    /// Sum-check proof for each layer.
    pub sumcheck_proofs: Vec<SumcheckProof>,
    /// Mask for each layer for each component.
    pub layer_masks_by_component: Vec<Vec<GkrMask>>,
    /// Column circuit outputs for each component.
    pub output_claims_by_component: Vec<Vec<SecureField>>,
}

/// Values obtained through verifying a GKR proof.
pub struct GkrBatchVerificationArtifact {
    /// Out-of-domain (OOD) point for evaluating columns in the top layer.
    pub ood_point: Vec<SecureField>,
    /// The claimed evaluation at `ood_point` for each column in the top layer of each component.
    pub claims_to_verify_by_component: Vec<Vec<SecureField>>,
}

/// Stores two evaluations of each column in a GKR layer.
#[derive(Debug, Clone)]
pub struct GkrMask {
    columns: Vec<[SecureField; 2]>,
}

impl GkrMask {
    pub fn new(columns: Vec<[SecureField; 2]>) -> Self {
        Self { columns }
    }

    fn to_rows(&self) -> [Vec<SecureField>; 2] {
        self.columns.iter().map(|[a, b]| (a, b)).unzip().into()
    }

    fn columns(&self) -> &[[SecureField; 2]] {
        &self.columns
    }

    /// Returns all `p_i(x)` where `p_i` interpolates column `i` of the mask on `{0, 1}`.
    fn reduce_at_point(&self, x: SecureField) -> Vec<SecureField> {
        self.columns
            .iter()
            .map(|&[v0, v1]| fold_mle_evals(x, v0, v1))
            .collect()
    }
}

/// Error encountered during GKR protocol verification.
///
/// Layer 0 corresponds to the output layer.
#[derive(Error, Debug)]
pub enum GkrError {
    /// The proof is malformed.
    #[error("proof data is invalid")]
    MalformedProof,
    /// There is a mismatch between the number of components in the proof and the number of
    /// components passed for verification.
    #[error("provided an invalid number of components (given {given}, proof expects {proof})")]
    NumComponentsMismatch { given: usize, proof: usize },
    /// There was an error with one of the sumcheck proofs.
    #[error("sum-check invalid in layer {layer}: {source}")]
    InvalidSumcheck { layer: usize, source: SumcheckError },
    /// The circuit polynomial the verifier evaluated doesn't match claim from sumcheck.
    #[error("circuit check failed in layer {layer} (calculated {output}, claim {claim})")]
    CircuitCheckFailure {
        claim: SecureField,
        output: SecureField,
        layer: usize,
    },
}

/// Corrects and interpolates GKR component sumcheck round polynomials that are generated with the
/// precomputed `eq(x, y)` evaluations provided by [`GkrBinaryLayer::into_multivariate_poly()`].
///
/// Let `y` be a fixed vector of length `n` and let `z` be a subvector comprising of the last `k`
/// elements of `y`. Returns the univariate polynomial `f(t) = sum_x eq((t, x), z) * p(t, x)` for
/// `x` in the boolean hypercube `{0, 1}^(k-1)` when provided with:
///
/// * `claim` equalling `f(0) + f(1)`.
/// * `eval_at_0/2` equalling `sum_x eq(({0}^(n-k+1), x), y) * p(t, x)` at `t=0,2` respectively.
///
/// Note that `f` must have degree <= 3.
///
/// For more context see [`GkrBinaryLayer::into_multivariate_poly()`] docs.
/// See also <https://ia.cr/2024/108> (section 3.2).
///
/// # Panics
///
/// Panics if:
/// * `k` is zero or greater than the length of `y`.
/// * `z_0` is zero.
pub fn correct_sum_as_poly_in_first_variable(
    eval_at_0: SecureField,
    eval_at_2: SecureField,
    claim: SecureField,
    y: &[SecureField],
    k: usize,
) -> UnivariatePoly<SecureField> {
    assert_ne!(k, 0);
    let n = y.len();
    assert!(k <= n);

    let z = &y[n - k..];

    // Corrects the difference between two sums:
    // 1. `sum_x eq(({0}^(n-k+1), x), y) * p(t, x)`
    // 2. `sum_x eq((0, x), z) * p(t, x)`
    let eq_y_to_z_correction_factor = eq(&vec![SecureField::zero(); n - k], &y[0..n - k]).inverse();

    // Corrects the difference between two sums:
    // 1. `sum_x eq((0, x), z) * p(t, x)`
    // 2. `sum_x eq((t, x), z) * p(t, x)`
    let eq_correction_factor_at = |t| eq(&[t], &[z[0]]) / eq(&[SecureField::zero()], &[z[0]]);

    // Let `v(t) = sum_x eq((0, x), z) * p(t, x)`. Apply trick from
    // <https://ia.cr/2024/108> (section 3.2) to obtain `f` from `v`.
    let t0: SecureField = BaseField::zero().into();
    let t1: SecureField = BaseField::one().into();
    let t2: SecureField = BaseField::from(2).into();
    let t3: SecureField = BaseField::from(3).into();

    // Obtain evals `v(0)`, `v(1)`, `v(2)`.
    let mut y0 = eq_y_to_z_correction_factor * eval_at_0;
    let mut y1 = (claim - y0) / eq_correction_factor_at(t1);
    let mut y2 = eq_y_to_z_correction_factor * eval_at_2;

    // Interpolate `v` to find `v(3)`. Note `v` has degree <= 2.
    let v = UnivariatePoly::interpolate_lagrange(&[t0, t1, t2], &[y0, y1, y2]);
    let mut y3 = v.eval_at_point(t3);

    // Obtain evals of `f(0)`, `f(1)`, `f(2)`, `f(3)`.
    y0 *= eq_correction_factor_at(t0);
    y1 *= eq_correction_factor_at(t1);
    y2 *= eq_correction_factor_at(t2);
    y3 *= eq_correction_factor_at(t3);

    // Interpolate `f(t)`. Note `f(t)` has degree <= 3.
    UnivariatePoly::interpolate_lagrange(&[t0, t1, t2, t3], &[y0, y1, y2, y3])
}

#[cfg(test)]
mod tests {
    use super::GkrError;
    use crate::core::backend::CpuBackend;
    use crate::core::channel::Channel;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::gkr::{
        partially_verify_batch, prove_batch, GkrBatchVerificationArtifact,
    };
    use crate::core::lookups::grandproduct::{GrandProductGate, GrandProductTrace};
    use crate::core::lookups::mle::Mle;
    use crate::core::test_utils::test_channel;

    #[test]
    fn prove_batch_works() -> Result<(), GkrError> {
        const N: usize = 1 << 5;
        let mut channel = test_channel();
        let col0 = GrandProductTrace::<CpuBackend>::new(Mle::new(channel.draw_felts(N)));
        let col1 = GrandProductTrace::<CpuBackend>::new(Mle::new(channel.draw_felts(N)));
        let product0 = col0.iter().product::<SecureField>();
        let product1 = col1.iter().product::<SecureField>();
        let top_layers = vec![col0.clone(), col1.clone()];
        let proof = prove_batch(&mut test_channel(), top_layers);

        let GkrBatchVerificationArtifact {
            ood_point,
            claims_to_verify_by_component,
        } = partially_verify_batch(vec![&GrandProductGate; 2], &proof, &mut test_channel())?;

        assert_eq!(proof.output_claims_by_component.len(), 2);
        assert_eq!(claims_to_verify_by_component.len(), 2);
        assert_eq!(proof.output_claims_by_component[0], &[product0]);
        assert_eq!(proof.output_claims_by_component[1], &[product1]);
        let claim0 = &claims_to_verify_by_component[0];
        let claim1 = &claims_to_verify_by_component[1];
        assert_eq!(claim0, &[col0.eval_at_point(&ood_point)]);
        assert_eq!(claim1, &[col1.eval_at_point(&ood_point)]);
        Ok(())
    }

    #[test]
    fn prove_batch_with_different_sizes_works() -> Result<(), GkrError> {
        const LOG_N0: usize = 5;
        const LOG_N1: usize = 7;
        let mut channel = test_channel();
        let col0 = GrandProductTrace::<CpuBackend>::new(Mle::new(channel.draw_felts(1 << LOG_N0)));
        let col1 = GrandProductTrace::<CpuBackend>::new(Mle::new(channel.draw_felts(1 << LOG_N1)));
        let product0 = col0.iter().product::<SecureField>();
        let product1 = col1.iter().product::<SecureField>();
        let top_layers = vec![col0.clone(), col1.clone()];
        let proof = prove_batch(&mut test_channel(), top_layers);

        let GkrBatchVerificationArtifact {
            ood_point,
            claims_to_verify_by_component,
        } = partially_verify_batch(vec![&GrandProductGate; 2], &proof, &mut test_channel())?;

        assert_eq!(proof.output_claims_by_component.len(), 2);
        assert_eq!(claims_to_verify_by_component.len(), 2);
        assert_eq!(proof.output_claims_by_component[0], &[product0]);
        assert_eq!(proof.output_claims_by_component[1], &[product1]);
        let claim0 = &claims_to_verify_by_component[0];
        let claim1 = &claims_to_verify_by_component[1];
        let n_vars = ood_point.len();
        assert_eq!(claim0, &[col0.eval_at_point(&ood_point[n_vars - LOG_N0..])]);
        assert_eq!(claim1, &[col1.eval_at_point(&ood_point[n_vars - LOG_N1..])]);
        Ok(())
    }
}
