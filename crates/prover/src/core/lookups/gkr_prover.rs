//! GKR batch prover for Grand Product and LogUp lookup arguments.
use std::iter::{successors, zip};
use std::ops::Deref;

use itertools::Itertools;
use num_traits::{One, Zero};
use thiserror::Error;

use super::gkr_verifier::{GkrArtifact, GkrBatchProof, GkrMask};
use super::mle::{Mle, MleOps};
use super::sumcheck::MultivariatePolyOracle;
use super::utils::{eq, random_linear_combination, UnivariatePoly};
use crate::core::backend::{Col, Column, ColumnOps};
use crate::core::channel::Channel;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::sumcheck;

pub trait GkrOps: MleOps<SecureField> {
    /// Returns evaluations `eq(x, y) * v` for all `x` in `{0, 1}^n`.
    ///
    /// Note [`Mle`] stores values in bit-reversed order.
    ///
    /// [`eq(x, y)`]: crate::core::lookups::utils::eq
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField>;

    /// Generates the next GKR layer from the current one.
    fn next_layer(layer: &Layer<Self>) -> Layer<Self>;

    /// Returns univariate polynomial `f(t) = sum_x h(t, x)` for all `x` in the boolean hypercube.
    ///
    /// `claim` equals `f(0) + f(1)`.
    ///
    /// For more context see docs of [`MultivariatePolyOracle::sum_as_poly_in_first_variable()`].
    fn sum_as_poly_in_first_variable(
        h: &GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField>;
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
    pub fn generate(y: &[SecureField]) -> Self {
        let y = y.to_vec();

        if y.is_empty() {
            let evals = Mle::new([SecureField::one()].into_iter().collect());
            return Self { evals, y };
        }

        let evals = B::gen_eq_evals(&y[1..], eq(&[SecureField::zero()], &[y[0]]));
        assert_eq!(evals.len(), 1 << (y.len() - 1));
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

/// Represents a layer in a binary tree structured GKR circuit.
///
/// Layers can contain multiple columns, for example [LogUp] which has separate columns for
/// numerators and denominators.
///
/// [LogUp]: https://eprint.iacr.org/2023/1284.pdf
pub enum Layer<B: GkrOps> {
    _LogUp(B),
    _GrandProduct(B),
}

impl<B: GkrOps> Layer<B> {
    /// Returns the number of variables used to interpolate the layer's gate values.
    fn n_variables(&self) -> usize {
        todo!()
    }

    /// Produces the next layer from the current layer.
    ///
    /// The next layer is strictly half the size of the current layer.
    /// Returns [`None`] if called on an output layer.
    fn next_layer(&self) -> Option<Layer<B>> {
        if self.is_output_layer() {
            return None;
        }

        Some(B::next_layer(self))
    }

    fn is_output_layer(&self) -> bool {
        self.n_variables() == 0
    }

    /// Returns each column output if the layer is an output layer, otherwise returns an `Err`.
    fn try_into_output_layer_values(self) -> Result<Vec<SecureField>, NotOutputLayerError> {
        todo!()
    }

    /// Represents the next GKR layer evaluation as a multivariate polynomial which uses this GKR
    /// layer as input.
    ///
    /// Layers can contain multiple columns `c_0, ..., c_{n-1}` with multivariate polynomial `g_i`
    /// representing[^note] `c_i` in the next layer. These polynomials must be combined with
    /// `lambda` into a single polynomial `h = g_0 + lambda * g_1 + ... + lambda^(n-1) *
    /// g_{n-1}`. The oracle for `h` should be returned.
    ///
    /// # Optimization: precomputed [`eq(x, y)`] evals
    ///
    /// Let `y` be a fixed vector of length `m` and let `z` be a subvector comprising of the
    /// last `k` elements of `y`. `h(x)` **must** equal some multivariate polynomial of the form
    /// `eq(x, z) * p(x)`. A common operation will be computing the univariate polynomial `f(t) =
    /// sum_x h(t, x)` for `x` in the boolean hypercube `{0, 1}^(k-1)`.
    ///
    /// `eq_evals` stores evaluations of `eq((0, x), y)` for `x` in a potentially extended boolean
    /// hypercube `{0, 1}^{m-1}`. These evaluations, on the extended hypercube, can be used directly
    /// in computing the sums of `h(x)`, however a correction factor must be applied to the final
    /// sum which is handled by [`correct_sum_as_poly_in_first_variable()`] in `O(m)`.
    ///
    /// Being able to compute sums of `h(x)` using `eq_evals` in this way leads to a more efficient
    /// implementation because the prover only has to generate `eq_evals` once for an entire batch
    /// of multiple GKR layer instances.
    ///
    /// [`eq(x, y)`]: crate::core::lookups::utils::eq
    /// [^note]: By "representing" we mean `g_i` agrees with the next layer's `c_i` on the boolean
    /// hypercube that interpolates `c_i`.
    fn into_multivariate_poly(
        self,
        _lambda: SecureField,
        _eq_evals: &EqEvals<B>,
    ) -> GkrMultivariatePolyOracle<'_, B> {
        todo!()
    }
}

#[derive(Debug)]
struct NotOutputLayerError;

/// A multivariate polynomial that expresses the relation between two consecutive GKR layers.
pub struct GkrMultivariatePolyOracle<'a, B: GkrOps> {
    /// `eq_evals` passed by `Layer::into_multivariate_poly()`.
    pub eq_evals: &'a EqEvals<B>,
    pub input_layer: Layer<B>,
    pub eq_fixed_var_correction: SecureField,
}

impl<'a, B: GkrOps> MultivariatePolyOracle for GkrMultivariatePolyOracle<'a, B> {
    fn n_variables(&self) -> usize {
        todo!()
    }

    fn sum_as_poly_in_first_variable(&self, _claim: SecureField) -> UnivariatePoly<SecureField> {
        todo!()
    }

    fn fix_first_variable(self, _challenge: SecureField) -> Self {
        todo!()
    }
}

impl<'a, B: GkrOps> GkrMultivariatePolyOracle<'a, B> {
    /// Returns all input layer columns restricted to a line.
    ///
    /// Let `l` be the line satisfying `l(0) = b*` and `l(1) = c*`. Oracles that represent constants
    /// are expressed by values `c_i(b*)` and `c_i(c*)` where `c_i` represents the input GKR layer's
    /// `i`th column (for binary tree GKR `b* = (r, 0)`, `c* = (r, 1)`).
    ///
    /// If this oracle represents a constant, then each `c_i` restricted to `l` is returned.
    /// Otherwise, an [`Err`] is returned.
    ///
    /// For more context see <https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf> page 64.
    fn try_into_mask(self) -> Result<GkrMask, NotConstantPolyError> {
        todo!()
    }
}

/// Error returned when a polynomial is expected to be constant but it is not.
#[derive(Debug, Error)]
#[error("polynomial is not constant")]
pub struct NotConstantPolyError;

/// Batch proves lookup circuits with GKR.
///
/// The input layers should be committed to the channel before calling this function.
// GKR algorithm: <https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf> (page 64)
pub fn prove_batch<B: GkrOps>(
    channel: &mut impl Channel,
    input_layer_by_instance: Vec<Layer<B>>,
) -> (GkrBatchProof, GkrArtifact) {
    let n_instances = input_layer_by_instance.len();
    let n_layers_by_instance = input_layer_by_instance
        .iter()
        .map(|l| l.n_variables())
        .collect_vec();
    let n_layers = *n_layers_by_instance.iter().max().unwrap();

    // Evaluate all instance circuits and collect the layer values.
    let mut layers_by_instance = input_layer_by_instance
        .into_iter()
        .map(|input_layer| gen_layers(input_layer).into_iter().rev())
        .collect_vec();

    let mut output_claims_by_instance = vec![None; n_instances];
    let mut layer_masks_by_instance = (0..n_instances).map(|_| Vec::new()).collect_vec();
    let mut sumcheck_proofs = Vec::new();

    let mut ood_point = Vec::new();
    let mut claims_to_verify_by_instance = vec![None; n_instances];

    for layer in 0..n_layers {
        let n_remaining_layers = n_layers - layer;

        // Check all the instances for output layers.
        for (instance, layers) in layers_by_instance.iter_mut().enumerate() {
            if n_layers_by_instance[instance] == n_remaining_layers {
                let output_layer = layers.next().unwrap();
                let output_layer_values = output_layer.try_into_output_layer_values().unwrap();
                claims_to_verify_by_instance[instance] = Some(output_layer_values.clone());
                output_claims_by_instance[instance] = Some(output_layer_values);
            }
        }

        // Seed the channel with layer claims.
        for claims_to_verify in claims_to_verify_by_instance.iter().flatten() {
            channel.mix_felts(claims_to_verify);
        }

        let eq_evals = EqEvals::generate(&ood_point);
        let sumcheck_alpha = channel.draw_felt();
        let instance_lambda = channel.draw_felt();

        let mut sumcheck_oracles = Vec::new();
        let mut sumcheck_claims = Vec::new();
        let mut sumcheck_instances = Vec::new();

        // Create the multivariate polynomial oracles used with sumcheck.
        for (instance, claims_to_verify) in claims_to_verify_by_instance.iter().enumerate() {
            if let Some(claims_to_verify) = claims_to_verify {
                let layer = layers_by_instance[instance].next().unwrap();
                sumcheck_oracles.push(layer.into_multivariate_poly(instance_lambda, &eq_evals));
                sumcheck_claims.push(random_linear_combination(claims_to_verify, instance_lambda));
                sumcheck_instances.push(instance);
            }
        }

        let (sumcheck_proof, sumcheck_ood_point, constant_poly_oracles, _) =
            sumcheck::prove_batch(sumcheck_claims, sumcheck_oracles, sumcheck_alpha, channel);

        sumcheck_proofs.push(sumcheck_proof);

        let masks = constant_poly_oracles
            .into_iter()
            .map(|oracle| oracle.try_into_mask().unwrap())
            .collect_vec();

        // Seed the channel with the layer masks.
        for (&instance, mask) in zip(&sumcheck_instances, &masks) {
            channel.mix_felts(mask.columns().flatten());
            layer_masks_by_instance[instance].push(mask.clone());
        }

        let challenge = channel.draw_felt();
        ood_point = sumcheck_ood_point;
        ood_point.push(challenge);

        // Set the claims to prove in the layer above.
        for (instance, mask) in zip(sumcheck_instances, masks) {
            claims_to_verify_by_instance[instance] = Some(mask.reduce_at_point(challenge));
        }
    }

    let output_claims_by_instance = output_claims_by_instance
        .into_iter()
        .map(Option::unwrap)
        .collect();

    let claims_to_verify_by_instance = claims_to_verify_by_instance
        .into_iter()
        .map(Option::unwrap)
        .collect();

    let proof = GkrBatchProof {
        sumcheck_proofs,
        layer_masks_by_instance,
        output_claims_by_instance,
    };

    let artifact = GkrArtifact {
        ood_point,
        claims_to_verify_by_instance,
        n_variables_by_instance: n_layers_by_instance,
    };

    (proof, artifact)
}

/// Executes the GKR circuit on the input layer and returns all the circuit's layers.
fn gen_layers<B: GkrOps>(input_layer: Layer<B>) -> Vec<Layer<B>> {
    let n_variables = input_layer.n_variables();
    let layers = successors(Some(input_layer), |layer| layer.next_layer()).collect_vec();
    assert_eq!(layers.len(), n_variables + 1);
    layers
}
