//! Batch GKR protocol implementation designed to prove lookup arguments.
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
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::lookups::sumcheck;

pub trait GkrOps: MleOps<BaseField> + MleOps<SecureField> {
    /// Returns evaluations `eq(x, y) * v` for all `x` in `{0, 1}^n`.
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
    pub fn new(y: &[SecureField]) -> Self {
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
    GrandProduct(Mle<B, SecureField>),
    LogUpGeneric {
        numerators: Mle<B, SecureField>,
        denominators: Mle<B, SecureField>,
    },
    LogUpMultiplicities {
        numerators: Mle<B, BaseField>,
        denominators: Mle<B, SecureField>,
    },
    /// All numerators implicitly equal "1".
    LogUpSingles {
        denominators: Mle<B, SecureField>,
    },
}

impl<B: GkrOps> Layer<B> {
    /// Returns the number of variables used to interpolate the layer's gate values.
    fn n_variables(&self) -> usize {
        match self {
            Self::GrandProduct(mle)
            | Self::LogUpSingles { denominators: mle }
            | Self::LogUpMultiplicities {
                denominators: mle, ..
            }
            | Self::LogUpGeneric {
                denominators: mle, ..
            } => mle.n_variables(),
        }
    }

    fn is_output_layer(&self) -> bool {
        self.n_variables() == 0
    }

    /// Produces the next layer from the current layer.
    ///
    /// The next layer is strictly half the size of the current layer.
    /// Returns [`None`] if called on an output layer.
    pub fn next_layer(&self) -> Option<Self> {
        if self.is_output_layer() {
            return None;
        }

        Some(B::next_layer(self))
    }

    /// Returns each column output if the layer is an output layer, otherwise returns an `Err`.
    fn try_into_output_layer_values(self) -> Result<Vec<SecureField>, NotOutputLayerError> {
        if !self.is_output_layer() {
            return Err(NotOutputLayerError);
        }

        Ok(match self {
            Layer::LogUpSingles { denominators } => {
                let numerator = SecureField::one();
                let denominator = denominators.at(0);
                vec![numerator, denominator]
            }
            Layer::LogUpMultiplicities {
                numerators,
                denominators,
            } => {
                let numerator = numerators.at(0).into();
                let denominator = denominators.at(0);
                vec![numerator, denominator]
            }
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => {
                let numerator = numerators.at(0);
                let denominator = denominators.at(0);
                vec![numerator, denominator]
            }
            Layer::GrandProduct(col) => {
                vec![col.at(0)]
            }
        })
    }

    /// Returns a transformed layer with the first variable of each column fixed to `assignment`.
    fn fix_first_variable(self, x0: SecureField) -> Self {
        if self.n_variables() == 0 {
            return self;
        }

        match self {
            Self::GrandProduct(mle) => Self::GrandProduct(mle.fix_first_variable(x0)),
            Self::LogUpGeneric {
                numerators,
                denominators,
            } => Self::LogUpGeneric {
                numerators: numerators.fix_first_variable(x0),
                denominators: denominators.fix_first_variable(x0),
            },
            Self::LogUpMultiplicities {
                numerators,
                denominators,
            } => Self::LogUpGeneric {
                numerators: numerators.fix_first_variable(x0),
                denominators: denominators.fix_first_variable(x0),
            },
            Self::LogUpSingles { denominators } => Self::LogUpSingles {
                denominators: denominators.fix_first_variable(x0),
            },
        }
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
        lambda: SecureField,
        eq_evals: &EqEvals<B>,
    ) -> GkrMultivariatePolyOracle<'_, B> {
        GkrMultivariatePolyOracle {
            eq_evals,
            input_layer: self,
            eq_fixed_var_correction: SecureField::one(),
            lambda,
        }
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
    /// Used by LogUp to perform a random linear combination of the numerators and denominators.
    pub lambda: SecureField,
}

impl<'a, B: GkrOps> MultivariatePolyOracle for GkrMultivariatePolyOracle<'a, B> {
    fn n_variables(&self) -> usize {
        self.input_layer.n_variables() - 1
    }

    fn sum_as_poly_in_first_variable(&self, claim: SecureField) -> UnivariatePoly<SecureField> {
        B::sum_as_poly_in_first_variable(self, claim)
    }

    fn fix_first_variable(self, challenge: SecureField) -> Self {
        if self.is_constant() {
            return self;
        }

        let z0 = self.eq_evals.y()[self.eq_evals.y().len() - self.n_variables()];
        let eq_fixed_var_correction = self.eq_fixed_var_correction * eq(&[challenge], &[z0]);

        Self {
            eq_evals: self.eq_evals,
            eq_fixed_var_correction,
            input_layer: self.input_layer.fix_first_variable(challenge),
            lambda: self.lambda,
        }
    }
}

impl<'a, B: GkrOps> GkrMultivariatePolyOracle<'a, B> {
    fn is_constant(&self) -> bool {
        self.n_variables() == 0
    }

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
        if !self.is_constant() {
            return Err(NotConstantPolyError);
        }

        let columns = match self.input_layer {
            Layer::GrandProduct(mle) => vec![mle.to_cpu().try_into().unwrap()],
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => {
                let numerators = numerators.to_cpu().try_into().unwrap();
                let denominators = denominators.to_cpu().try_into().unwrap();
                vec![numerators, denominators]
            }
            // Should never get called.
            Layer::LogUpMultiplicities { .. } => unimplemented!(),
            Layer::LogUpSingles { denominators } => {
                let numerators = [SecureField::one(); 2];
                let denominators = denominators.to_cpu().try_into().unwrap();
                vec![numerators, denominators]
            }
        };

        Ok(GkrMask::new(columns))
    }
}

/// Error returned when a polynomial is expected to be constant but it is not.
#[derive(Debug, Error)]
#[error("polynomial is not constant")]
pub struct NotConstantPolyError;

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

        let eq_evals = EqEvals::new(&ood_point);
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

/// Corrects and interpolates GKR instance sumcheck round polynomials that are generated with the
/// precomputed `eq(x, y)` evaluations provided by `Layer::into_multivariate_poly()`.
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
/// For more context see `Layer::into_multivariate_poly()` docs.
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
