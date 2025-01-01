//! GKR batch prover for Grand Product and LogUp lookup arguments.
use std::borrow::Cow;
use std::iter::{successors, zip};
use std::ops::Deref;

use educe::Educe;
use itertools::Itertools;
use num_traits::{One, Zero};
use thiserror::Error;

use super::gkr_verifier::{GkrArtifact, GkrBatchProof, GkrMask};
use super::mle::{Mle, MleOps};
use super::sumcheck::MultivariatePolyOracle;
use super::utils::{eq, random_linear_combination, UnivariatePoly};
use crate::core::backend::{Col, Column, ColumnOps, CpuBackend};
use crate::core::channel::Channel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{Field, FieldExpOps};
use crate::core::lookups::sumcheck;

pub trait GkrOps: MleOps<BaseField> + MleOps<SecureField> {
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
#[derive(Educe)]
#[educe(Debug, Clone)]
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
#[derive(Educe)]
#[educe(Debug, Clone)]
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
    pub fn n_variables(&self) -> usize {
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
            eq_evals: Cow::Borrowed(eq_evals),
            input_layer: self,
            eq_fixed_var_correction: SecureField::one(),
            lambda,
        }
    }

    /// Returns a copy of this layer with the [`CpuBackend`].
    ///
    /// This operation is expensive but can be useful for small traces that are difficult to handle
    /// depending on the backend. For example, the SIMD backend offloads to the CPU backend when
    /// trace length becomes smaller than the SIMD lane count.
    pub fn to_cpu(&self) -> Layer<CpuBackend> {
        match self {
            Layer::GrandProduct(mle) => Layer::GrandProduct(Mle::new(mle.to_cpu())),
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => Layer::LogUpGeneric {
                numerators: Mle::new(numerators.to_cpu()),
                denominators: Mle::new(denominators.to_cpu()),
            },
            Layer::LogUpMultiplicities {
                numerators,
                denominators,
            } => Layer::LogUpMultiplicities {
                numerators: Mle::new(numerators.to_cpu()),
                denominators: Mle::new(denominators.to_cpu()),
            },
            Layer::LogUpSingles { denominators } => Layer::LogUpSingles {
                denominators: Mle::new(denominators.to_cpu()),
            },
        }
    }
}

#[derive(Debug)]
struct NotOutputLayerError;

/// Multivariate polynomial `P` that expresses the relation between two consecutive GKR layers.
///
/// When the input layer is [`Layer::GrandProduct`] (represented by multilinear column `inp`)
/// the polynomial represents:
///
/// ```text
/// P(x) = eq(x, y) * inp(x, 0) * inp(x, 1)
/// ```
///
/// When the input layer is LogUp (represented by multilinear columns `inp_numer` and
/// `inp_denom`) the polynomial represents:
///
/// ```text
/// numer(x) = inp_numer(x, 0) * inp_denom(x, 1) + inp_numer(x, 1) * inp_denom(x, 0)
/// denom(x) = inp_denom(x, 0) * inp_denom(x, 1)
///
/// P(x) = eq(x, y) * (numer(x) + lambda * denom(x))
/// ```
pub struct GkrMultivariatePolyOracle<'a, B: GkrOps> {
    /// `eq_evals` passed by `Layer::into_multivariate_poly()`.
    pub eq_evals: Cow<'a, EqEvals<B>>,
    pub input_layer: Layer<B>,
    pub eq_fixed_var_correction: SecureField,
    /// Used by LogUp to perform a random linear combination of the numerators and denominators.
    pub lambda: SecureField,
}

impl<B: GkrOps> MultivariatePolyOracle for GkrMultivariatePolyOracle<'_, B> {
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

    /// Returns a copy of this oracle with the [`CpuBackend`].
    ///
    /// This operation is expensive but can be useful for small oracles that are difficult to handle
    /// depending on the backend. For example, the SIMD backend offloads to the CPU backend when
    /// trace length becomes smaller than the SIMD lane count.
    pub fn to_cpu(&self) -> GkrMultivariatePolyOracle<'a, CpuBackend> {
        // TODO(andrew): This block is not ideal.
        let n_eq_evals = 1 << (self.n_variables() - 1);
        let eq_evals = Cow::Owned(EqEvals {
            evals: Mle::new((0..n_eq_evals).map(|i| self.eq_evals.at(i)).collect()),
            y: self.eq_evals.y.to_vec(),
        });

        GkrMultivariatePolyOracle {
            eq_evals,
            eq_fixed_var_correction: self.eq_fixed_var_correction,
            input_layer: self.input_layer.to_cpu(),
            lambda: self.lambda,
        }
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
            channel.mix_felts(mask.columns().as_flattened());
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

/// Computes `r(t) = sum_x eq((t, x), y[-k:]) * p(t, x)` from evaluations of
/// `f(t) = sum_x eq(({0}^(n - k), 0, x), y) * p(t, x)`.
///
/// Note `claim` must equal `r(0) + r(1)` and `r` must have degree <= 3.
///
/// For more context see `Layer::into_multivariate_poly()` docs.
/// See also <https://ia.cr/2024/108> (section 3.2).
pub fn correct_sum_as_poly_in_first_variable(
    f_at_0: SecureField,
    f_at_2: SecureField,
    claim: SecureField,
    y: &[SecureField],
    k: usize,
) -> UnivariatePoly<SecureField> {
    assert_ne!(k, 0);
    let n = y.len();
    assert!(k <= n);

    // We evaluated `f(0)` and `f(2)` - the inputs.
    // We want to compute `r(t) = f(t) * eq(t, y[n - k]) / eq(0, y[:n - k + 1])`.
    let a_const = eq(&vec![SecureField::zero(); n - k + 1], &y[..n - k + 1]).inverse();

    // Find the additional root of `r(t)`, by finding the root of `eq(t, y[n - k])`:
    //    0 = eq(t, y[n - k])
    //      = t * y[n - k] + (1 - t)(1 - y[n - k])
    //      = 1 - y[n - k] - t(1 - 2 * y[n - k])
    // => t = (1 - y[n - k]) / (1 - 2 * y[n - k])
    //      = b
    let b_const = (SecureField::one() - y[n - k]) / (SecureField::one() - y[n - k].double());

    // We get that `r(t) = f(t) * eq(t, y[n - k]) * a`.
    let r_at_0 = f_at_0 * eq(&[SecureField::zero()], &[y[n - k]]) * a_const;
    let r_at_1 = claim - r_at_0;
    let r_at_2 = f_at_2 * eq(&[BaseField::from(2).into()], &[y[n - k]]) * a_const;
    let r_at_b = SecureField::zero();

    // Interpolate.
    UnivariatePoly::interpolate_lagrange(
        &[
            SecureField::zero(),
            SecureField::one(),
            SecureField::from(BaseField::from(2)),
            b_const,
        ],
        &[r_at_0, r_at_1, r_at_2, r_at_b],
    )
}
