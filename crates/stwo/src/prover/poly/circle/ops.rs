#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{CircleCoefficients, CircleEvaluation};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::core::ColumnVec;
use crate::prover::air::component_prover::Poly;
use crate::prover::backend::{Col, ColumnOps};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;

/// Operations on BaseField polynomials.
pub trait PolyOps: ColumnOps<BaseField> + ColumnOps<SecureField> + Sized {
    // TODO(alont): Use a column instead of this type.
    /// The type for precomputed twiddles.
    type Twiddles;

    /// Computes a minimal [CircleCoefficients] that evaluates to the same values as this
    /// evaluation. Used by the [`CircleEvaluation::interpolate()`] function.
    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        itwiddles: &TwiddleTree<Self>,
    ) -> CircleCoefficients<Self>;

    fn interpolate_columns(
        columns: Vec<CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CircleCoefficients<Self>> {
        #[cfg(feature = "parallel")]
        let iter = columns.into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = columns.into_iter();

        iter.map(|eval| eval.interpolate_with_twiddles(twiddles))
            .collect()
    }

    /// Evaluates the polynomial at a single point.
    /// Used by the [`CircleCoefficients::eval_at_point()`] function.
    fn eval_at_point(
        poly: &CircleCoefficients<Self>,
        point: CirclePoint<SecureField>,
    ) -> SecureField;

    /// Computes the weights for Barycentric Lagrange interpolation for point `p` on `coset`.
    /// `p` must not be in the domain.
    /// Used by the [`CircleEvaluation::barycentric_weights()`] function.
    fn barycentric_weights(
        coset: CanonicCoset,
        p: CirclePoint<SecureField>,
    ) -> Col<Self, SecureField>;

    /// Evaluates a polynomial at a point using the barycentric interpolation formula,
    /// given its evaluations on a circle domain and precomputed barycentric weights for the domain
    /// at the sampled point.
    /// Used by the [`CircleEvaluation::barycentric_eval_at_point()`] function.
    fn barycentric_eval_at_point(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        weights: &Col<Self, SecureField>,
    ) -> SecureField;

    /// Evaluates a polynomial, represented by it's evaluations, at a point using folding.
    /// Used by the [`CircleEvaluation::eval_at_point_by_folding()`] function.
    fn eval_at_point_by_folding(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        point: CirclePoint<SecureField>,
        twiddles: &TwiddleTree<Self>,
    ) -> SecureField;

    /// Extends the polynomial to a larger degree bound.
    /// Used by the [`CircleCoefficients::extend()`] function.
    fn extend(poly: &CircleCoefficients<Self>, log_size: u32) -> CircleCoefficients<Self>;

    /// Evaluates the polynomial at all points in the domain.
    /// Used by the [`CircleCoefficients::evaluate()`] function.
    fn evaluate(
        poly: &CircleCoefficients<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder>;

    fn evaluate_polynomials(
        polynomials: ColumnVec<CircleCoefficients<Self>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<Self>,
        store_polynomials_coefficients: bool,
    ) -> Vec<Poly<Self>>
    where
        Self: crate::prover::backend::Backend,
    {
        #[cfg(feature = "parallel")]
        let iter = polynomials.into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = polynomials.into_iter();

        iter.map(|poly_coeffs| {
            let evals = poly_coeffs.evaluate_with_twiddles(
                CanonicCoset::new(poly_coeffs.log_size() + log_blowup_factor).circle_domain(),
                twiddles,
            );
            Poly::new(store_polynomials_coefficients.then_some(poly_coeffs), evals)
        })
        .collect()
    }

    /// Precomputes twiddles for a given coset.
    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self>;

    /// Given a polynomial `p`, it outputs two polynomials `p_left`, `p_right` of half the degree,
    /// which satisfy the identity
    ///
    /// `p(z) = p_left(z) + pi^{L-2}(z.x) * p_right(z)`.
    ///
    /// where `L` is the log size of the coefficient vector and `z` is a circle point.
    /// If a polynomial is given by its vector of coefficients (in terms of the FFT basis in natural
    /// order), this decomposition corresponds exactly to dividing the coefficient vector in the
    /// middle. In fact, for `n` in `[0, 2^L)`, the basis element corresponding to the n-th
    /// coefficient is
    ///
    /// `(pi^{L-2}(x))^b_{L-1} * ... * (pi(x))^b_2 * x^b_1* y^b_0`,
    ///
    /// where `b_{L-1}, ... , b_0` is the bit decomposition of n (from most to least significant
    /// bit). Therefore, splitting the coefficient vector in the middle, corresponds to separating
    /// the ones with the MSB, b_{L-1} == 1, from the ones with the MSB, b_{L-1} == 0, meaning
    /// separating the basis elements divisible by `pi^{L-2}(x)` from those that are not.
    fn split_at_mid(
        poly: CircleCoefficients<Self>,
    ) -> (CircleCoefficients<Self>, CircleCoefficients<Self>);
}
