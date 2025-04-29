use itertools::Itertools;

use super::{CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly};
use crate::core::backend::{Col, ColumnOps};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

/// Operations on BaseField polynomials.
pub trait PolyOps: ColumnOps<BaseField> + ColumnOps<SecureField> + Sized {
    // TODO(alont): Use a column instead of this type.
    /// The type for precomputed twiddles.
    type Twiddles;

    /// Computes a minimal [CirclePoly] that evaluates to the same values as this evaluation.
    /// Used by the [`CircleEvaluation::interpolate()`] function.
    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        itwiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self>;

    fn interpolate_columns(
        columns: impl IntoIterator<Item = CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CirclePoly<Self>> {
        columns
            .into_iter()
            .map(|eval| eval.interpolate_with_twiddles(twiddles))
            .collect()
    }

    /// Evaluates the polynomial at a single point.
    /// Used by the [`CirclePoly::eval_at_point()`] function.
    fn eval_at_point(poly: &CirclePoly<Self>, point: CirclePoint<SecureField>) -> SecureField;

    /// Computes the weights for Barycentric Lagrange interpolation for point `p` on the canonic
    /// coset of size `log_size`.
    /// Used by the [`CircleEvaluation::barycentric_weights()`] function.
    fn barycentric_weights(
        log_size: u32,
        sample_point: CirclePoint<SecureField>,
    ) -> Col<Self, SecureField>;

    /// Evaluates a polynomial at a point using the barycentric interpolation formula.
    /// Used by the [`CircleEvaluation::barycentric_eval_at_point()`] function.
    fn barycentric_eval_at_point(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        point: CirclePoint<SecureField>,
        weights: &Col<Self, SecureField>,
    ) -> SecureField;

    /// Extends the polynomial to a larger degree bound.
    /// Used by the [`CirclePoly::extend()`] function.
    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self>;

    /// Evaluates the polynomial at all points in the domain.
    /// Used by the [`CirclePoly::evaluate()`] function.
    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder>;

    fn evaluate_polynomials(
        polynomials: &ColumnVec<CirclePoly<Self>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CircleEvaluation<Self, BaseField, BitReversedOrder>> {
        polynomials
            .iter()
            .map(|poly| {
                poly.evaluate_with_twiddles(
                    CanonicCoset::new(poly.log_size() + log_blowup_factor).circle_domain(),
                    twiddles,
                )
            })
            .collect_vec()
    }

    /// Precomputes twiddles for a given coset.
    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self>;
}
