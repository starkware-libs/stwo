use super::{CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly};
use crate::core::backend::Col;
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldOps;
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::BitReversedOrder;

/// Operations on BaseField polynomials.
pub trait PolyOps: FieldOps<BaseField> + Sized {
    // TODO(spapini): Use a column instead of this type.
    /// The type for precomputed twiddles.
    type Twiddles;

    /// Creates a [CircleEvaluation] from values ordered according to [CanonicCoset].
    /// Used by the [`CircleEvaluation::new_canonical_ordered()`] function.
    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder>;

    /// Computes a minimal [CirclePoly] for each evaluation, that evaluates to the same values as
    /// that evaluation.
    /// Used by the [`CircleEvaluation::interpolate()`] function.
    fn interpolate_batch(
        evals: Vec<CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        itwiddles: &TwiddleTree<Self>,
    ) -> Vec<CirclePoly<Self>>;

    /// Evaluates each polynomial at a set of points.
    /// Used by the [`CirclePoly::eval_at_point()`] function.
    fn eval_at_points(
        poly: &[&CirclePoly<Self>],
        points: &[Vec<CirclePoint<SecureField>>],
    ) -> Vec<Vec<SecureField>>;

    /// Extends the polynomial to a larger degree bound.
    /// Used by the [`CirclePoly::extend()`] function.
    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self>;

    /// Evaluates each polynomial at a domain.
    /// Used by the [`CirclePoly::evaluate()`] function.
    fn evaluate_batch(
        poly: &[&CirclePoly<Self>],
        domains: &[CircleDomain],
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CircleEvaluation<Self, BaseField, BitReversedOrder>>;

    /// Precomputes twiddles for a given coset.
    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self>;
}
