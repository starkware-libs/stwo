use super::{CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::{Col, ExtensionOf, FieldOps};

pub trait PolyOps<F: ExtensionOf<BaseField>>: FieldOps<F> + Sized {
    /// Creates a [CircleEvaluation] from values ordered according to [CanonicCoset].
    /// Used by the [`CircleEvaluation::new_canonical_ordered()`] function.
    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, F>,
    ) -> CircleEvaluation<Self, F>;

    /// Computes a minimal [CirclePoly] that evaluates to the same values as this evaluation.
    /// Used by the [`CircleEvaluation::interpolate()`] function.
    fn interpolate(eval: CircleEvaluation<Self, F>) -> CirclePoly<Self, F>;

    /// Evaluates the polynomial at a single point.
    /// Used by the [`CirclePoly::eval_at_point()`] function.
    fn eval_at_point<E: ExtensionOf<F>>(poly: &CirclePoly<Self, F>, point: CirclePoint<E>) -> E;

    /// Extends the polynomial to a larger degree bound.
    /// Used by the [`CirclePoly::extend()`] function.
    fn extend(poly: &CirclePoly<Self, F>, log_size: u32) -> CirclePoly<Self, F>;

    /// Evaluates the polynomial at all points in the domain.
    /// Used by the [`CirclePoly::evaluate()`] function.
    fn evaluate(poly: &CirclePoly<Self, F>, domain: CircleDomain) -> CircleEvaluation<Self, F>;
}
