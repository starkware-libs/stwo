use super::{CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::{Col, ExtensionOf, FieldOps};
use crate::core::poly::BitReversedOrder;

pub trait PolyOps: FieldOps<BaseField> + Sized {
    /// Creates a [CircleEvaluation] from values ordered according to [CanonicCoset].
    /// Used by the [`CircleEvaluation::new_canonical_ordered()`] function.
    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder>;

    /// Computes a minimal [CirclePoly] that evaluates to the same values as this evaluation.
    /// Used by the [`CircleEvaluation::interpolate()`] function.
    fn interpolate(eval: CircleEvaluation<Self, BaseField, BitReversedOrder>) -> CirclePoly<Self>;

    /// Evaluates the polynomial at a single point.
    /// Used by the [`CirclePoly::eval_at_point()`] function.
    fn eval_at_point<E: ExtensionOf<BaseField>>(
        poly: &CirclePoly<Self>,
        point: CirclePoint<E>,
    ) -> E;

    /// Extends the polynomial to a larger degree bound.
    /// Used by the [`CirclePoly::extend()`] function.
    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self>;

    /// Evaluates the polynomial at all points in the domain.
    /// Used by the [`CirclePoly::evaluate()`] function.
    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder>;
}
