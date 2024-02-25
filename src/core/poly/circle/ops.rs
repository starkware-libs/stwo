use super::{CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::{Col, ExtensionOf, FieldOps};
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::BitReversedOrder;

pub trait PolyOps<F: ExtensionOf<BaseField>>: FieldOps<F> + Sized {
    /// Creates a [CircleEvaluation] from values ordered according to [CanonicCoset].
    /// Used by the [`CircleEvaluation::new_canonical_ordered()`] function.
    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, F>,
    ) -> CircleEvaluation<Self, F, BitReversedOrder>;

    /// Computes a minimal [CirclePoly] that evaluates to the same values as this evaluation.
    /// Used by the [`CircleEvaluation::interpolate()`] function.
    fn interpolate(
        eval: CircleEvaluation<Self, F, BitReversedOrder>,
        itwiddles: &TwiddleTree<Self, F>,
    ) -> CirclePoly<Self, F>;

    /// Evaluates the polynomial at a single point.
    /// Used by the [`CirclePoly::eval_at_point()`] function.
    fn eval_at_point<E: ExtensionOf<F>>(poly: &CirclePoly<Self, F>, point: CirclePoint<E>) -> E;

    /// Extends the polynomial to a larger degree bound.
    /// Used by the [`CirclePoly::extend()`] function.
    fn extend(poly: &CirclePoly<Self, F>, log_size: u32) -> CirclePoly<Self, F>;

    /// Evaluates the polynomial at all points in the domain.
    /// Used by the [`CirclePoly::evaluate()`] function.
    fn evaluate(
        poly: &CirclePoly<Self, F>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self, F>,
    ) -> CircleEvaluation<Self, F, BitReversedOrder>;

    type Twiddles;
    /// Precomputes twiddles for a given coset.
    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self, F>;
}
