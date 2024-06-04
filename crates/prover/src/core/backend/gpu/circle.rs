use super::GpuBackend;
use crate::core::backend::Col;
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::BitReversedOrder;

impl PolyOps for GpuBackend {
    // TODO: This type may need to be changed
    type Twiddles = Vec<BaseField>;

    fn new_canonical_ordered(
        _coset: CanonicCoset,
        _values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        todo!()
    }

    fn interpolate(
        _eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        _itwiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        todo!()
    }

    fn eval_at_point(_poly: &CirclePoly<Self>, _point: CirclePoint<SecureField>) -> SecureField {
        todo!()
    }

    fn extend(_poly: &CirclePoly<Self>, _log_size: u32) -> CirclePoly<Self> {
        todo!()
    }

    fn evaluate(
        _poly: &CirclePoly<Self>,
        _domain: CircleDomain,
        _twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        todo!()
    }

    fn precompute_twiddles(_coset: Coset) -> TwiddleTree<Self> {
        todo!()
    }
}
