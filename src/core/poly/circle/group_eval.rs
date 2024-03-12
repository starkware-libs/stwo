use super::{CircleEvaluation, Group};
use crate::core::circle::CirclePointIndex;
use crate::core::fields::m31::BaseField;
use crate::core::fields::{Col, Column, FieldOps};
use crate::core::poly::BitReversedOrder;

// TODO(spapini): Reference the group line domain here.
/// An evaluation defined on a [CircleDomain].
/// The values are ordered according to the group line domain ordering.
#[derive(Clone, Debug)]
pub struct CircleGroupEvaluation<B: FieldOps<BaseField>> {
    pub domain: Group,
    pub values: Col<B, BaseField>,
}

impl<B: FieldOps<BaseField>> CircleGroupEvaluation<B> {
    pub fn new(domain: Group, values: Col<B, BaseField>) -> Self {
        assert_eq!(domain.size(), values.len());
        // TODO: get eval_at_minus_1 from invariant.
        Self { domain, values }
    }

    pub fn get_at(&self, _point_index: CirclePointIndex) -> BaseField {
        todo!();
    }

    pub fn extend_with_canonic(&mut self, _eval: CircleEvaluation<B, BaseField, BitReversedOrder>) {
        // TODO: asserts.
        // TODO: extend values.
        todo!();
    }
}
