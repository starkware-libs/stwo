use super::WebBackend;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{BaseField, Col};
use crate::core::poly::circle::{CircleEvaluation, CirclePoly, SecureEvaluation};
use crate::core::poly::line::LineEvaluation;
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::BitReversedOrder;
use crate::core::secure_column::SecureColumnByCoords;

// WARNING: This works because they are literally the same object layout.
//
// The only difference is the backend methods.
// When we implement all methods for WebGPU,
// we will no longer need this to convert back/forth.
pub fn transmute_col_refs<'a>(
    input: &'a [&Col<WebBackend, BaseField>],
) -> &'a [&'a Col<SimdBackend, BaseField>] {
    assert_eq!(std::mem::size_of::<WebBackend>(), 0);
    assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
    unsafe {
        std::mem::transmute::<&'a [&Col<WebBackend, BaseField>], &'a [&Col<SimdBackend, BaseField>]>(
            input,
        )
    }
}

impl AsRef<LineEvaluation<SimdBackend>> for LineEvaluation<WebBackend> {
    fn as_ref(&self) -> &LineEvaluation<SimdBackend> {
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl Into<LineEvaluation<WebBackend>> for LineEvaluation<SimdBackend> {
    fn into(self) -> LineEvaluation<WebBackend> {
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl AsMut<LineEvaluation<SimdBackend>> for LineEvaluation<WebBackend> {
    fn as_mut(&mut self) -> &mut LineEvaluation<SimdBackend> {
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl AsRef<SecureEvaluation<SimdBackend, BitReversedOrder>>
    for SecureEvaluation<WebBackend, BitReversedOrder>
{
    fn as_ref(&self) -> &SecureEvaluation<SimdBackend, BitReversedOrder> {
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl Into<SecureEvaluation<WebBackend, BitReversedOrder>>
    for SecureEvaluation<SimdBackend, BitReversedOrder>
{
    fn into(self) -> SecureEvaluation<WebBackend, BitReversedOrder> {
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl AsRef<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>
    for CircleEvaluation<WebBackend, BaseField, BitReversedOrder>
{
    fn as_ref(&self) -> &CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl AsMut<SecureColumnByCoords<SimdBackend>> for SecureColumnByCoords<WebBackend> {
    fn as_mut(&mut self) -> &mut SecureColumnByCoords<SimdBackend> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl AsRef<SecureColumnByCoords<SimdBackend>> for SecureColumnByCoords<WebBackend> {
    fn as_ref(&self) -> &SecureColumnByCoords<SimdBackend> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

pub fn convert_web_to_simd_column(col: Col<WebBackend, BaseField>) -> Col<SimdBackend, BaseField> {
    assert_eq!(std::mem::size_of::<WebBackend>(), 0);
    assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
    unsafe { std::mem::transmute(col) }
}

impl Into<CircleEvaluation<WebBackend, BaseField, BitReversedOrder>>
    for CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>
{
    fn into(self) -> CircleEvaluation<WebBackend, BaseField, BitReversedOrder> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl Into<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>
    for CircleEvaluation<WebBackend, BaseField, BitReversedOrder>
{
    fn into(self) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl AsRef<TwiddleTree<SimdBackend>> for TwiddleTree<WebBackend> {
    fn as_ref(&self) -> &TwiddleTree<SimdBackend> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl AsRef<CirclePoly<SimdBackend>> for CirclePoly<WebBackend> {
    fn as_ref(&self) -> &CirclePoly<SimdBackend> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl<'a> AsRef<CirclePoly<WebBackend>> for &'a CirclePoly<WebBackend> {
    fn as_ref(&self) -> &CirclePoly<WebBackend> {
        self
    }
}

impl<'a> AsRef<CircleEvaluation<WebBackend, BaseField, BitReversedOrder>>
    for &'a CircleEvaluation<WebBackend, BaseField, BitReversedOrder>
{
    fn as_ref(&self) -> &CircleEvaluation<WebBackend, BaseField, BitReversedOrder> {
        self
    }
}

impl Into<CirclePoly<WebBackend>> for CirclePoly<SimdBackend> {
    fn into(self) -> CirclePoly<WebBackend> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl Into<TwiddleTree<WebBackend>> for TwiddleTree<SimdBackend> {
    fn into(self) -> TwiddleTree<WebBackend> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}
