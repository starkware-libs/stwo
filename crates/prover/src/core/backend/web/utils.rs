use super::WebBackend;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{BaseField, Col};

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
