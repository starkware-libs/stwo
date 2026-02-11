use std_shims::Vec;

/// # Safety
///
/// The caller must ensure that the vector is initialized before use.
#[allow(clippy::uninit_vec)]
pub unsafe fn uninit_vec<T>(len: usize) -> Vec<T> {
    let mut vec = Vec::with_capacity(len);
    vec.set_len(len);
    vec
}
