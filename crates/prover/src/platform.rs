#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
#[allow(unreachable_code)]
pub fn avx512_detected() -> bool {
    // Static check, e.g. for building with target-cpu=native.
    if cfg!(feature = "avx512") {
        return true;
    }

    // Dynamic check, if std is enabled.
    is_x86_feature_detected!("avx512f")
}
