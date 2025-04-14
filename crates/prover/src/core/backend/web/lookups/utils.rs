use crate::core::backend::simd::SimdBackend;
use crate::core::backend::web::WebBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::QM31;
use crate::core::lookups::gkr_prover::{GkrMultivariatePolyOracle, Layer};
use crate::core::lookups::mle::Mle;

// WARNING: This works because they are literally the same object layout.
//
// The only difference is the backend methods.
// When we implement all methods for WebGPU,
// we will no longer need this to convert back/forth.
impl AsRef<Layer<SimdBackend>> for Layer<WebBackend> {
    fn as_ref(&self) -> &Layer<SimdBackend> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl<'a> AsRef<GkrMultivariatePolyOracle<'a, SimdBackend>>
    for GkrMultivariatePolyOracle<'a, WebBackend>
{
    fn as_ref(&self) -> &GkrMultivariatePolyOracle<'a, SimdBackend> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl Into<Layer<WebBackend>> for Layer<SimdBackend> {
    fn into(self) -> Layer<WebBackend> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl Into<Mle<SimdBackend, BaseField>> for Mle<WebBackend, BaseField> {
    fn into(self) -> Mle<SimdBackend, BaseField> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl Into<Mle<SimdBackend, QM31>> for Mle<WebBackend, QM31> {
    fn into(self) -> Mle<SimdBackend, QM31> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl Into<Mle<WebBackend, BaseField>> for Mle<SimdBackend, BaseField> {
    fn into(self) -> Mle<WebBackend, BaseField> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}

impl Into<Mle<WebBackend, QM31>> for Mle<SimdBackend, QM31> {
    fn into(self) -> Mle<WebBackend, QM31> {
        assert_eq!(std::mem::size_of::<SimdBackend>(), 0);
        assert_eq!(std::mem::size_of::<WebBackend>(), 0);
        unsafe { std::mem::transmute(self) }
    }
}
