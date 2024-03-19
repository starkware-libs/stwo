use crate::core::air::evaluation::SecureColumn;
use crate::core::backend::avx512::AVX512Backend;
use crate::core::backend::CPUBackend;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr::GkrOps;

impl GkrOps for AVX512Backend {
    type EqEvals = SecureColumn<Self>;

    fn gen_eq_evals(y: &[SecureField]) -> SecureColumn<Self> {
        // TODO: Implement AVX version
        CPUBackend::gen_eq_evals(y).into_iter().collect()
    }
}
