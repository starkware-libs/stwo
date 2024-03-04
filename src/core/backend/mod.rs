use std::fmt::Debug;

pub use cpu::CPUBackend;

use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::fields::FieldOps;
use super::poly::circle::PolyOps;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub mod avx512;
pub mod cpu;

pub trait Backend:
    Copy
    + Clone
    + Debug
    + FieldOps<BaseField>
    + FieldOps<SecureField>
    + PolyOps<BaseField>
    + PolyOps<SecureField>
{
}
