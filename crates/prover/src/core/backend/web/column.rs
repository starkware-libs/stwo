use super::WebBackend;
use crate::core::backend::simd::column::BaseColumn;
use crate::core::backend::CpuBackend;
use crate::core::secure_column::SecureColumnByCoords;

impl SecureColumnByCoords<WebBackend> {
    pub fn from_cpu(cpu: SecureColumnByCoords<CpuBackend>) -> Self {
        Self {
            columns: cpu.columns.map(BaseColumn::from_cpu),
        }
    }
}
