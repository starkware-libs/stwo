use tracing::{span, Level};

use crate::constraint_framework::preprocessed_columns::gen_is_first;
use crate::core::backend::simd::column::BaseColumn;
use crate::core::backend::simd::SimdBackend;
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

// TODO(Gali): Add documentation and remove allow dead code.
#[allow(dead_code)]
#[derive(Debug)]
pub struct XorTable {
    pub n_bits: u32,
    pub n_expand_bits: u32,
    pub index_in_table: usize,
}
impl XorTable {
    pub const fn new(n_bits: u32, n_expand_bits: u32, index_in_table: usize) -> Self {
        Self {
            n_bits,
            n_expand_bits,
            index_in_table,
        }
    }

    pub fn id(&self) -> String {
        format!(
            "preprocessed_xor_table_{}_{}_{}",
            self.n_bits, self.n_expand_bits, self.index_in_table
        )
    }

    // TODO(Gali): Remove allow dead code.
    #[allow(dead_code)]
    pub const fn limb_bits(&self) -> u32 {
        self.n_bits - self.n_expand_bits
    }

    // TODO(Gali): Remove allow dead code.
    #[allow(dead_code)]
    pub const fn column_bits(&self) -> u32 {
        2 * self.limb_bits()
    }

    /// Generates the Preprocessed trace for the xor table.
    /// Returns the Preprocessed trace, the Preprocessed trace, and the claimed sum.
    // TODO(Gali): Remove allow dead code.
    #[allow(dead_code)]
    #[allow(clippy::type_complexity)]
    pub fn generate_constant_trace(
        &self,
    ) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        let limb_bits = self.limb_bits();
        let _span = span!(Level::INFO, "Xor Preprocessed trace").entered();

        // Generate the constant columns. In reality, these should be generated before the
        // proof even began.
        let a_col: BaseColumn = (0..(1 << self.column_bits()))
            .map(|i| BaseField::from_u32_unchecked((i >> limb_bits) as u32))
            .collect();
        let b_col: BaseColumn = (0..(1 << self.column_bits()))
            .map(|i| BaseField::from_u32_unchecked((i & ((1 << limb_bits) - 1)) as u32))
            .collect();
        let c_col: BaseColumn = (0..(1 << self.column_bits()))
            .map(|i| {
                BaseField::from_u32_unchecked(
                    ((i >> limb_bits) ^ (i & ((1 << limb_bits) - 1))) as u32,
                )
            })
            .collect();

        let mut constant_trace = [a_col, b_col, c_col]
            .map(|x| {
                CircleEvaluation::new(CanonicCoset::new(self.column_bits()).circle_domain(), x)
            })
            .to_vec();
        constant_trace.push(gen_is_first(self.column_bits()));
        constant_trace
    }
}
