use tracing::{span, Level};

use crate::constraint_framework::preprocessed_columns::{IsFirst, PreProcessedColumnId};
use crate::core::backend::simd::column::BaseColumn;
use crate::core::backend::simd::SimdBackend;
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

/// A preprocessed table for the xor operation of 2 n_bits numbers.
/// n_expand_bits is an optimization parameter reducing the table's cloumns' length to
/// 2^(n_bits - n_expand_bits), while storing multiplicities for the n_expand_bits xor operation.
/// The index_in_table is the column index in the preprocessed table.

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

    pub fn id(&self) -> PreProcessedColumnId {
        PreProcessedColumnId {
            id: format!(
                "preprocessed_xor_table_{}_{}_{}",
                self.n_bits, self.n_expand_bits, self.index_in_table
            ),
        }
    }

    pub const fn limb_bits(&self) -> u32 {
        self.n_bits - self.n_expand_bits
    }

    pub const fn column_bits(&self) -> u32 {
        2 * self.limb_bits()
    }

    /// Generates the Preprocessed trace for the xor table.
    /// Returns the Preprocessed trace, the Preprocessed trace, and the claimed sum.
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
        // TODO!(ShaharS): Remove this line.
        constant_trace.push(IsFirst::new(self.column_bits()).gen_column_simd());
        constant_trace
    }
}
