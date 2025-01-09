use crate::constraint_framework::preprocessed_columns::PreprocessedColumnTrait;

// TODO(Gali): Add documentation.
#[derive(Debug)]
pub struct XorTable {
    pub n_bits: u32,
    pub n_expand_bits: u32,
    pub index_in_table: usize,
}
impl XorTable {
    //TODO(Gali): Remove allow dead code.
    #[allow(dead_code)]
    pub const fn new(n_bits: u32, n_expand_bits: u32, index_in_table: usize) -> Self {
        Self {
            n_bits,
            n_expand_bits,
            index_in_table,
        }
    }
}
impl PreprocessedColumnTrait for XorTable {
    fn name(&self) -> &'static str {
        "preprocessed_xor_table"
    }

    fn id(&self) -> String {
        format!(
            "XorTable(n_bits: {}, n_expand_bits: {}, index_in_table: {})",
            self.n_bits, self.n_expand_bits, self.index_in_table
        )
    }

    fn log_size(&self) -> u32 {
        2 * (self.n_bits - self.n_expand_bits)
    }
}