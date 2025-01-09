// TODO(Gali): Add documentation and remove allow dead code.
#[allow(dead_code)]
#[derive(Debug)]
pub struct XorTable {
    pub n_bits: u32,
    pub n_expand_bits: u32,
    pub index_in_table: usize,
}
impl XorTable {
    // TODO(Gali): Remove allow dead code.
    #[allow(dead_code)]
    pub const fn new(n_bits: u32, n_expand_bits: u32, index_in_table: usize) -> Self {
        Self {
            n_bits,
            n_expand_bits,
            index_in_table,
        }
    }

    // TODO(Gali): Remove allow dead code.
    #[allow(dead_code)]
    pub fn id(&self) -> String {
        format!(
            "preprocessed_xor_table_{}_{}_{}",
            self.n_bits, self.n_expand_bits, self.index_in_table
        )
    }
}
