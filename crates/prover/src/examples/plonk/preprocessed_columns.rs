use crate::constraint_framework::preprocessed_columns::PreprocessedColumn;

// TODO(Gali): Add documentation.
#[derive(Debug)]
pub struct Plonk {
    pub wire: usize,
}
impl Plonk {
    pub const fn new(wire: usize) -> Self {
        Self { wire }
    }
}
impl PreprocessedColumn for Plonk {
    fn name(&self) -> &'static str {
        "preprocessed_plonk"
    }

    fn id(&self) -> String {
        format!("Plonk(wire: {})", self.wire)
    }

    fn log_size(&self) -> u32 {
        todo!("Plonk::log_size")
    }
}
