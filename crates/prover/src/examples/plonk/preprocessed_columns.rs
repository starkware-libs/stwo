// TODO(Gali): Add documentation.
#[derive(Debug)]
pub struct Plonk {
    pub wire: usize,
}
impl Plonk {
    pub const fn new(wire: usize) -> Self {
        Self { wire }
    }

    pub fn id(&self) -> String {
        format!("preprocessed_plonk_{}", self.wire)
    }
}
