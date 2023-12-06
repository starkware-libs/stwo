use crate::core::fields::Field;

pub type ColumnRefArray<'a, F> = Vec<&'a Vec<F>>;
pub type ColumnsToInject<'a, F> = Vec<ColumnRefArray<'a, F>>;

/// The configuration of a Merkle-Tree Mixed-Degree commitment scheme.
/// A map from the depth of the tree requested to be injected to the to-be-injected columns.
///
/// # Example
///
/// ```rust
/// use prover_research::commitment_scheme::mdconfig::MerkleTreeInput;
/// use prover_research::core::fields::m31::M31;
///
/// let mut input = MerkleTreeInput::<M31>::new(3);
/// let column = vec![M31::from_u32_unchecked(0); 1024];
/// input.insert(2, &column);
/// input.insert(3, &column);
/// input.insert(3, &column);
///
/// assert_eq!(input.get_columns(2).unwrap().len(), 1);
/// assert_eq!(input.get_columns(3).unwrap().len(), 2);
/// assert_eq!(input.max_injected_depth(), 3);
/// ````
#[derive(Default)]
pub struct MerkleTreeInput<'a, F: Field>(ColumnsToInject<'a, F>);

impl<'a, F: Field> MerkleTreeInput<'a, F> {
    pub fn new(max_depth: usize) -> Self {
        Self(vec![Vec::new(); max_depth])
    }
    pub fn insert(&mut self, depth: usize, column: &'a Vec<F>) {
        assert_ne!(depth, 0, "Injection to layer 0 undefined!");
        assert!(
            column.len().is_power_of_two(),
            "Column is of size: {} ,not a power of 2!",
            column.len()
        );
        assert!(
            column.len() >= 2usize.pow(depth as u32),
            "Column of size: {} is too small for injection at layer:{}",
            column.len(),
            depth
        );
        assert!(
            depth <= self.0.len(),
            "Injection at layer: {} is too deep for the current configuration!",
            depth
        );
        self.0[depth - 1].push(column);
    }

    pub fn get_columns(&'a self, depth: usize) -> Option<&'a ColumnRefArray<'a, F>> {
        self.0.get(depth - 1)
    }

    pub fn max_injected_depth(&self) -> usize {
        self.0.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::core::fields::m31::M31;

    #[test]
    pub fn md_config_insert_test() {
        let mut config = super::MerkleTreeInput::<M31>::new(3);
        let column = vec![M31::from_u32_unchecked(0); 1024];

        config.insert(3, &column);
        config.insert(3, &column);
        config.insert(2, &column);

        assert_eq!(config.get_columns(3).unwrap().len(), 2);
        assert_eq!(config.get_columns(2).unwrap().len(), 1);
    }

    #[test]
    pub fn md_config_max_depth_test() {
        let mut config = super::MerkleTreeInput::<M31>::new(3);
        let column = vec![M31::from_u32_unchecked(0); 1024];

        config.insert(3, &column);
        config.insert(2, &column);

        assert_eq!(config.max_injected_depth(), 3);
    }

    #[test]
    #[should_panic]
    pub fn mt_input_column_too_short_test() {
        let mut config = super::MerkleTreeInput::<M31>::new(11);
        let column = vec![M31::from_u32_unchecked(0); 1024];

        config.insert(11, &column);
    }

    #[test]
    #[should_panic]
    pub fn mt_input_wrong_size_test() {
        let mut config = super::MerkleTreeInput::<M31>::default();
        let not_pow_2_column = vec![M31::from_u32_unchecked(0); 1023];

        config.insert(2, &not_pow_2_column);
    }
}
