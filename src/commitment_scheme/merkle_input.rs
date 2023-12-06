use crate::core::fields::Field;

/// The Input of a Merkle-Tree Mixed-Degree commitment scheme.
/// A map from the depth of the tree requested to be injected to the to-be-injected columns.
///
/// # Example
///
/// ```rust
/// use prover_research::commitment_scheme::merkle_input::MerkleTreeInput;
/// use prover_research::core::fields::m31::M31;
///
/// let mut input = MerkleTreeInput::<M31>::new();
/// let column = vec![M31::from_u32_unchecked(0); 1024];
/// input.insert_column(2, &column);
/// input.insert_column(3, &column);
/// input.insert_column(3, &column);
///
/// assert_eq!(input.get_columns(2).unwrap().len(), 1);
/// assert_eq!(input.get_columns(3).unwrap().len(), 2);
/// assert_eq!(input.max_injected_depth(), 3);
/// ````
#[derive(Default)]
pub struct MerkleTreeInput<'a, F: Field> {
    columns_to_inject: Vec<LayerColumns<'a, F>>,
}

pub type LayerColumns<'a, F> = Vec<&'a [F]>;

impl<'a, F: Field> MerkleTreeInput<'a, F> {
    pub fn new() -> Self {
        Self {
            columns_to_inject: vec![],
        }
    }

    pub fn insert_column(&mut self, depth: usize, column: &'a [F]) {
        assert_ne!(depth, 0, "Injection to layer 0 undefined!");
        assert!(
            column.len().is_power_of_two(),
            "Column is of size: {}, not a power of 2!",
            column.len()
        );

        // TODO(Ohad): implement embedd by repeatition and remove assert.
        assert!(
            column.len() >= 2usize.pow(depth as u32),
            "Column of size: {} is too small for injection at layer:{}",
            column.len(),
            depth
        );

        if self.columns_to_inject.len() < depth {
            self.columns_to_inject.resize(depth, vec![]);
        }
        self.columns_to_inject[depth - 1].push(column);
    }

    pub fn get_columns(&'a self, depth: usize) -> Option<&'a LayerColumns<'a, F>> {
        self.columns_to_inject.get(depth - 1)
    }

    pub fn max_injected_depth(&self) -> usize {
        self.columns_to_inject.len()
    }

    /// Splits the input into two parts, the first part is the input for the first layer, the second
    /// part is the input for the deeper layers.
    pub fn split(&mut self, split_at: usize) -> Self {
        assert!(split_at > 0);
        assert!(split_at <= self.max_injected_depth());
        Self {
            columns_to_inject: self.columns_to_inject.split_off(split_at - 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::fields::m31::M31;

    #[test]
    pub fn md_input_insert_test() {
        let mut input = super::MerkleTreeInput::<M31>::new();
        let column = vec![M31::from_u32_unchecked(0); 1024];

        input.insert_column(3, &column);
        input.insert_column(3, &column);
        input.insert_column(2, &column);

        assert_eq!(input.get_columns(3).unwrap().len(), 2);
        assert_eq!(input.get_columns(2).unwrap().len(), 1);
    }

    #[test]
    pub fn md_input_max_depth_test() {
        let mut input = super::MerkleTreeInput::<M31>::new();
        let column = vec![M31::from_u32_unchecked(0); 1024];

        input.insert_column(3, &column);
        input.insert_column(2, &column);

        assert_eq!(input.max_injected_depth(), 3);
    }

    #[test]
    #[should_panic]
    pub fn mt_input_column_too_short_test() {
        let mut input = super::MerkleTreeInput::<M31>::new();
        let column = vec![M31::from_u32_unchecked(0); 1024];

        input.insert_column(11, &column);
    }

    #[test]
    #[should_panic]
    pub fn mt_input_wrong_size_test() {
        let mut input = super::MerkleTreeInput::<M31>::default();
        let not_pow_2_column = vec![M31::from_u32_unchecked(0); 1023];

        input.insert_column(2, &not_pow_2_column);
    }

    #[test]
    pub fn test_split() {
        let mut input = super::MerkleTreeInput::<M31>::new();
        let column = vec![M31::from_u32_unchecked(0); 1024];
        input.insert_column(3, column.as_ref());
        let input_for_deeper_layers = input.split(2);
        assert_eq!(input.max_injected_depth(), 1);
        assert_eq!(input_for_deeper_layers.max_injected_depth(), 2);
    }
}
