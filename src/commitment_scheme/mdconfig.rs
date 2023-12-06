use std::collections::BTreeMap;

use thiserror::Error;

use crate::core::fields::Field;

/// The configuration of a Merkle-Tree Mixed-Degree commitment scheme.
/// A map from the depth of the tree requested to be injected to the to-be-injected columns.
///
/// # Example
///
/// ```
/// use prover_research::commitment_scheme::mdconfig::MDConfig;
/// use prover_research::core::fields::m31::M31;
///
/// let mut config = MDConfig::<M31>::default();
/// let column = vec![M31::from_u32_unchecked(0); 1024];
/// config.insert(2, &column).unwrap();
/// config.insert(3, &column).unwrap();
/// config.insert(3, &column).unwrap();
///
/// assert_eq!(config.get(2).unwrap().len(), 1);
/// assert_eq!(config.get(3).unwrap().len(), 2);
/// assert_eq!(config.max_depth, 3);
/// ````
#[derive(Default)]
pub struct MDConfig<'a, F: Field> {
    map: BTreeMap<usize, Vec<&'a Vec<F>>>,
    pub max_depth: usize,
}

impl<'a, F: Field> MDConfig<'a, F> {
    pub fn insert(&mut self, depth: usize, column: &'a Vec<F>) -> Result<(), MDConfigError> {
        if !column.len().is_power_of_two() {
            return Err(MDConfigError::ColumnSizeNotPowerOfTwo);
        }
        if column.len() < 2usize.pow(depth as u32) {
            return Err(MDConfigError::ColumnTooSmall);
        }

        match self.map.get_mut(&depth) {
            Some(c) => {
                c.push(column);
            }
            None => {
                self.map.insert(depth, vec![column]);
            }
        }
        self.max_depth = std::cmp::max(self.max_depth, depth);

        Ok(())
    }

    pub fn get(&'a self, depth: usize) -> Option<&'a Vec<&'a Vec<F>>> {
        self.map.get(&depth)
    }
}

#[derive(Error, Debug, PartialEq)]
pub enum MDConfigError {
    #[error("The requested depth is too large for that column.")]
    ColumnTooSmall,
    #[error("The inserted column's size is not a power of 2.")]
    ColumnSizeNotPowerOfTwo,
}

#[cfg(test)]
mod tests {
    use crate::core::fields::m31::M31;

    #[test]
    pub fn md_config_insert_test() {
        let mut config = super::MDConfig::<M31>::default();
        let column = vec![M31::from_u32_unchecked(0); 1024];

        config.insert(3, &column).unwrap();
        config.insert(3, &column).unwrap();
        config.insert(2, &column).unwrap();

        assert_eq!(config.get(3).unwrap().len(), 2);
        assert_eq!(config.get(2).unwrap().len(), 1);
    }

    #[test]
    pub fn md_config_max_depth_test() {
        let mut config = super::MDConfig::<M31>::default();
        let column = vec![M31::from_u32_unchecked(0); 1024];

        config.insert(3, &column).unwrap();
        config.insert(2, &column).unwrap();

        assert_eq!(config.max_depth, 3);
    }

    #[test]
    pub fn md_config_extract_test() {
        let mut config = super::MDConfig::<M31>::default();
        let column = vec![M31::from_u32_unchecked(0); 1024];
        let not_pow_2_column = vec![M31::from_u32_unchecked(0); 1023];

        config.insert(3, &column).unwrap();
        let column_too_small_result = config.insert(11, &column).unwrap_err();
        let column_not_power_of_two_result = config.insert(2, &not_pow_2_column).unwrap_err();

        assert_eq!(
            column_too_small_result,
            super::MDConfigError::ColumnTooSmall
        );
        assert_eq!(
            column_not_power_of_two_result,
            super::MDConfigError::ColumnSizeNotPowerOfTwo
        );
    }
}
