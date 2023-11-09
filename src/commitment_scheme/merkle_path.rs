use std::fmt;

/// Merkle authentication path.
pub struct MerklePath<const NODE_SIZE: usize>(Vec<[u8; NODE_SIZE]>);

impl<const NODE_SIZE: usize> MerklePath<NODE_SIZE> {
    pub fn push(&mut self, val: &[u8; NODE_SIZE]) {
        self.0.push(*val);
    }

    pub fn new(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the number of nodes in the path.
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl<const NODE_SIZE: usize> std::ops::Index<usize> for MerklePath<NODE_SIZE> {
    type Output = [u8; NODE_SIZE];

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.0.len());
        &self.0[index]
    }
}

impl<const NODE_SIZE: usize> fmt::Debug for MerklePath<NODE_SIZE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.last() {
            Some(last_node) => {
                for (i, node) in self.0.iter().enumerate().take(self.0.len() - 1) {
                    f.write_str(&std::format!("Node #[{}]: [", i))?;
                    f.write_str(&(hex::encode(&node[..]) + "]"))?;
                    f.write_str("\n")?;
                }
                f.write_str(&std::format!("Node #[{}]: [", self.0.len() - 1))?;
                f.write_str(&(hex::encode(&last_node[..]) + "]"))?;
                Ok(())
            }
            None => Err(std::fmt::Error),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::{blake3_hash::Blake3Hasher, hasher::Hasher};

    use super::MerklePath;

    #[test]
    pub fn merkle_path_struct_test() {
        const NODE_SIZE: usize = Blake3Hasher::OUTPUT_SIZE_IN_BYTES;

        let mut m_path = MerklePath::<NODE_SIZE>::new(3);
        m_path.push(&[0_u8; NODE_SIZE]);
        m_path.push(&[1_u8; NODE_SIZE]);
        m_path.push(&[2_u8; NODE_SIZE]);

        assert_eq!(m_path.len(), 3);
        assert_eq!(m_path[0], [0_u8; NODE_SIZE]);
        assert_eq!(m_path[1], [1_u8; NODE_SIZE]);
        assert_eq!(m_path[2], [2_u8; NODE_SIZE]);
    }
}
