use std::fmt;

/// Merkle authentication path.
pub struct MerklePath<const NODE_SIZE: usize> {
    path: Vec<[u8; NODE_SIZE]>,
}

impl<const NODE_SIZE: usize> MerklePath<NODE_SIZE> {
    pub fn push(&mut self, val: &[u8; NODE_SIZE]) {
        self.path.push(*val);
    }

    pub fn new(capacity: usize) -> Self {
        Self {
            path: Vec::with_capacity(capacity),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.path.is_empty()
    }

    /// Returns the number of nodes in the path.
    /// Does not include the root.
    pub fn len(&self) -> usize {
        self.path.len()
    }
}

impl<const NODE_SIZE: usize> std::ops::Index<usize> for MerklePath<NODE_SIZE> {
    type Output = [u8; NODE_SIZE];

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.path.len());
        &self.path[index]
    }
}

impl<const NODE_SIZE: usize> fmt::Display for MerklePath<NODE_SIZE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for node in self.path.iter().take(self.path.len() - 1) {
            f.write_str(&hex::encode(node))?;
            f.write_str("\n")?;
        }
        f.write_str(&hex::encode(self.path.last().unwrap()))?;
        Ok(())
    }
}

impl<const NODE_SIZE: usize> fmt::Debug for MerklePath<NODE_SIZE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, node) in self.path.iter().enumerate().take(self.path.len() - 1) {
            f.write_str(&std::format!("Node #[{}]: [", i))?;
            f.write_str(&(hex::encode(&node[..]) + "]"))?;
            f.write_str("\n")?;
        }
        let last_node = self.path.last().unwrap();
        f.write_str(&std::format!("Node #[{}]: [", self.path.len() - 1))?;
        f.write_str(&(hex::encode(&last_node[..]) + "]"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::MerklePath;

    #[test]
    pub fn merkle_path_build_test() {
        let mut m_path = MerklePath::new(3);
        m_path.push(&[0_u8; 32]);
        m_path.push(&[1_u8; 32]);
        m_path.push(&[2_u8; 32]);

        assert_eq!(m_path.len(), 3);
        assert_eq!(m_path[0], [0_u8; 32]);
        assert_eq!(m_path[1], [1_u8; 32]);
        assert_eq!(m_path[2], [2_u8; 32]);
    }
}
