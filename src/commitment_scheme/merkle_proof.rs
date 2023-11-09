/// Merkle authentication path.
/// Initialized correctly, each node is part hash of it's children and part hash of it's sibling.
/// Can be used as bottom up or top down. root is set manually and is optional.
/// Currently does not enforce the path to be valid, as it might get used with different hash functions.
pub struct MerklePath<const NODE_SIZE: usize> {
    path: Vec<[u8; NODE_SIZE]>,
    root: Option<Vec<u8>>,
}

impl<const NODE_SIZE: usize> MerklePath<NODE_SIZE> {
    pub fn push(&mut self, val: &[u8; NODE_SIZE]) {
        self.path.push(*val);
    }

    pub fn set_root(&mut self, val: &[u8]) {
        self.root = Some(val.to_vec());
    }

    pub fn root(&self) -> Option<&[u8]> {
        self.root.as_deref()
    }

    pub fn new(capacity: usize) -> Self {
        Self {
            path: Vec::with_capacity(capacity),
            root: None,
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

impl<const NODE_SIZE: usize> std::fmt::Display for MerklePath<NODE_SIZE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for node in self.path.iter().take(self.path.len() - 1) {
            f.write_str(&hex::encode(node))?;
            f.write_str("\n")?;
        }
        f.write_str(&hex::encode(self.path.last().unwrap()))?;

        if let Some(root) = &self.root {
            f.write_str("\n")?;
            f.write_str(&hex::encode(root))?
        }
        Ok(())
    }
}

impl<const NODE_SIZE: usize> std::fmt::Debug for MerklePath<NODE_SIZE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, node) in self.path.iter().enumerate().take(self.path.len() - 1) {
            f.write_str(&std::format!("Node #[{}]: [", i))?;
            f.write_str(&(hex::encode(&node[..]) + "]"))?;
            f.write_str("\n")?;
        }
        let last_node = self.path.last().unwrap();
        f.write_str(&std::format!("Node #[{}]: [", self.path.len() - 1))?;
        f.write_str(&(hex::encode(&last_node[..]) + "]"))?;

        if let Some(root) = &self.root {
            f.write_str(&(("\nRoot: [".to_owned()) + &hex::encode(root) + "]"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::{blake3_hash::Blake3Hasher, hasher::Hasher};

    use super::MerklePath;

    pub fn generate_test_merkle_proof() -> MerklePath<32> {
        let mut merkle_proof = MerklePath::<32>::new(3);

        let mut node = [0; 32];

        // Layer 1.
        merkle_proof.push(&node[..].try_into().unwrap());

        // Layer 2.
        Blake3Hasher::hash_one_in_place(&merkle_proof.path.last().unwrap()[..], &mut node[..]);
        merkle_proof.push(&node[..].try_into().unwrap());

        // Layer 3.
        Blake3Hasher::hash_one_in_place(&merkle_proof.path.last().unwrap()[..], &mut node[..]);
        merkle_proof.push(&node[..].try_into().unwrap());

        Blake3Hasher::hash_one_in_place(&merkle_proof.path.last().unwrap()[..], &mut node[..]);
        merkle_proof.set_root(&node[..]);

        merkle_proof
    }

    #[test]
    pub fn merkle_path_build_test() {
        let m_path = generate_test_merkle_proof();
        assert_eq!(m_path.len(), 3);

        // Check that the initial block is all zeros.
        assert_eq!(m_path.path[0][..32], [0_u8; 32]);

        // Check that the each block is the hash of the previous block.
        assert_eq!(
            m_path.path[1][..32],
            Vec::<u8>::from(Blake3Hasher::hash(&[0; 32]))[..]
        );
        assert_eq!(
            m_path.path[2][..32],
            Vec::<u8>::from(Blake3Hasher::hash(&m_path.path[1][..]))[..32]
        );
        assert_eq!(
            m_path.root.unwrap(),
            Vec::<u8>::from(Blake3Hasher::hash(&m_path.path[2][..]))[..32]
        );
    }

    #[test]
    pub fn merkle_path_debug_test() {
        let m_path = generate_test_merkle_proof();
        println!("{:?}", m_path);
    }
}
