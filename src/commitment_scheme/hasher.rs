use std::fmt::Display;

pub trait Hasher : Clone {
    fn hash(data: &[u8]) -> Self::Hash;
    
    fn concat_and_hash(v1: &Self::Hash, v2: &Self::Hash ) -> Self::Hash;

    type Hash: Copy + Into<Vec<u8>> + TryFrom<Vec<u8>> + Display;
}

