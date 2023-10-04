use std::{borrow::Cow, fmt::Display};

pub trait Name {
    const NAME: Cow<'static, str>;
}
pub trait Hasher: Clone {
    type Hash: Copy + Display + self::Name + Into<Vec<u8>> + TryFrom<Vec<u8>>;
    const BLOCK_SIZE: usize;
    fn concat_and_hash(v1: &Self::Hash, v2: &Self::Hash) -> Self::Hash;
    fn hash(data: &[u8]) -> Self::Hash;
}
