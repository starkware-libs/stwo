use std::{borrow::Cow, fmt::Display};

pub trait Name {
    const NAME: Cow<'static, str>;
}
pub trait Hasher {
    type Hash: Copy + Display + self::Name + Into<Vec<u8>> + TryFrom<Vec<u8>>;

    // Input size of the compression function.
    // TODO(Ohad): Consider packing hash paramaters in a dedicated struct.
    const BLOCK_SIZE_IN_BYTES: usize;
    const OUTPUT_SIZE_IN_BYTES: usize;
    fn concat_and_hash(v1: &Self::Hash, v2: &Self::Hash) -> Self::Hash;
    fn hash(data: &[u8]) -> Self::Hash;
}
