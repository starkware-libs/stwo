use std::borrow::Cow;
use std::fmt::Display;

pub trait Name {
    const NAME: Cow<'static, str>;
}

pub trait Hasher {
    type Hash: Copy
        + Display
        + self::Name
        + Into<Vec<u8>>
        + TryFrom<Vec<u8>>
        + AsRef<[u8]>
        + for<'a> From<&'a [u8]>;

    // Input size of the compression function.
    // TODO(Ohad): Consider packing hash paramaters in a dedicated struct.
    const BLOCK_SIZE_IN_BYTES: usize;
    const OUTPUT_SIZE_IN_BYTES: usize;

    fn concat_and_hash(v1: &Self::Hash, v2: &Self::Hash) -> Self::Hash;

    fn hash(data: &[u8]) -> Self::Hash;

    fn hash_one_in_place(data: &[u8], dst: &mut [u8]);

    fn hash_many(data: &[Vec<u8>]) -> Vec<Self::Hash>;

    /// Hash many inputs of the same length.
    /// Writes output directly to corresponding pointers in dst.
    ///
    /// # Safety
    ///
    /// Inputs must be of the same size. output locations must all point to valid, allocated and
    /// distinct locations in memory.
    unsafe fn hash_many_in_place(
        data: &[*const u8],
        single_input_length_bytes: usize,
        dst: &[*mut u8],
    );
}
