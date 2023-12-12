use std::borrow::Cow;
use std::fmt::Display;

pub trait Name {
    const NAME: Cow<'static, str>;
}

pub trait Hasher {
    // TODO(Ohad): Define a 'hash' trait to enforce all these traits on an implementor.
    type Hash: Copy
        + Default
        + Display
        + self::Name
        + Into<Vec<Self::NativeType>>
        + TryFrom<Vec<Self::NativeType>>
        + AsRef<[Self::NativeType]>
        + for<'a> From<&'a [Self::NativeType]>
        + Send
        + Sync;
    type NativeType: Sized + Eq;
    // Input size of the compression function.
    const BLOCK_SIZE: usize;
    const OUTPUT_SIZE: usize;

    fn concat_and_hash(v1: &Self::Hash, v2: &Self::Hash) -> Self::Hash;

    fn hash(data: &[Self::NativeType]) -> Self::Hash;

    fn hash_one_in_place(data: &[Self::NativeType], dst: &mut [Self::NativeType]);

    fn hash_many(data: &[Vec<Self::NativeType>]) -> Vec<Self::Hash>;

    /// Hash many inputs of the same length.
    /// Writes output directly to corresponding pointers in dst.
    ///
    /// # Safety
    ///
    /// Inputs must be of the same size. output locations must all point to valid, allocated and
    /// distinct locations in memory.
    // TODO(Ohad): make redundent and delete.
    unsafe fn hash_many_in_place(
        data: &[*const Self::NativeType],
        single_input_length_bytes: usize,
        dst: &[*mut Self::NativeType],
    );

    // TODO(Ohad): Consider adding a trait for hashers that support multi-source hashing, and
    // defining proper input structure for it.
    fn hash_many_multi_src(data: &[Vec<&[Self::NativeType]>]) -> Vec<Self::Hash>;
}
