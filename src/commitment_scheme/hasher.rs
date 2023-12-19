use std::borrow::Cow;
use std::fmt::{Debug, Display};

pub trait Name {
    const NAME: Cow<'static, str>;
}

pub trait BasicHasher {
    // TODO(Ohad): Define a 'hash' trait to enforce all these traits on an implementor.
    type Hash: Hash<Self::NativeType>;
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

    fn hash_many_multi_src_in_place(data: &[Vec<&[Self::NativeType]>], dst: &mut [Self::Hash]);
}

pub trait Hash<NativeType: Sized + Eq>:
    Copy
    + Default
    + Display
    + Debug
    + self::Name
    + Into<Vec<NativeType>>
    + TryFrom<Vec<NativeType>>
    + AsRef<[NativeType]>
    + for<'a> From<&'a [NativeType]>
    + Send
    + Sync
{
}

/// An API for hash functions that support incremental hashing.
///
/// # Example
///
/// ```
/// use prover_research::commitment_scheme::blake3_hash::Blake3Hasher;
/// use prover_research::commitment_scheme::hasher::{BasicHasher, IncrementalHasher};
///
/// let mut hasher = Blake3Hasher::new();
/// hasher.update(&[1, 2, 3]);
/// hasher.update(&[4, 5, 6]);
/// let hash = hasher.finalize();
///
/// assert_eq!(hash, Blake3Hasher::hash(&[1, 2, 3, 4, 5, 6]));
/// ```
pub trait IncrementalHasher<NativeType: Sized + Eq, H: Hash<NativeType>> {
    fn new() -> Self;
    fn reset(&mut self);
    fn update(&mut self, data: &[NativeType]);
    fn finalize(self) -> H;
    fn finalize_reset(&mut self) -> H;
}
