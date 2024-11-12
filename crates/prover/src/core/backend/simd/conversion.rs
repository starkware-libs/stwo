use super::m31::{PackedM31, N_LANES};
use crate::core::fields::m31::M31;

/// A trait to define the conversion from every M31-based type to it's packed complement.
pub trait Pack: Copy {
    type SimdType: Copy;
    fn pack(inputs: [Self; N_LANES]) -> Self::SimdType;
}

/// A trait to define the inverse operation of [`Pack::pack`].
pub trait Unpack: Copy {
    type CpuType: Copy;
    fn unpack(self) -> [Self::CpuType; N_LANES];
}

/// Implement the [`Pack`] and [`Unpack`] traits for the [`M31`] type.
/// This is the base case for the recursion implementations below.
impl Pack for M31 {
    type SimdType = PackedM31;

    fn pack(inputs: [M31; N_LANES]) -> Self::SimdType {
        PackedM31::from_array(inputs)
    }
}

impl Unpack for PackedM31 {
    type CpuType = M31;
    fn unpack(self) -> [M31; N_LANES] {
        self.to_array()
    }
}

impl<T: Pack, const N: usize> Pack for [T; N] {
    type SimdType = [T::SimdType; N];

    fn pack(inputs: [[T; N]; N_LANES]) -> Self::SimdType {
        std::array::from_fn(|i| T::pack(std::array::from_fn(|j| inputs[j][i])))
    }
}

impl<T: Unpack, const N: usize> Unpack for [T; N] {
    type CpuType = [T::CpuType; N];
    fn unpack(self) -> [Self::CpuType; N_LANES] {
        std::array::from_fn(|i| std::array::from_fn(|j| T::unpack(self[j])[i]))
    }
}

macro_rules! impl_tuple_conversion {
        ($($idx:tt : $type:ident),+) => {
            impl<$($type: Pack),+> Pack for ($($type,)+) {
                type SimdType = ($($type::SimdType,)+);

                fn pack(inputs: [($($type,)+); N_LANES]) -> Self::SimdType {
                    (
                        $(
                            $type::pack(std::array::from_fn(|i|
                                inputs[i].$idx
                            )),
                        )+
                    )
                }
            }

            impl<$($type: Unpack),+> Unpack for ($($type,)+) {
                type CpuType = ($($type::CpuType,)+);

                fn unpack(self) -> [Self::CpuType; N_LANES] {
                    let arrays = (
                        $(
                            $type::unpack(self.$idx),
                        )+
                    );
                    std::array::from_fn(|i| ($(arrays.$idx[i],)+))
                }
            }
        };

    }

impl_tuple_conversion!(0: T1);
impl_tuple_conversion!(0: T1, 1: T2);
impl_tuple_conversion!(0: T1, 1: T2, 2: T3);
impl_tuple_conversion!(0: T1, 1: T2, 2: T3, 3: T4);
impl_tuple_conversion!(0: T1, 1: T2, 2: T3, 3: T4, 4: T5);

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::backend::simd::conversion::{Pack, Unpack};
    use crate::core::backend::simd::m31::N_LANES;
    use crate::core::fields::m31::M31;

    #[test]
    fn test_type_conversion() {
        let mut rng = SmallRng::seed_from_u64(0);
        type MyType = (M31, [M31; 3], [M31; 15]);
        let mut rand_input = || -> MyType {
            (
                M31::from(rng.gen::<u32>()),
                [
                    M31::from(rng.gen::<u32>()),
                    M31::from(rng.gen::<u32>()),
                    M31::from(rng.gen::<u32>()),
                ],
                std::array::from_fn(|_| M31::from(rng.gen::<u32>())),
            )
        };
        let inputs: [_; N_LANES] = std::array::from_fn(|_| rand_input());

        let simd_vals = <_ as Pack>::pack(inputs);
        let outputs = simd_vals.unpack();

        assert_eq!(inputs, outputs);
    }
}
