use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{Mul, MulAssign, Neg};

use num_traits::{NumAssign, NumAssignOps, NumOps, One};

use super::backend::ColumnOps;

pub mod cm31;
pub mod m31;
pub mod qm31;
pub mod secure_column;

pub trait FieldOps<F: Field>: ColumnOps<F> {
    // TODO(Ohad): change to use a mutable slice.
    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column);
}

pub trait FieldExpOps: Mul<Output = Self> + MulAssign + Sized + One + Copy {
    fn square(&self) -> Self {
        (*self) * (*self)
    }

    fn pow(&self, exp: u128) -> Self {
        let mut res = Self::one();
        let mut base = *self;
        let mut exp = exp;
        while exp > 0 {
            if exp & 1 == 1 {
                res *= base;
            }
            base = base.square();
            exp >>= 1;
        }
        res
    }

    fn inverse(&self) -> Self;

    /// Inverts a batch of elements using Montgomery's trick.
    fn batch_inverse(column: &[Self], dst: &mut [Self]) {
        const WIDTH: usize = 4;
        let n = column.len();
        debug_assert!(dst.len() >= n);

        if n <= WIDTH || n % WIDTH != 0 {
            batch_inverse_classic(column, dst);
            return;
        }

        // First pass. Compute 'WIDTH' cumulative products in an interleaving fashion, reducing
        // instruction dependency and allowing better pipelining.
        let mut cum_prod: [Self; WIDTH] = [Self::one(); WIDTH];
        dst[..WIDTH].copy_from_slice(&cum_prod);
        for i in 0..n {
            cum_prod[i % WIDTH] *= column[i];
            dst[i] = cum_prod[i % WIDTH];
        }

        // Inverse cumulative products.
        // Use classic batch inversion.
        let mut tail_inverses = [Self::one(); WIDTH];
        batch_inverse_classic(&dst[n - WIDTH..], &mut tail_inverses);

        // Second pass.
        for i in (WIDTH..n).rev() {
            dst[i] = dst[i - WIDTH] * tail_inverses[i % WIDTH];
            tail_inverses[i % WIDTH] *= column[i];
        }
        dst[0..WIDTH].copy_from_slice(&tail_inverses);
    }
}

/// Assumes dst is initialized and of the same length as column.
fn batch_inverse_classic<T: FieldExpOps>(column: &[T], dst: &mut [T]) {
    let n = column.len();
    debug_assert!(dst.len() >= n);

    dst[0] = column[0];
    // First pass.
    for i in 1..n {
        dst[i] = dst[i - 1] * column[i];
    }

    // Inverse cumulative product.
    let mut curr_inverse = dst[n - 1].inverse();

    // Second pass.
    for i in (1..n).rev() {
        dst[i] = dst[i - 1] * curr_inverse;
        curr_inverse *= column[i];
    }
    dst[0] = curr_inverse;
}

pub trait Field:
    NumAssign
    + Neg<Output = Self>
    + ComplexConjugate
    + Copy
    + Default
    + Debug
    + Display
    + PartialOrd
    + Ord
    + Send
    + Sync
    + Sized
    + FieldExpOps
    + Product
    + for<'a> Product<&'a Self>
    + Sum
    + for<'a> Sum<&'a Self>
{
    fn double(&self) -> Self {
        (*self) + (*self)
    }
}

/// # Safety
///
/// Do not use unless you are aware of the endianess in the platform you are compiling for, and the
/// Field element's representation in memory.
// TODO(Ohad): Do not compile on non-le targets.
pub unsafe trait IntoSlice<T: Sized>: Sized {
    fn into_slice(sl: &[Self]) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(
                sl.as_ptr() as *const T,
                std::mem::size_of_val(sl) / std::mem::size_of::<T>(),
            )
        }
    }
}

unsafe impl<F: Field> IntoSlice<u8> for F {}

pub trait ComplexConjugate {
    /// # Example
    ///
    /// ```
    /// use stwo_prover::core::fields::m31::P;
    /// use stwo_prover::core::fields::qm31::QM31;
    /// use stwo_prover::core::fields::ComplexConjugate;
    ///
    /// let x = QM31::from_u32_unchecked(1, 2, 3, 4);
    /// assert_eq!(
    ///     x.complex_conjugate(),
    ///     QM31::from_u32_unchecked(1, 2, P - 3, P - 4)
    /// );
    /// ```
    fn complex_conjugate(&self) -> Self;
}

pub trait ExtensionOf<F: Field>: Field + From<F> + NumOps<F> + NumAssignOps<F> {
    const EXTENSION_DEGREE: usize;
}

impl<F: Field> ExtensionOf<F> for F {
    const EXTENSION_DEGREE: usize = 1;
}

#[macro_export]
macro_rules! impl_field {
    ($field_name: ty, $field_size: ident) => {
        use std::iter::{Product, Sum};

        use num_traits::{Num, One, Zero};
        use $crate::core::fields::Field;

        impl Num for $field_name {
            type FromStrRadixErr = Box<dyn std::error::Error>;

            fn from_str_radix(_str: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                unimplemented!(
                    "Num::from_str_radix is not implemented for {}",
                    stringify!($field_name)
                );
            }
        }

        impl Field for $field_name {}

        impl AddAssign for $field_name {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl SubAssign for $field_name {
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl MulAssign for $field_name {
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl Div for $field_name {
            type Output = Self;

            #[allow(clippy::suspicious_arithmetic_impl)]
            fn div(self, rhs: Self) -> Self::Output {
                self * rhs.inverse()
            }
        }

        impl DivAssign for $field_name {
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }

        impl Rem for $field_name {
            type Output = Self;

            fn rem(self, _rhs: Self) -> Self::Output {
                unimplemented!("Rem is not implemented for {}", stringify!($field_name));
            }
        }

        impl RemAssign for $field_name {
            fn rem_assign(&mut self, _rhs: Self) {
                unimplemented!(
                    "RemAssign is not implemented for {}",
                    stringify!($field_name)
                );
            }
        }

        impl Product for $field_name {
            fn product<I>(mut iter: I) -> Self
            where
                I: Iterator<Item = Self>,
            {
                let first = iter.next().unwrap_or_else(Self::one);
                iter.fold(first, |a, b| a * b)
            }
        }

        impl<'a> Product<&'a Self> for $field_name {
            fn product<I>(iter: I) -> Self
            where
                I: Iterator<Item = &'a Self>,
            {
                iter.map(|&v| v).product()
            }
        }

        impl Sum for $field_name {
            fn sum<I>(mut iter: I) -> Self
            where
                I: Iterator<Item = Self>,
            {
                let first = iter.next().unwrap_or_else(Self::zero);
                iter.fold(first, |a, b| a + b)
            }
        }

        impl<'a> Sum<&'a Self> for $field_name {
            fn sum<I>(iter: I) -> Self
            where
                I: Iterator<Item = &'a Self>,
            {
                iter.map(|&v| v).sum()
            }
        }
    };
}

/// Used to extend a field (with characteristic M31) by 2.
#[macro_export]
macro_rules! impl_extension_field {
    ($field_name: ident, $extended_field_name: ty) => {
        use rand::distributions::{Distribution, Standard};
        use $crate::core::fields::ExtensionOf;

        impl ExtensionOf<M31> for $field_name {
            const EXTENSION_DEGREE: usize =
                <$extended_field_name as ExtensionOf<M31>>::EXTENSION_DEGREE * 2;
        }

        impl Add for $field_name {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self(self.0 + rhs.0, self.1 + rhs.1)
            }
        }

        impl Neg for $field_name {
            type Output = Self;

            fn neg(self) -> Self::Output {
                Self(-self.0, -self.1)
            }
        }

        impl Sub for $field_name {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                Self(self.0 - rhs.0, self.1 - rhs.1)
            }
        }

        impl One for $field_name {
            fn one() -> Self {
                Self(
                    <$extended_field_name>::one(),
                    <$extended_field_name>::zero(),
                )
            }
        }

        impl Zero for $field_name {
            fn zero() -> Self {
                Self(
                    <$extended_field_name>::zero(),
                    <$extended_field_name>::zero(),
                )
            }

            fn is_zero(&self) -> bool {
                *self == Self::zero()
            }
        }

        impl Add<M31> for $field_name {
            type Output = Self;

            fn add(self, rhs: M31) -> Self::Output {
                Self(self.0 + rhs, self.1)
            }
        }

        impl Add<$field_name> for M31 {
            type Output = $field_name;

            fn add(self, rhs: $field_name) -> Self::Output {
                rhs + self
            }
        }

        impl Sub<M31> for $field_name {
            type Output = Self;

            fn sub(self, rhs: M31) -> Self::Output {
                Self(self.0 - rhs, self.1)
            }
        }

        impl Sub<$field_name> for M31 {
            type Output = $field_name;

            fn sub(self, rhs: $field_name) -> Self::Output {
                -rhs + self
            }
        }

        impl Mul<M31> for $field_name {
            type Output = Self;

            fn mul(self, rhs: M31) -> Self::Output {
                Self(self.0 * rhs, self.1 * rhs)
            }
        }

        impl Mul<$field_name> for M31 {
            type Output = $field_name;

            fn mul(self, rhs: $field_name) -> Self::Output {
                rhs * self
            }
        }

        impl Div<M31> for $field_name {
            type Output = Self;

            fn div(self, rhs: M31) -> Self::Output {
                Self(self.0 / rhs, self.1 / rhs)
            }
        }

        impl Div<$field_name> for M31 {
            type Output = $field_name;

            #[allow(clippy::suspicious_arithmetic_impl)]
            fn div(self, rhs: $field_name) -> Self::Output {
                rhs.inverse() * self
            }
        }

        impl ComplexConjugate for $field_name {
            fn complex_conjugate(&self) -> Self {
                Self(self.0, -self.1)
            }
        }

        impl From<M31> for $field_name {
            fn from(x: M31) -> Self {
                Self(x.into(), <$extended_field_name>::zero())
            }
        }

        impl AddAssign<M31> for $field_name {
            fn add_assign(&mut self, rhs: M31) {
                *self = *self + rhs;
            }
        }

        impl SubAssign<M31> for $field_name {
            fn sub_assign(&mut self, rhs: M31) {
                *self = *self - rhs;
            }
        }

        impl MulAssign<M31> for $field_name {
            fn mul_assign(&mut self, rhs: M31) {
                *self = *self * rhs;
            }
        }

        impl DivAssign<M31> for $field_name {
            fn div_assign(&mut self, rhs: M31) {
                *self = *self / rhs;
            }
        }

        impl Rem<M31> for $field_name {
            type Output = Self;

            fn rem(self, _rhs: M31) -> Self::Output {
                unimplemented!("Rem is not implemented for {}", stringify!($field_name));
            }
        }

        impl RemAssign<M31> for $field_name {
            fn rem_assign(&mut self, _rhs: M31) {
                unimplemented!(
                    "RemAssign is not implemented for {}",
                    stringify!($field_name)
                );
            }
        }

        impl Distribution<$field_name> for Standard {
            // Not intended for cryptographic use. Should only be used in tests and benchmarks.
            fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> $field_name {
                $field_name(rng.gen(), rng.gen())
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::fields::m31::M31;
    use crate::core::fields::FieldExpOps;

    #[test]
    fn test_slice_batch_inverse() {
        let mut rng = SmallRng::seed_from_u64(0);
        let elements: [M31; 16] = rng.gen();
        let expected = elements.iter().map(|e| e.inverse()).collect::<Vec<_>>();
        let mut dst = [M31::zero(); 16];

        M31::batch_inverse(&elements, &mut dst);

        assert_eq!(expected, dst);
    }

    #[test]
    #[should_panic]
    fn test_slice_batch_inverse_wrong_dst_size() {
        let mut rng = SmallRng::seed_from_u64(0);
        let elements: [M31; 16] = rng.gen();
        let mut dst = [M31::zero(); 15];

        M31::batch_inverse(&elements, &mut dst);
    }
}
