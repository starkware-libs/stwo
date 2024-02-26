use std::fmt::{Debug, Display};
use std::ops::{Index, Neg};

use num_traits::{NumAssign, NumAssignOps, NumOps};

#[cfg(target_arch = "x86_64")]
pub mod avx512_m31;
pub mod cm31;
pub mod m31;
pub mod qm31;

pub trait FieldOps<F: Field> {
    type Column: Column<F>;
    fn bit_reverse_column(column: &mut Self::Column);
}

pub type Col<B, F> = <B as FieldOps<F>>::Column;

// TODO(spapini): Consider removing the generic parameter and only support BaseField.
pub trait Column<F: Field>: Clone + Debug + Index<usize, Output = F> + FromIterator<F> {
    fn zeros(len: usize) -> Self;
    fn to_vec(&self) -> Vec<F>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait Field:
    NumAssign
    + Neg<Output = Self>
    + ComplexConjugate
    + Copy
    + Debug
    + Display
    + PartialOrd
    + Ord
    + Send
    + Sync
    + Sized
{
    fn square(&self) -> Self {
        (*self) * (*self)
    }

    fn double(&self) -> Self {
        (*self) + (*self)
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
    /// use stwo::core::fields::m31::P;
    /// use stwo::core::fields::qm31::QM31;
    /// use stwo::core::fields::ComplexConjugate;
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

        impl Field for $field_name {
            fn inverse(&self) -> Self {
                assert!(*self != Self::zero(), "0 has no inverse");
                self.pow(($field_size - 2) as u128)
            }
        }

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
    };
}

/// Used to extend a field (with characteristic M31) by 2.
#[macro_export]
macro_rules! impl_extension_field {
    ($field_name: ty, $extended_field_name: ty) => {
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
    };
}
