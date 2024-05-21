use std::fmt::Display;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use bytemuck::{Pod, Zeroable};
use rand::distributions::{Distribution, Standard};

use super::{ComplexConjugate, FieldExpOps};
use crate::impl_field;

pub const MODULUS_BITS: u32 = 31;
pub const N_BYTES_FELT: usize = 4;
pub const P: u32 = 2147483647; // 2 ** 31 - 1

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]

/// Mersenne Field
/// # M31
/// 
/// A struct representing a `M31` Field.
/// M31 is the prime of the form p = 2<sup>31</sup> - 1. It enables very efficient arithmetic on 32 bit architectures.
///
/// This struct wraps a `u32` value.
///
/// # Example
///
/// ```
///     use crate::stwo_prover::core::fields::m31::{M31};
/// 
///     let m31_value = M31(42);
///     println!("M31 value: {}", m31_value.0);
/// ```
pub struct M31(pub u32);
pub type BaseField = M31;

impl_field!(M31, P);

impl M31 {
    /// Computes the square root of `M31`.
    ///
    /// # Arguments
    ///
    /// * `self` - The `M31` instance.
    ///
    /// # Returns
    ///
    /// - `Some(M31(sqrt))` if the square root is a valid real number.
    /// - `None` if the square root is not a real number.
    ///
    /// # Example
    /// ```
    ///     use crate::stwo_prover::core::fields::m31::{M31};
    /// 
    ///     let x: u32 = 13300609;
    ///     let mx = M31(x);
    ///     println!("sqrt: {:?}", mx.sqrt());
    ///     assert_eq!(mx.sqrt(), Some(M31(3647)))
    /// ```
    pub fn sqrt(&self) -> Option<Self> {
        let result = self.pow(1 << 29);
        (result.square() == *self).then_some(result)
    }

    /// Assumes that `val` is in the range [0, 2 * `P`) and returns `val` % `P`.
    pub fn partial_reduce(val: u32) -> Self {
        Self(val.checked_sub(P).unwrap_or(val))
    }

    /// Assumes that `val` is in the range [0, `P`.pow(2)) and returns `val` % `P`.
    pub fn reduce(val: u64) -> Self {
        Self((((((val >> MODULUS_BITS) + val + 1) >> MODULUS_BITS) + val) & (P as u64)) as u32)
    }

    pub const fn from_u32_unchecked(arg: u32) -> Self {
        Self(arg)
    }
}

/// Implementation of the `Display` trait for `M31` for custom formatting when displayed.
impl Display for M31 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Implementation of the `Add` trait for `M31`, allowing addition of two `M31` instances.
impl Add for M31 {
    type Output = Self;
    /// Adds two `M31` instances.
    ///
    /// # Arguments
    ///
    /// * `self` - The first `M31` instance.
    /// * `rhs` - The second `M31` instance to be added.
    ///
    /// # Returns
    ///
    /// A new `M31` instance with the sum of the values.
    ///
    /// # Examples
    /// 
    /// ```
    ///     use crate::stwo_prover::core::fields::m31::{M31, P};
    ///     let x: u32 = 1697131274;
    ///     let y: u32 = 1369454381;
    ///     let mx = M31(x);
    ///     let my = M31(y);
    /// 
    ///     let sum = mx + my;
    ///     assert_eq!(sum.0, (x+y)%P)
    /// ```
    fn add(self, rhs: Self) -> Self::Output {
        Self::partial_reduce(self.0 + rhs.0)
    }
}

/// Implementation of the `Neg` trait for `M31`, enables computation of negative elements.
impl Neg for M31 {
    type Output = Self;

    /// The `neg` function instances of M31 enables computation of negative elements.
    /// It is simply the additive inverse of another number, meaning it's the element that, when added to the original, results in zero.
    ///
    /// # Arguments
    ///
    /// * `self` - The `M31` instance.
    ///
    /// # Returns
    ///
    /// A new `M31` instance containing the negated value.
    ///
    /// # Examples
    /// 
    /// ```
    ///     use crate::stwo_prover::core::fields::m31::{M31, P};
    ///     let x: u32 = 1697131274;
    ///     let mx = M31(x);
    ///     let neg_mx = -mx;
    /// 
    ///     assert_eq!((neg_mx).0, (P - x));
    ///     assert_eq!((neg_mx + mx), M31(0), "Neg of mx + mx must be 0");
    /// ```
    /// 
    fn neg(self) -> Self::Output {
        Self::partial_reduce(P - self.0)
    }
}

/// Implementation of the `Sub` trait for `M31`, allowing subtraction of two `M31` instances.
impl Sub for M31 {
    type Output = Self;
    /// Subtract two `M31` instances.
    ///
    /// # Arguments
    ///
    /// * `self` - The first `M31` instance.
    /// * `rhs` - The second `M31` instance to be subtracted.
    ///
    /// # Returns
    ///
    /// A new `M31` instance with the subtracted value.
    ///
    /// # Examples
    /// 
    /// ```
    ///     use crate::stwo_prover::core::fields::m31::{M31, P};
    ///     let x: u32 = 1697131274;
    ///     let y: u32 = 1369454381;
    ///     let mx = M31(x);
    ///     let my = M31(y);
    /// 
    ///     let minus = mx - my;
    ///     assert_eq!(minus.0, (x-y)%P);
    /// ```
    fn sub(self, rhs: Self) -> Self::Output {
        Self::partial_reduce(self.0 + P - rhs.0)
    }
}

/// Implementation of the `Mul` trait for `M31`, allowing multiplication of two `M31` instances.
impl Mul for M31 {
    type Output = Self;
    /// Multiply two `M31` instances.
    ///
    /// # Arguments
    ///
    /// * `self` - The first `M31` instance.
    /// * `rhs` - The second `M31` instance to be multiplied by.
    ///
    /// # Returns
    ///
    /// A new `M31` instance with the multiplied value.
    ///
    /// # Examples
    /// 
    /// ```
    ///     use crate::stwo_prover::core::fields::m31::{M31, P};
    /// 
    ///     let x: u32 = 141234;
    ///     let y: u32 = 24455;
    ///     let mx = M31(x);
    ///     let my = M31(y);
    /// 
    ///     let mul = mx * my;
    ///     assert_eq!(mul.0, (x*y)%P);
    /// ```
    fn mul(self, rhs: Self) -> Self::Output {
        Self::reduce((self.0 as u64) * (rhs.0 as u64))
    }
}

impl FieldExpOps for M31 {
    /// The `inverse` function computes the inverse of an element in M31
    /// The inverse of an element is the number you multiply it with to get the finite field element 1.
    /// It uses the pow2147483645 function.
    ///     
    /// # Arguments
    ///
    /// * `self` - The `M31` instance.
    ///
    /// # Returns
    ///
    /// A new `M31` instance with the inverse value.
    ///
    /// # Examples
    /// 
    /// ```
    ///     use crate::stwo_prover::core::fields::m31::{M31, P};
    ///     use crate::stwo_prover::core::fields::FieldExpOps;
    /// 
    ///     let x: u32 = 141234;
    ///     let mx = M31(x);
    ///     let inv = mx.inverse();
    /// 
    ///     assert_eq!(inv * mx, M31(1), "inverse of mx multiplied by mx must be 1");
    /// ```
    fn inverse(&self) -> Self {
        assert!(!self.is_zero(), "0 has no inverse");
        pow2147483645(*self)
    }
}

impl ComplexConjugate for M31 {
    fn complex_conjugate(&self) -> Self {
        *self
    }
}

/// Implementation of the `One` trait for `M31`
impl One for M31 {
    /// Creates a new `M31` instance with the value 1.
    ///
    /// # Returns
    ///
    /// A new `M31` instance containing the value 1. (the multiplicative identity element one)
    ///
    /// # Examples
    /// 
    /// ```
    ///     use num_traits::one;
    ///     use crate::stwo_prover::core::fields::m31::{M31, P};
    /// 
    ///     let m31_one: M31 = one();
    ///     println!("One is : {:?}", m31_one);
    /// ```
    fn one() -> Self {
        Self(1)
    }
}

/// Implementation of the `Zero` trait for `M31`
impl Zero for M31 {
    /// Creates a new `M31` instance with the value 0.
    ///
    /// # Returns
    ///
    /// A new `M31` instance containing the value 0. (the additive identity element zero)
    ///
    /// # Examples
    /// 
    /// ```
    ///     use num_traits::zero;
    ///     use crate::stwo_prover::core::fields::m31::{M31, P};
    /// 
    ///     let m31_zero: M31 = zero();
    ///     println!("Zero is : {:?}", m31_zero);
    /// ```
    fn zero() -> Self {
        Self(0)
    }

    ///  Checks if a`M31` instance is 0.
    ///
    /// # Returns
    ///
    /// A `bool`.
    ///
    /// # Examples
    /// 
    /// ```
    ///     use num_traits::{zero, Zero};
    ///     use crate::stwo_prover::core::fields::m31::{M31, P};
    /// 
    ///     let m31_zero: M31 = zero();
    ///     println!("m31_zero is Zero: {:?}", m31_zero.is_zero());
    /// 
    /// ```
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

/// Implementation of the `From` trait for `usize` allowing conversion into `M31`.
impl From<usize> for M31 {
    /// Converts a `usize` into a `M31`.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to be converted into `M31`.
    ///
    /// # Returns
    ///
    /// A new `M31` instance of the `value`. ///
    /// # Example
    ///
    /// ```    
    ///     use stwo_prover::core::fields::m31::M31;
    /// 
    ///     let value: usize = 4294967295;
    ///     let m31: M31 = M31::from(value);
    ///     println!("M31 value: {:?}", m31);
    ///     assert_eq!(m31, M31(1));
    /// ```
    fn from(value: usize) -> Self {
        M31::reduce(value.try_into().unwrap())
    }
}

/// Implementation of the `From` trait for `u32` allowing conversion into `M31`.
impl From<u32> for M31 {
    /// Converts a `u32` into a `M31`.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to be converted into `M31`.
    ///
    /// # Returns
    ///
    /// A new `M31` instance of the `value`.
    ///
    /// # Example
    ///
    /// ```
    ///     use stwo_prover::core::fields::m31::M31;
    /// 
    ///     let value: u32 = 4294967295;
    ///     let m31: M31 = M31::from(value);
    ///     println!("M31 value: {:?}", m31);
    ///     assert_eq!(m31, M31(1));
    /// ```
    fn from(value: u32) -> Self {
        M31::reduce(value.into())
    }
}

/// Implementation of the `From` trait for `i32` allowing conversion into `M31`.
impl From<i32> for M31 {
    /// Converts a `i32` into a `M31`.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to be converted into `M31`.
    ///
    /// # Returns
    ///
    /// A new `M31` instance of the `value`.
    ///
    /// # Example
    ///
    /// ```
    ///     use stwo_prover::core::fields::m31::M31;
    /// 
    ///     let value: i32 = 2147483647;
    ///     let m31: M31 = M31::from(value);
    ///     println!("M31 value: {:?}", m31);
    ///     assert_eq!(m31, M31(0));
    /// ```
    fn from(value: i32) -> Self {
        M31::reduce(value.try_into().unwrap())
    }
}

impl Distribution<M31> for Standard {
    // Not intended for cryptographic use. Should only be used in tests and benchmarks.
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> M31 {
        M31(rng.gen_range(0..P))
    }
}

#[cfg(test)]
#[macro_export]
macro_rules! m31 {
    ($m:expr) => {
        $crate::core::fields::m31::M31::from_u32_unchecked($m)
    };
}

/// Computes `v^((2^31-1)-2)`.
///
/// Computes the multiplicative inverse of [`M31`] elements with 37 multiplications vs naive 60
/// multiplications. Made generic to support both vectorized and non-vectorized implementations.
/// Multiplication tree found with [addchain](https://github.com/mmcloughlin/addchain).
pub fn pow2147483645<T: FieldExpOps>(v: T) -> T {
    let t0 = sqn::<2, T>(v) * v;
    let t1 = sqn::<1, T>(t0) * t0;
    let t2 = sqn::<3, T>(t1) * t0;
    let t3 = sqn::<1, T>(t2) * t0;
    let t4 = sqn::<8, T>(t3) * t3;
    let t5 = sqn::<8, T>(t4) * t3;
    sqn::<7, T>(t5) * t2
}

/// Computes `v^(2*n)`.
fn sqn<const N: usize, T: FieldExpOps>(mut v: T) -> T {
    for _ in 0..N {
        v = v.square();
    }
    v
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::{M31, P};
    use crate::core::fields::m31::{pow2147483645, BaseField};
    use crate::core::fields::{FieldExpOps, IntoSlice};

    fn mul_p(a: u32, b: u32) -> u32 {
        ((a as u64 * b as u64) % P as u64) as u32
    }

    fn add_p(a: u32, b: u32) -> u32 {
        (a + b) % P
    }

    fn neg_p(a: u32) -> u32 {
        if a == 0 {
            0
        } else {
            P - a
        }
    }

    #[test]
    fn test_basic_ops() {
        let mut rng = SmallRng::seed_from_u64(0);
        for _ in 0..10000 {
            let x: u32 = rng.gen::<u32>() % P;
            let y: u32 = rng.gen::<u32>() % P;
            assert_eq!(m31!(add_p(x, y)), m31!(x) + m31!(y));
            assert_eq!(m31!(mul_p(x, y)), m31!(x) * m31!(y));
            assert_eq!(m31!(neg_p(x)), -m31!(x));
        }
    }

    #[test]
    fn test_into_slice() {
        let mut rng = SmallRng::seed_from_u64(0);
        let x = (0..100).map(|_| rng.gen()).collect::<Vec<M31>>();

        let slice = M31::into_slice(&x);

        for i in 0..100 {
            assert_eq!(
                x[i],
                m31!(u32::from_le_bytes(
                    slice[i * 4..(i + 1) * 4].try_into().unwrap()
                ))
            );
        }
    }

    #[test]
    fn pow2147483645_works() {
        let v = BaseField::from(19);

        assert_eq!(pow2147483645(v), v.pow(2147483645));
    }
}
