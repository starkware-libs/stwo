use std::fmt::{Debug, Display};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use super::{ComplexConjugate, FieldExpOps};
use crate::core::fields::m31::M31;
use crate::{impl_extension_field, impl_field};

pub const P2: u64 = 4611686014132420609; // (2 ** 31 - 1) ** 2

/// Complex extension field of M31.
/// Equivalent to M31\[x\] over (x^2 + 1) as the irreducible polynomial.
/// Represented as (a, b) of a + bi.
#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CM31(pub M31, pub M31);

impl_field!(CM31, P2);
impl_extension_field!(CM31, M31);

impl CM31 {
    pub const fn from_u32_unchecked(a: u32, b: u32) -> CM31 {
        Self(M31::from_u32_unchecked(a), M31::from_u32_unchecked(b))
    }

    /// Converts two `M31` into a `CM31`.
    ///
    /// # Arguments
    ///
    /// * `a` - The first `M31` instance.
    /// * `b` - The second `M31` instance.
    ///
    /// # Returns
    ///
    /// A new `CM31` instance of the `a` and `b` of the form (a + bi).
    ///
    /// # Example
    ///
    /// ```
    ///     use stwo_prover::core::fields::m31::M31;
    ///     use stwo_prover::core::fields::cm31::CM31;
    /// 
    ///     let ma = M31(8);
    ///     let mb = M31(5);
    /// 
    ///     let cm = CM31::from_m31(ma, mb);
    ///     println!("CM31 value: {:?}", cm); // CM31 value: 8 + 5i
    /// ```
    pub fn from_m31(a: M31, b: M31) -> CM31 {
        Self(a, b)
    }
}

/// Implementation of the `Display` trait for `CM31` for custom formatting when displayed.
impl Display for CM31 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} + {}i", self.0, self.1)
    }
}

/// Implementation of the `Debug` trait for `CM31` for custom formatting when displayed.
impl Debug for CM31 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} + {}i", self.0, self.1)
    }
}

/// Implementation of the `Mul` trait for `CM31`, allowing multiplication of two `CM31` instances.
impl Mul for CM31 {
    type Output = Self;

    /// Multiply two `CM31` instances.
    ///
    /// # Arguments
    ///
    /// * `self` - The first `CM31` instance.
    /// * `rhs` - The second `CM31` instance to be multiplied by.
    ///
    /// # Returns
    ///
    /// A new `CM31` instance with the multiplied value.
    ///
    /// # Examples
    /// 
    /// ```
    ///     use stwo_prover::core::fields::m31::M31;
    ///     use stwo_prover::core::fields::cm31::CM31;
    /// 
    ///    let ma1 = M31(8);
    ///    let mb1 = M31(5);
    ///    let cm1 = CM31::from_m31(ma1, mb1);  // 8 + 5i
    ///
    ///    let ma2 = M31(2);
    ///    let mb2 = M31(4);
    ///    let cm2 = CM31::from_m31(ma2, mb2);  // 2 + 4i
    ///
    ///    let prod = cm1 * cm2;                // (8 * 2) - (5i * 4i) + (8 * 4i) + (5i * 2)  = (16 - 20) + (32i + 10i)
    ///    println!("prod value: {:?}", prod);  // (-4 + 42i) = (-4 % 2147483647 + 42i) = (2147483643 + 42i).
    /// ```
    fn mul(self, rhs: Self) -> Self::Output {
        // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i.
        Self(
            self.0 * rhs.0 - self.1 * rhs.1,
            self.0 * rhs.1 + self.1 * rhs.0,
        )
    }
}

impl FieldExpOps for CM31 {
    /// The `inverse` function computes the inverse of CM31.
    /// The inverse of an element is the number you multiply it with to get the finite field element 1.
    ///     
    /// # Arguments
    ///
    /// * `self` - The `CM31` instance.
    ///
    /// # Returns
    ///
    /// A new `CM31` instance with the inverse value.
    ///
    /// # Examples
    /// 
    /// ```
    ///     use stwo_prover::core::fields::m31::M31;
    ///     use stwo_prover::core::fields::cm31::CM31;
    ///     use crate::stwo_prover::core::fields::FieldExpOps;
    /// 
    ///     let ma = M31(1);
    ///     let mb = M31(2);
    ///     let cm = CM31::from_m31(ma, mb);
    ///
    ///     let cm_inv = cm.inverse();
    ///     println!("inverse value: {:?}", cm_inv);
    /// ```
    fn inverse(&self) -> Self {
        assert!(!self.is_zero(), "0 has no inverse");
        // 1 / (a + bi) = (a - bi) / (a^2 + b^2).
        Self(self.0, -self.1) * (self.0.square() + self.1.square()).inverse()
    }
}

#[cfg(test)]
#[macro_export]
macro_rules! cm31 {
    ($m0:expr, $m1:expr) => {
        CM31::from_u32_unchecked($m0, $m1)
    };
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::CM31;
    use crate::core::fields::m31::P;
    use crate::core::fields::{FieldExpOps, IntoSlice};
    use crate::m31;

    #[test]
    fn test_inverse() {
        let cm = cm31!(1, 2);
        let cm_inv = cm.inverse();
        assert_eq!(cm * cm_inv, cm31!(1, 0));
    }

    #[test]
    fn test_ops() {
        let cm0 = cm31!(1, 2);
        let cm1 = cm31!(4, 5);
        let m = m31!(8);
        let cm = CM31::from(m);
        let cm0_x_cm1 = cm31!(P - 6, 13);

        assert_eq!(cm0 + cm1, cm31!(5, 7));
        assert_eq!(cm1 + m, cm1 + cm);
        assert_eq!(cm0 * cm1, cm0_x_cm1);
        assert_eq!(cm1 * m, cm1 * cm);
        assert_eq!(-cm0, cm31!(P - 1, P - 2));
        assert_eq!(cm0 - cm1, cm31!(P - 3, P - 3));
        assert_eq!(cm1 - m, cm1 - cm);
        assert_eq!(cm0_x_cm1 / cm1, cm31!(1, 2));
        assert_eq!(cm1 / m, cm1 / cm);
    }

    #[test]
    fn test_into_slice() {
        let mut rng = SmallRng::seed_from_u64(0);
        let x = (0..100).map(|_| rng.gen()).collect::<Vec<CM31>>();

        let slice = CM31::into_slice(&x);

        for i in 0..100 {
            let corresponding_sub_slice = &slice[i * 8..(i + 1) * 8];
            assert_eq!(
                x[i],
                cm31!(
                    u32::from_le_bytes(corresponding_sub_slice[..4].try_into().unwrap()),
                    u32::from_le_bytes(corresponding_sub_slice[4..].try_into().unwrap())
                )
            )
        }
    }
}
