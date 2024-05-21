use std::fmt::{Debug, Display};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use super::{ComplexConjugate, FieldExpOps};
use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::M31;
use crate::{impl_extension_field, impl_field};

pub const P4: u128 = 21267647892944572736998860269687930881; // (2 ** 31 - 1) ** 4
pub const R: CM31 = CM31::from_u32_unchecked(2, 1);

/// Extension field of CM31.
/// Equivalent to CM31\[x\] over (x^2 - 2 - i) as the irreducible polynomial.
/// Represented as ((a, b), (c, d)) of (a + bi) + (c + di)u.
#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QM31(pub CM31, pub CM31);
pub type SecureField = QM31;

impl_field!(QM31, P4);
impl_extension_field!(QM31, CM31);

impl QM31 {
    pub const fn from_u32_unchecked(a: u32, b: u32, c: u32, d: u32) -> Self {
        Self(
            CM31::from_u32_unchecked(a, b),
            CM31::from_u32_unchecked(c, d),
        )
    }

    /// Converts four `M31` into a `QM31`.
    ///
    /// # Arguments
    ///
    /// * `a` - The first `M31` instance.
    /// * `b` - The second `M31` instance.
    /// * `c` - The third `M31` instance.
    /// * `d` - The fourth `M31` instance.
    ///
    /// # Returns
    ///
    /// A new `QM31` instance of the `a`, `b`, `c` and `d` in the form (a + bi) + (c + di)u.
    ///
    /// # Example
    ///
    /// ```
    ///     use stwo_prover::core::fields::m31::M31;
    ///     use stwo_prover::core::fields::cm31::CM31;
    ///     use stwo_prover::core::fields::qm31::QM31;
    /// 
    ///     let a = M31(1);
    ///     let b = M31(2);
    ///     let c = M31(3);
    ///     let d = M31(4);
    /// 
    ///     let qm = QM31::from_m31(a, b, c, d);
    ///     println!("QM31 value: {:?}", qm);
    /// ```
    pub fn from_m31(a: M31, b: M31, c: M31, d: M31) -> Self {
        Self(CM31::from_m31(a, b), CM31::from_m31(c, d))
    }

    /// Converts an array of four `M31` elements into a `QM31`.
    ///
    /// # Arguments
    ///
    /// * `array` - An array of four `M31` instance.
    ///
    /// # Returns
    ///
    /// A new `QM31` instance from the array .
    ///
    /// # Example
    ///
    /// ```
    ///     use stwo_prover::core::fields::m31::M31;
    ///     use stwo_prover::core::fields::qm31::QM31;
    /// 
    ///     let a = M31(1);
    ///     let b = M31(2);
    ///     let c = M31(3);
    ///     let d = M31(4);
    ///     let array: [M31; 4] = [a, b, c, d];
    /// 
    ///     let qm = QM31::from_m31_array(array);
    ///     println!("QM31 value: {:?}", qm);
    /// ```
    pub fn from_m31_array(array: [M31; 4]) -> Self {
        Self::from_m31(array[0], array[1], array[2], array[3])
    }

    /// Converts a `QM31` instance into an array of `M31`.
    ///
    /// # Arguments
    ///
    /// * `self` - The `QM31` instance.
    ///
    /// # Returns
    ///
    /// An array of `M31`.
    ///
    /// # Example
    ///
    /// ```
    ///     use stwo_prover::core::fields::m31::M31;
    ///     use stwo_prover::core::fields::qm31::QM31;
    /// 
    ///     let a = M31(1);
    ///     let b = M31(2);
    ///     let c = M31(3);
    ///     let d = M31(4);
    /// 
    ///     let qm = QM31::from_m31(a, b, c, d);
    /// 
    ///     let m31_array = qm.to_m31_array();
    ///     println!("M31 array values: {:?}", m31_array);
    /// ```
    pub fn to_m31_array(self) -> [M31; 4] {
        [self.0 .0, self.0 .1, self.1 .0, self.1 .1]
    }
}

/// Implementation of the `Display` trait for `QM31` for custom formatting when displayed.
impl Display for QM31 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) + ({})u", self.0, self.1)
    }
}

/// Implementation of the `Debug` trait for `QM31` for custom formatting when displayed.
impl Debug for QM31 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) + ({})u", self.0, self.1)
    }
}

/// Implementation of the `Mul` trait for `QM31`, allowing multiplication of two `QM31` instances.
impl Mul for QM31 {
    type Output = Self;

    /// Multiply two `QM31` instances.
    ///
    /// # Arguments
    ///
    /// * `self` - The first `QM31` instance.
    /// * `rhs` - The second `QM31` instance to be multiplied by.
    ///
    /// # Returns
    ///
    /// A new `QM31` instance with the multiplied value.
    ///
    /// # Examples
    /// 
    /// ```
    ///     use stwo_prover::core::fields::m31::M31;
    ///     use stwo_prover::core::fields::cm31::CM31;
    ///     use stwo_prover::core::fields::qm31::QM31;
    /// 
    ///     let a = M31(1);
    ///     let b = M31(2);
    ///     let c = M31(3);
    ///     let d = M31(4);
    /// 
    ///     let qm = QM31::from_m31(a, b, c, d);
    ///     let prod = qm * qm;
    ///     println!("prod value: {:?}", prod)
    /// ```
    fn mul(self, rhs: Self) -> Self::Output {
        // (a + bu) * (c + du) = (ac + rbd) + (ad + bc)u.
        Self(
            self.0 * rhs.0 + R * self.1 * rhs.1,
            self.0 * rhs.1 + self.1 * rhs.0,
        )
    }
}

impl FieldExpOps for QM31 {
    /// The `inverse` function computes the inverse of QM31.
    /// The inverse of an element is the number you multiply it with to get the finite field element 1.
    ///     
    /// # Arguments
    ///
    /// * `self` - The `QM31` instance.
    ///
    /// # Returns
    ///
    /// A new `QM31` instance with the inverse value.
    ///
    /// # Examples
    /// 
    /// ```
    ///     use stwo_prover::core::fields::m31::M31;
    ///     use stwo_prover::core::fields::qm31::QM31;
    ///     use crate::stwo_prover::core::fields::FieldExpOps;
    /// 
    ///     let a = M31(1);
    ///     let b = M31(2);
    ///     let c = M31(3);
    ///     let d = M31(4);
    /// 
    ///     let qm = QM31::from_m31(a, b, c, d);
    ///
    ///     let qm_inv = qm.inverse();
    ///     println!("inverse value: {:?}", qm_inv);
    /// ```
    fn inverse(&self) -> Self {
        assert!(!self.is_zero(), "0 has no inverse");
        // (a + bu)^-1 = (a - bu) / (a^2 - (2+i)b^2).
        let b2 = self.1.square();
        let ib2 = CM31(-b2.1, b2.0);
        let denom = self.0.square() - (b2 + b2 + ib2);
        let denom_inverse = denom.inverse();
        Self(self.0 * denom_inverse, -self.1 * denom_inverse)
    }
}

#[cfg(test)]
#[macro_export]
macro_rules! qm31 {
    ($m0:expr, $m1:expr, $m2:expr, $m3:expr) => {{
        use $crate::core::fields::qm31::QM31;
        QM31::from_u32_unchecked($m0, $m1, $m2, $m3)
    }};
}

#[cfg(test)]
mod tests {
    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::QM31;
    use crate::core::fields::m31::P;
    use crate::core::fields::{FieldExpOps, IntoSlice};
    use crate::m31;

     #[test]
    fn test_inverse() {
        let qm = qm31!(1, 2, 3, 4);
        let qm_inv = qm.inverse();
        assert_eq!(qm * qm_inv, QM31::one());
    }

    #[test]
    fn test_ops() {
        let qm0 = qm31!(1, 2, 3, 4);
        let qm1 = qm31!(4, 5, 6, 7);
        let m = m31!(8);
        let qm = QM31::from(m);
        let qm0_x_qm1 = qm31!(P - 71, 93, P - 16, 50);

        assert_eq!(qm0 + qm1, qm31!(5, 7, 9, 11));
        assert_eq!(qm1 + m, qm1 + qm);
        assert_eq!(qm0 * qm1, qm0_x_qm1);
        assert_eq!(qm1 * m, qm1 * qm);
        assert_eq!(-qm0, qm31!(P - 1, P - 2, P - 3, P - 4));
        assert_eq!(qm0 - qm1, qm31!(P - 3, P - 3, P - 3, P - 3));
        assert_eq!(qm1 - m, qm1 - qm);
        assert_eq!(qm0_x_qm1 / qm1, qm31!(1, 2, 3, 4));
        assert_eq!(qm1 / m, qm1 / qm);
    }

    #[test]
    fn test_into_slice() {
        let mut rng = SmallRng::seed_from_u64(0);
        let x = (0..100).map(|_| rng.gen()).collect::<Vec<QM31>>();

        let slice = QM31::into_slice(&x);

        for i in 0..100 {
            let corresponding_sub_slice = &slice[i * 16..(i + 1) * 16];
            assert_eq!(
                x[i],
                qm31!(
                    u32::from_le_bytes(corresponding_sub_slice[..4].try_into().unwrap()),
                    u32::from_le_bytes(corresponding_sub_slice[4..8].try_into().unwrap()),
                    u32::from_le_bytes(corresponding_sub_slice[8..12].try_into().unwrap()),
                    u32::from_le_bytes(corresponding_sub_slice[12..16].try_into().unwrap())
                )
            )
        }
    }
}
