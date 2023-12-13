use std::fmt::Display;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use super::IntoSlice;
use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::M31;
use crate::{impl_extension_field, impl_field};

pub const P4: u128 = 21267647892944572736998860269687930881; // (2 ** 31 - 1) ** 4
pub const R: CM31 = CM31::from_u32_unchecked(1, 2);

/// Extension field of CM31.
/// Equivalent to CM31\[x\] over (x^2 - 1 - 2i) as the irreducible polynomial.
/// Represented as ((a, b), (c, d)) of (a + bi) + (c + di)u.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QM31(CM31, CM31);
pub type ExtensionField = QM31;

impl_field!(QM31, P4);
impl_extension_field!(QM31, CM31);

impl QM31 {
    pub const fn from_u32_unchecked(a: u32, b: u32, c: u32, d: u32) -> Self {
        Self(
            CM31::from_u32_unchecked(a, b),
            CM31::from_u32_unchecked(c, d),
        )
    }

    pub fn from_m31(a: M31, b: M31, c: M31, d: M31) -> Self {
        Self(CM31::from_m31(a, b), CM31::from_m31(c, d))
    }
}

impl Display for QM31 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) + ({})u", self.0, self.1)
    }
}

impl Mul for QM31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // (a + bu) * (c + du) = (ac + rbd) + (ad + bc)u.
        Self(
            self.0 * rhs.0 + R * self.1 * rhs.1,
            self.0 * rhs.1 + self.1 * rhs.0,
        )
    }
}

unsafe impl IntoSlice<u8> for QM31 {
    fn into_slice(sl: &[Self]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(sl.as_ptr() as *const u8, std::mem::size_of_val(sl)) }
    }
}

#[cfg(test)]
#[macro_export]
macro_rules! qm31 {
    ($m0:expr, $m1:expr, $m2:expr, $m3:expr) => {
        QM31::from_u32_unchecked($m0, $m1, $m2, $m3)
    };
}

#[cfg(test)]
mod tests {
    use super::QM31;
    use crate::core::fields::m31::{M31, P};
    use crate::m31;

    #[test]
    fn test_ops() {
        let qm0 = qm31!(1, 2, 3, 4);
        let qm1 = qm31!(4, 5, 6, 7);
        let m = m31!(8);
        let qm = QM31::from(m);
        let qm0_x_qm1 = qm31!(P - 106, 38, P - 16, 50);

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
}
