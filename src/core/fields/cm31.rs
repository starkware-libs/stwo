use std::fmt::Display;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use crate::core::fields::m31::M31;
use crate::{impl_extension_field, impl_field};

pub const P2: u64 = 4611686014132420609; // (2 ** 31 - 1) ** 2

/// Complex extension field of M31.
/// Equivalent to M31\[x\] over (x^2 + 1) as the irreducible polynomial.
/// Represented as (a, b) of a + bi.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CM31(M31, M31);

impl_field!(CM31, P2);
impl_extension_field!(CM31, M31);

impl CM31 {
    pub const fn from_u32_unchecked(a: u32, b: u32) -> CM31 {
        Self(M31::from_u32_unchecked(a), M31::from_u32_unchecked(b))
    }
}

impl Display for CM31 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} + {}i", self.0, self.1)
    }
}

impl Mul for CM31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i.
        Self(
            self.0 * rhs.0 - self.1 * rhs.1,
            self.0 * rhs.1 + self.1 * rhs.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::ThreadRng;
    use rand::Rng;

    use super::CM31;
    use crate::core::fields::m31::{M31, P};
    use crate::core::fields::Field;

    #[test]
    fn test_ops() {
        let cm0 = CM31::from_u32_unchecked(1, 2);
        let cm1 = CM31::from_u32_unchecked(4, 5);
        let m = M31::from_u32_unchecked(8);
        let cm = CM31::from(m);
        let cm0_x_cm1 = CM31::from_u32_unchecked(P - 6, 13);

        assert_eq!(cm0 + cm1, CM31::from_u32_unchecked(5, 7));
        assert_eq!(cm1 + m, cm1 + cm);
        assert_eq!(cm0 * cm1, cm0_x_cm1);
        assert_eq!(cm1 * m, cm1 * cm);
        assert_eq!(-cm0, CM31::from_u32_unchecked(P - 1, P - 2));
        assert_eq!(cm0 - cm1, CM31::from_u32_unchecked(P - 3, P - 3));
        assert_eq!(cm1 - m, cm1 - cm);
        assert_eq!(cm0_x_cm1 / cm1, CM31::from_u32_unchecked(1, 2));
        assert_eq!(cm1 / m, cm1 / cm);
    }

    #[test]
    fn test_cm_circle_point() {
        pub fn get_random_element(rng: &mut ThreadRng) -> M31 {
            M31::from_u32_unchecked(rng.gen::<u32>() % P)
        }
        let mut rng = rand::thread_rng();
        let mut cnt = 0;
        for _ in 1..1000 {
            let ax = get_random_element(&mut rng);
            let bx = get_random_element(&mut rng);

            let b = -M31::from_u32_unchecked(1) + ax.square() - bx.square();
            let delta = (b.square() + M31::from_u32_unchecked(4) * ax.square() * bx.square())
                .sqrt()
                .unwrap_or(M31::from_u32_unchecked(1));
            let ay = ((-b + delta) / M31::from_u32_unchecked(2))
                .sqrt()
                .unwrap_or(M31::from_u32_unchecked(1));
            let by = -(ax * bx) / ay;
            let x = CM31::from_u32_unchecked(ax.0, bx.0);
            let y = CM31::from_u32_unchecked(ay.0, by.0);
            let res = x.square() + y.square();
            if res == CM31::from_u32_unchecked(1, 0) {
                cnt += 1;
            }
        }
        dbg!(cnt);
    }
}
