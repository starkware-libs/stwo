use std::fmt::Display;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

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

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CirclePoint {
    pub x: QM31,
    pub y: QM31,
}

impl CirclePoint {
    pub fn zero() -> Self {
        Self {
            x: QM31::one(),
            y: QM31::zero(),
        }
    }

    pub fn double(&self) -> Self {
        *self + *self
    }

    pub fn order_bits(&self) -> usize {
        let mut res = 0;
        let mut cur = *self;
        while cur != Self::zero() {
            cur = cur.double();
            res += 1;
        }
        res
    }

    pub fn mul(&self, mut scalar: u128) -> CirclePoint {
        let mut res = Self::zero();
        let mut cur = *self;
        while scalar > 0 {
            if scalar & 1 == 1 {
                res = res + cur;
            }
            cur = cur.double();
            scalar >>= 1;
        }
        res
    }

    pub fn repeated_double(&self, n: usize) -> Self {
        let mut res = *self;
        for _ in 0..n {
            res = res.double();
        }
        res
    }

    pub fn conjugate(&self) -> CirclePoint {
        Self {
            x: self.x,
            y: -self.y,
        }
    }

    pub fn antipode(&self) -> CirclePoint {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl Add for CirclePoint {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let x = self.x * rhs.x - self.y * rhs.y;
        let y = self.x * rhs.y + self.y * rhs.x;
        Self { x, y }
    }
}

impl Neg for CirclePoint {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.conjugate()
    }
}

impl Sub for CirclePoint {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Display for CirclePoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "x = {}, y = {}", self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::{CirclePoint, P4, QM31};
    use crate::core::fields::m31::{M31, P};
    use crate::core::fields::Field;

    #[test]
    fn test_ops() {
        let qm0 = QM31::from_u32_unchecked(1, 2, 3, 4);
        let qm1 = QM31::from_u32_unchecked(4, 5, 6, 7);
        let m = M31::from_u32_unchecked(8);
        let qm = QM31::from(m);
        let qm0_x_qm1 = QM31::from_u32_unchecked(P - 106, 38, P - 16, 50);

        assert_eq!(qm0 + qm1, QM31::from_u32_unchecked(5, 7, 9, 11));
        assert_eq!(qm1 + m, qm1 + qm);
        assert_eq!(qm0 * qm1, qm0_x_qm1);
        assert_eq!(qm1 * m, qm1 * qm);
        assert_eq!(-qm0, QM31::from_u32_unchecked(P - 1, P - 2, P - 3, P - 4));
        assert_eq!(
            qm0 - qm1,
            QM31::from_u32_unchecked(P - 3, P - 3, P - 3, P - 3)
        );
        assert_eq!(qm1 - m, qm1 - qm);
        assert_eq!(qm0_x_qm1 / qm1, QM31::from_u32_unchecked(1, 2, 3, 4));
        assert_eq!(qm1 / m, qm1 / qm);
    }

    fn is_circle_generator(x: CirclePoint) -> bool {
        let prime_factors: Vec<u128> = vec![2, 3, 5, 7, 11, 31, 151, 331, 733, 1709, 368140581013];
        for p in prime_factors {
            if x.mul((P4 - 1) / p) == CirclePoint::zero() {
                return false;
            }
        }
        x.mul(P4 - 1) == CirclePoint::zero()
    }

    #[test]
    fn test_qm_circle_point() {
        let ax = M31::from_u32_unchecked(1);
        let bx = M31::from_u32_unchecked(0);
        let cy = M31::from_u32_unchecked(0);
        let mut points_on_circle = 0;
        let mut generators = 0;
        for i in 0..1000 {
            let s = M31::from_u32_unchecked(i);
            let ay = ((s.square() + s - M31::from_u32_unchecked(1))
                / (s.pow(4)
                    + M31::from_u32_unchecked(2) * s.square()
                    + M31::from_u32_unchecked(1)))
            .sqrt()
            .unwrap_or(M31::from_u32_unchecked(0));
            if ay == M31::from_u32_unchecked(0) {
                continue;
            }
            let by = s * ay;
            let dy = (-(ay * by)
                / (by.square() - ay * by - ay.square() - M31::from_u32_unchecked(1)))
            .sqrt()
            .unwrap_or(M31::from_u32_unchecked(0));
            if ay == M31::from_u32_unchecked(0) {
                continue;
            }
            let cx = by * dy;
            let dx = -ay * dy;

            let x = QM31::from_u32_unchecked(ax.0, bx.0, cx.0, dx.0);
            let y = QM31::from_u32_unchecked(ay.0, by.0, cy.0, dy.0);
            let on_circle = x.square() + y.square() == QM31::from_u32_unchecked(1, 0, 0, 0);

            if on_circle {
                points_on_circle += 1;
                let point = CirclePoint { x, y };
                if is_circle_generator(point) {
                    println!("generator: {}", point);
                    generators += 1;
                }
            }
        }
        println!("points on circle: {}", points_on_circle);
        println!("generators: {}", generators);
    }
}
