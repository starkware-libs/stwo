use std::ops::{Add, AddAssign};

use crate::core::fields::Field;

#[derive(Copy, Clone, Debug)]
pub struct Fraction<T: Field> {
    pub numer: T,
    pub denom: T,
}

impl<T: Field> Fraction<T> {
    pub fn new(numer: T, denom: T) -> Self {
        Self { numer, denom }
    }
}

impl<T: Field> Add for Fraction<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.numer * rhs.denom + self.denom * rhs.numer,
            self.denom * rhs.denom,
        )
    }
}

impl<T: Field> AddAssign<Fraction<T>> for Fraction<T> {
    fn add_assign(&mut self, rhs: Fraction<T>) {
        *self = *self + rhs;
    }
}

impl<T: Field> Add<T> for Fraction<T> {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        Self::new(self.numer * rhs + self.denom, self.denom * rhs)
    }
}

impl<T: Field> AddAssign<T> for Fraction<T> {
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs;
    }
}

impl<T: Field> PartialEq for Fraction<T> {
    fn eq(&self, other: &Self) -> bool {
        self.numer * other.denom == self.denom * other.numer
    }
}

#[cfg(test)]
mod tests {
    use num_traits::One;

    use super::Fraction;
    use crate::core::fields::m31::{M31, P};

    #[test]
    fn test_fracion() {
        let a = Fraction::new(M31::one(), M31::from_u32_unchecked(P - 1));
        let b = Fraction::new(M31::one(), M31::from_u32_unchecked(2));

        let res = a + b;
        let d = res + res;
        let expected = Fraction::new(M31::from_u32_unchecked(1), M31::from_u32_unchecked(P - 2));

        assert_eq!(res, expected);
        assert_eq!(d, expected + expected);
    }
}
