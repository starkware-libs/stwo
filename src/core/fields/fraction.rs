use num_traits::{Num, NumAssign};
use std::ops::{Add, AddAssign};

#[derive(Copy, Clone, Debug)]
pub struct Frac<T> {
    pub numer: T,
    pub denom: T,
}

impl<T> Frac<T>
where
    T: Copy,
{
    pub fn new(numer: T, denom: T) -> Self {
        Self { numer, denom }
    }
}

impl<T: Num + Copy> Add for Frac<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.numer * rhs.denom + self.denom * rhs.numer,
            self.denom * rhs.denom,
        )
    }
}

impl<T: NumAssign + Copy> AddAssign<Frac<T>> for Frac<T> {
    fn add_assign(&mut self, rhs: Frac<T>) {
        *self = *self + rhs;
    }
}

impl<T: Num + Copy> Add<T> for Frac<T> {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        Self::new(self.numer * rhs + self.denom, self.denom * rhs)
    }
}

impl<T: NumAssign + Copy> AddAssign<T> for Frac<T> {
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs;
    }
}

impl<T: Num + Copy> PartialEq for Frac<T> {
    fn eq(&self, other: &Self) -> bool {
        self.numer * other.denom == self.denom * other.numer
    }
}

#[test]
fn test_frac() {
    use super::m31::{M31, P};
    use num_traits::One;

    let a = Frac::new(M31::one(), M31::from_u32_unchecked(P - 1));
    let b = Frac::new(M31::one(), M31::from_u32_unchecked(2));
    let mut c = a + b;

    let res = Frac::new(M31::from_u32_unchecked(1), M31::from_u32_unchecked(P - 2));
    assert_eq!(c, res);

    c += c;
    assert_eq!(c, res + res);
}
