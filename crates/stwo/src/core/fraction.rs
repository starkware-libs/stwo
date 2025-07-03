use std::iter::Sum;
use std::ops::{Add, Mul};

use num_traits::{One, Zero};

/// Projective fraction.
#[derive(Debug, Clone, Copy)]
pub struct Fraction<N, D> {
    pub numerator: N,
    pub denominator: D,
}

impl<N, D> Fraction<N, D> {
    pub const fn new(numerator: N, denominator: D) -> Self {
        Self {
            numerator,
            denominator,
        }
    }
}

impl<N, D: Add<Output = D> + Add<N, Output = D> + Mul<N, Output = D> + Mul<Output = D> + Clone> Add
    for Fraction<N, D>
{
    type Output = Fraction<D, D>;

    fn add(self, rhs: Self) -> Fraction<D, D> {
        Fraction {
            numerator: rhs.denominator.clone() * self.numerator
                + self.denominator.clone() * rhs.numerator,
            denominator: self.denominator * rhs.denominator,
        }
    }
}

impl<N: Zero, D: One + Zero> Zero for Fraction<N, D>
where
    Self: Add<Output = Self>,
{
    fn zero() -> Self {
        Self {
            numerator: N::zero(),
            denominator: D::one(),
        }
    }

    fn is_zero(&self) -> bool {
        self.numerator.is_zero() && !self.denominator.is_zero()
    }
}

impl<N, D> Sum for Fraction<N, D>
where
    Self: Zero,
{
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let first = iter.next().unwrap_or_else(Self::zero);
        iter.fold(first, |a, b| a + b)
    }
}
