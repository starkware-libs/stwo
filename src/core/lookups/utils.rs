use std::iter::zip;
use std::ops::{Add, Deref, Mul, Neg, Sub};

use crate::core::fields::Field;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Polynomial<F: Field>(Vec<F>);

impl<F: Field> Polynomial<F> {
    fn x() -> Self {
        Self(vec![F::zero(), F::one()])
    }

    pub fn zero() -> Self {
        Self(vec![])
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|v| F::zero() == *v)
    }

    pub fn degree(&self) -> usize {
        let mut coeffs = self.0.iter().rev();
        let _ = (&mut coeffs).take_while(|v| v.is_zero());
        coeffs.len().saturating_sub(1)
    }

    pub fn new(coeffs: Vec<F>) -> Self {
        let mut polynomial = Self(coeffs);
        polynomial.truncate_leading_zeros();
        polynomial
    }

    fn truncate_leading_zeros(&mut self) {
        while self.0.last() == Some(&F::zero()) {
            self.0.pop();
        }
    }

    pub fn eval(&self, x: F) -> F {
        horner_eval(&self.0, x)
    }

    // https://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Polynomial_interpolation
    pub fn interpolate_lagrange(xs: &[F], ys: &[F]) -> Self {
        assert_eq!(xs.len(), ys.len());

        let mut coeffs = Polynomial::zero();

        for (i, (&xi, &yi)) in zip(xs, ys).enumerate() {
            let mut tmp = yi;

            for (j, &xj) in xs.iter().enumerate() {
                if i != j {
                    tmp /= xi - xj;
                }
            }

            let mut term = Polynomial::new(vec![tmp]);

            for (j, &xj) in xs.iter().enumerate() {
                if i != j {
                    term = term * (Self::x() - Self::new(vec![xj]));
                }
            }

            coeffs = coeffs + term;
        }

        coeffs.truncate_leading_zeros();
        coeffs
    }
}

impl<F: Field> Deref for Polynomial<F> {
    type Target = [F];

    fn deref(&self) -> &[F] {
        &self.0
    }
}

impl<F: Field> Mul for Polynomial<F> {
    type Output = Self;

    fn mul(mut self, mut rhs: Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }

        self.truncate_leading_zeros();
        rhs.truncate_leading_zeros();

        let mut res = vec![F::zero(); self.0.len() + rhs.0.len() - 1];

        for (i, coeff_a) in self.0.into_iter().enumerate() {
            for (j, &coeff_b) in rhs.0.iter().enumerate() {
                res[i + j] += coeff_a * coeff_b;
            }
        }

        Self::new(res)
    }
}

impl<F: Field> Mul<F> for Polynomial<F> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self {
        self.0.iter_mut().for_each(|coeff| *coeff *= rhs);
        self
    }
}

impl<F: Field> Add for Polynomial<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let n = self.0.len().max(rhs.0.len());
        let mut res = Vec::new();

        for i in 0..n {
            res.push(match (self.0.get(i), rhs.0.get(i)) {
                (Some(&a), Some(&b)) => a + b,
                (Some(&a), None) | (None, Some(&a)) => a,
                _ => unreachable!(),
            })
        }

        Self(res)
    }
}

impl<F: Field> Sub for Polynomial<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<F: Field> Neg for Polynomial<F> {
    type Output = Self;

    fn neg(self) -> Self {
        Self(self.0.into_iter().map(|v| -v).collect())
    }
}

/// Evaluates the lagrange kernel of the boolean hypercube.
///
/// The lagrange kernel of the boolean hyperbube is a multilinear extension of the function that
/// when given `x, y` in `{0, 1}^n` evaluates to 1 if `x = y`, and evaluates to 0 otherwise.
pub fn eq<F: Field>(x_assignments: &[F], y_assignments: &[F]) -> F {
    assert_eq!(x_assignments.len(), y_assignments.len());
    zip(x_assignments, y_assignments)
        .map(|(&xi, &wi)| xi * wi + (F::one() - xi) * (F::one() - wi))
        .product::<F>()
}

/// Evaluates univariate polynomial using [Horner's method].
///
/// [Horner's method]: https://en.wikipedia.org/wiki/Horner%27s_method
pub fn horner_eval<F: Field>(coeffs: &[F], x: F) -> F {
    coeffs
        .iter()
        .rfold(F::zero(), |acc, &coeff| acc * x + coeff)
}
