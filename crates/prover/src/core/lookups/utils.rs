use std::iter::{zip, Sum};
use std::ops::{Add, AddAssign, Deref, Mul, MulAssign, Neg, Sub};

use num_traits::{One, Zero};

use crate::core::fields::qm31::SecureField;
use crate::core::fields::{ExtensionOf, Field, FieldExpOps};

/// Univariate polynomial stored as coefficients in the monomial basis.
#[derive(Debug, Clone)]
pub struct UnivariatePoly<F: Field>(Vec<F>);

impl<F: Field> UnivariatePoly<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        let mut polynomial = Self(coeffs);
        polynomial.truncate_leading_zeros();
        polynomial
    }

    pub fn eval_at_point(&self, x: F) -> F {
        horner_eval(&self.0, x)
    }

    // <https://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Polynomial_interpolation>
    pub fn interpolate_lagrange(xs: &[F], ys: &[F]) -> Self {
        assert_eq!(xs.len(), ys.len());

        let mut coeffs = Self::zero();

        for (i, (xi, yi)) in zip(xs, ys).enumerate() {
            let mut prod = yi.clone();

            for (j, xj) in xs.iter().enumerate() {
                if i != j {
                    prod /= xi.clone() - xj.clone();
                }
            }

            let mut term = Self::new(vec![prod]);

            for (j, xj) in xs.iter().enumerate() {
                if i != j {
                    term = term * (Self::x() - Self::new(vec![xj.clone()]));
                }
            }

            coeffs = coeffs + term;
        }

        coeffs.truncate_leading_zeros();

        coeffs
    }

    pub fn degree(&self) -> usize {
        let mut coeffs = self.0.iter().rev();
        let _ = (&mut coeffs).take_while(|v| v.is_zero());
        coeffs.len().saturating_sub(1)
    }

    fn x() -> Self {
        Self(vec![F::zero(), F::one()])
    }

    fn truncate_leading_zeros(&mut self) {
        while self.0.last() == Some(&F::zero()) {
            self.0.pop();
        }
    }
}

impl<F: Field> From<F> for UnivariatePoly<F> {
    fn from(value: F) -> Self {
        Self::new(vec![value])
    }
}

impl<F: Field> Mul<F> for UnivariatePoly<F> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self {
        self.0.iter_mut().for_each(|coeff| *coeff *= rhs.clone());
        self
    }
}

impl<F: Field> Mul for UnivariatePoly<F> {
    type Output = Self;

    fn mul(mut self, mut rhs: Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }

        self.truncate_leading_zeros();
        rhs.truncate_leading_zeros();

        let mut res = vec![F::zero(); self.0.len() + rhs.0.len() - 1];

        for (i, coeff_a) in self.0.into_iter().enumerate() {
            for (j, coeff_b) in rhs.0.iter().enumerate() {
                res[i + j] += coeff_a.clone() * coeff_b.clone();
            }
        }

        Self::new(res)
    }
}

impl<F: Field> Add for UnivariatePoly<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let n = self.0.len().max(rhs.0.len());
        let mut res = Vec::new();

        for i in 0..n {
            res.push(match (self.0.get(i), rhs.0.get(i)) {
                (Some(a), Some(b)) => a.clone() + b.clone(),
                (Some(a), None) | (None, Some(a)) => a.clone(),
                _ => unreachable!(),
            })
        }

        Self(res)
    }
}

impl<F: Field> Sub for UnivariatePoly<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<F: Field> Neg for UnivariatePoly<F> {
    type Output = Self;

    fn neg(self) -> Self {
        Self(self.0.into_iter().map(|v| -v).collect())
    }
}

impl<F: Field> Zero for UnivariatePoly<F> {
    fn zero() -> Self {
        Self(vec![])
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(F::is_zero)
    }
}

impl<F: Field> Deref for UnivariatePoly<F> {
    type Target = [F];

    fn deref(&self) -> &[F] {
        &self.0
    }
}

/// Evaluates univariate polynomial using [Horner's method].
///
/// [Horner's method]: https://en.wikipedia.org/wiki/Horner%27s_method
pub fn horner_eval<F: Field>(coeffs: &[F], x: F) -> F {
    coeffs
        .iter()
        .rfold(F::zero(), |acc, coeff| acc * x.clone() + coeff.clone())
}

/// Returns `v_0 + alpha * v_1 + ... + alpha^(n-1) * v_{n-1}`.
pub fn random_linear_combination(v: &[SecureField], alpha: SecureField) -> SecureField {
    horner_eval(v, alpha)
}

/// Evaluates the lagrange kernel of the boolean hypercube.
///
/// The lagrange kernel of the boolean hypercube is a multilinear extension of the function that
/// when given `x, y` in `{0, 1}^n` evaluates to 1 if `x = y`, and evaluates to 0 otherwise.
pub fn eq<F: Field>(x: &[F], y: &[F]) -> F {
    assert_eq!(x.len(), y.len());
    zip(x, y)
        .map(|(xi, yi)| xi.clone() * yi.clone() + (F::one() - xi.clone()) * (F::one() - yi.clone()))
        .product()
}

/// Computes `eq(0, assignment) * eval0 + eq(1, assignment) * eval1`.
pub fn fold_mle_evals<F>(assignment: SecureField, eval0: F, eval1: F) -> SecureField
where
    F: Field,
    SecureField: ExtensionOf<F>,
{
    assignment * (eval1 - eval0.clone()) + eval0.clone()
}

/// Projective fraction.
#[derive(Debug, Clone)]
pub struct Fraction<N, D> {
    pub numerator: N,
    pub denominator: D,
}

impl<N: Copy, D: Copy> Copy for Fraction<N, D> {}

impl<N, D> Fraction<N, D> {
    pub fn new(numerator: N, denominator: D) -> Self {
        Self {
            numerator,
            denominator,
        }
    }
}

impl<N, D> Add for Fraction<N, D>
where
    N: Clone,
    D: Add<Output = D> + Add<N, Output = D> + Mul<N, Output = D> + Mul<Output = D> + Clone,
{
    type Output = Fraction<D, D>;

    fn add(self, rhs: Self) -> Fraction<D, D> {
        Fraction {
            numerator: rhs.denominator.clone() * self.numerator.clone()
                + self.denominator.clone() * rhs.numerator.clone(),
            denominator: self.denominator * rhs.denominator,
        }
    }
}

impl<N, D> Sub for Fraction<N, D>
where
    N: Clone,
    D: Sub<Output = D> + Add<N, Output = D> + Mul<N, Output = D> + Mul<Output = D> + Clone,
{
    type Output = Fraction<D, D>;

    fn sub(self, rhs: Self) -> Fraction<D, D> {
        Fraction {
            numerator: rhs.denominator.clone() * self.numerator.clone()
                - self.denominator.clone() * rhs.numerator.clone(),
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

impl<N: Neg<Output = N>, D> Neg for Fraction<N, D> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.numerator, self.denominator)
    }
}

impl<N, D> AddAssign for Fraction<N, D>
where
    N: Mul<D, Output = N> + Add<Output = N> + Clone,
    D: Mul<Output = D> + Clone,
{
    fn add_assign(&mut self, rhs: Self) {
        self.numerator = self.numerator.clone() * rhs.denominator.clone()
            + rhs.numerator.clone() * self.denominator.clone();
        self.denominator = self.denominator.clone() * rhs.denominator;
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

impl<N: Mul<Output = N>, D: Mul<Output = D>> Mul for Fraction<N, D> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.numerator * rhs.numerator,
            self.denominator * rhs.denominator,
        )
    }
}

impl<N: MulAssign, D: MulAssign> MulAssign for Fraction<N, D> {
    fn mul_assign(&mut self, rhs: Self) {
        self.numerator *= rhs.numerator;
        self.denominator *= rhs.denominator;
    }
}

impl<N: One, D: One> One for Fraction<N, D> {
    fn one() -> Self {
        Self::new(N::one(), D::one())
    }
}

impl<F: Zero + Clone + One + MulAssign> FieldExpOps for Fraction<F, F> {
    fn inverse(&self) -> Self {
        assert!(!self.denominator.is_zero());
        Self::new(self.denominator.clone(), self.numerator.clone())
    }
}

/// Represents the fraction `1 / x`
pub struct Reciprocal<T> {
    x: T,
}

impl<T> Reciprocal<T> {
    pub fn new(x: T) -> Self {
        Self { x }
    }
}

impl<T: Add<Output = T> + Mul<Output = T> + Clone> Add for Reciprocal<T> {
    type Output = Fraction<T, T>;

    fn add(self, rhs: Self) -> Fraction<T, T> {
        // `1/a + 1/b = (a + b)/(a * b)`
        Fraction {
            numerator: self.x.clone() + rhs.x.clone(),
            denominator: self.x * rhs.x,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use num_traits::{One, Zero};

    use super::{horner_eval, UnivariatePoly};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::FieldExpOps;
    use crate::core::lookups::utils::{eq, Fraction};

    #[test]
    fn lagrange_interpolation_works() {
        let xs = [5, 1, 3, 9].map(BaseField::from);
        let ys = [1, 2, 3, 4].map(BaseField::from);

        let poly = UnivariatePoly::interpolate_lagrange(&xs, &ys);

        for (x, y) in zip(xs, ys) {
            assert_eq!(poly.eval_at_point(x), y, "mismatch for x={x}");
        }
    }

    #[test]
    fn horner_eval_works() {
        let coeffs = [BaseField::from(9), BaseField::from(2), BaseField::from(3)];
        let x = BaseField::from(7);

        let eval = horner_eval(&coeffs, x);

        assert_eq!(eval, coeffs[0] + coeffs[1] * x + coeffs[2] * x.square());
    }

    #[test]
    fn eq_identical_hypercube_points_returns_one() {
        let zero = SecureField::zero();
        let one = SecureField::one();
        let a = &[one, zero, one];

        let eq_eval = eq(a, a);

        assert_eq!(eq_eval, one);
    }

    #[test]
    fn eq_different_hypercube_points_returns_zero() {
        let zero = SecureField::zero();
        let one = SecureField::one();
        let a = &[one, zero, one];
        let b = &[one, zero, zero];

        let eq_eval = eq(a, b);

        assert_eq!(eq_eval, zero);
    }

    #[test]
    #[should_panic]
    fn eq_different_size_points() {
        let zero = SecureField::zero();
        let one = SecureField::one();

        eq(&[zero, one], &[zero]);
    }

    #[test]
    fn fraction_addition_works() {
        let a = Fraction::new(BaseField::from(1), BaseField::from(3));
        let b = Fraction::new(BaseField::from(2), BaseField::from(6));

        let Fraction {
            numerator,
            denominator,
        } = a + b;

        assert_eq!(
            numerator / denominator,
            BaseField::from(2) / BaseField::from(3)
        );
    }
}
