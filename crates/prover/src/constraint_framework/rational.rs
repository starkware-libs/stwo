use std::fmt::{Display, Formatter};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use num_traits::{One, Zero};

use crate::constraint_framework::poly::Polynomial;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::utils::Fraction;

pub type Rational<F> = Fraction<Polynomial<F>, Polynomial<F>>;

impl<F> Rational<F>
where
    F: From<BaseField> + One + AddAssign + Clone + Add<Output = F>,
{
    /// Returns the rational function x_ind / 1.
    pub fn from_var_index(ind: usize) -> Rational<F> {
        Self::new(Polynomial::from_var_index(ind), Polynomial::one())
    }
}

impl<F> From<Polynomial<F>> for Rational<F>
where
    F: From<BaseField> + One + AddAssign + Clone + Add<Output = F>,
{
    fn from(polynomial: Polynomial<F>) -> Self {
        Fraction::new(polynomial, Polynomial::one())
    }
}

impl<F> From<F> for Rational<F>
where
    F: From<BaseField> + One + AddAssign + Clone + Add<Output = F>,
{
    fn from(value: F) -> Self {
        Fraction::new(value.into(), Polynomial::one())
    }
}

impl<F> AddAssign<BaseField> for Rational<F>
where
    F: From<BaseField> + One + Clone + Mul<Output = F> + AddAssign + Add<Output = F> + Zero,
{
    fn add_assign(&mut self, other: BaseField) {
        self.numerator = self.numerator.clone() + (self.denominator.clone() * other);
    }
}

impl<F> Mul<BaseField> for Rational<F>
where
    F: From<BaseField> + One + Clone + Mul<Output = F> + AddAssign + Add<Output = F> + Zero,
{
    type Output = Rational<F>;

    fn mul(self, other: BaseField) -> Self::Output {
        Self::new(self.numerator * other, self.denominator)
    }
}

impl<F> Sub<BaseField> for Rational<F>
where
    F: From<BaseField>
        + One
        + Clone
        + Mul<Output = F>
        + AddAssign
        + Add<Output = F>
        + Zero
        + Neg<Output = F>,
{
    type Output = Rational<F>;

    fn sub(self, other: BaseField) -> Self::Output {
        Self::new(
            self.numerator - self.denominator.clone() * other,
            self.denominator,
        )
    }
}

impl From<Rational<BaseField>> for Rational<SecureField> {
    fn from(rational: Rational<BaseField>) -> Self {
        Fraction::new(rational.numerator.into(), rational.denominator.into())
    }
}

impl Add<SecureField> for Rational<BaseField> {
    type Output = Rational<SecureField>;
    fn add(self, other: SecureField) -> Self::Output {
        Rational::<SecureField>::from(self) + Rational::<SecureField>::from(other)
    }
}

impl Mul<SecureField> for Rational<BaseField> {
    type Output = Rational<SecureField>;
    fn mul(self, other: SecureField) -> Self::Output {
        Rational::<SecureField>::from(self) * Rational::<SecureField>::from(other)
    }
}

impl<F> Add<F> for Rational<F>
where
    F: From<BaseField> + One + AddAssign + Clone + Add<Output = F> + Mul<Output = F> + Zero,
{
    type Output = Rational<F>;
    fn add(self, other: F) -> Rational<F> {
        Self::new(
            self.numerator + self.denominator.clone() * Polynomial::<F>::from(other),
            self.denominator,
        )
    }
}

impl Sub<SecureField> for Rational<SecureField> {
    type Output = Rational<SecureField>;
    fn sub(self, other: SecureField) -> Self::Output {
        self - Rational::<SecureField>::from(other)
    }
}

impl Mul<SecureField> for Rational<SecureField> {
    type Output = Rational<SecureField>;
    fn mul(self, other: SecureField) -> Self::Output {
        self * Rational::<SecureField>::from(other)
    }
}

impl Add<Rational<BaseField>> for Rational<SecureField> {
    type Output = Rational<SecureField>;
    fn add(self, other: Rational<BaseField>) -> Self::Output {
        self + Rational::<SecureField>::from(other)
    }
}

impl Mul<Rational<BaseField>> for Rational<SecureField> {
    type Output = Rational<SecureField>;
    fn mul(self, other: Rational<BaseField>) -> Self::Output {
        self * Rational::<SecureField>::from(other)
    }
}

impl<F> Div for Rational<F>
where
    F: From<BaseField> + One + AddAssign + Clone + Add<Output = F> + Mul<Output = F> + Zero,
{
    type Output = Rational<F>;
    fn div(self, other: Rational<F>) -> Rational<F> {
        Self::new(
            self.numerator.clone() * other.denominator.clone(),
            self.denominator.clone() * other.numerator.clone(),
        )
    }
}

impl<F> Display for Rational<F>
where
    F: Display + From<BaseField> + Clone + Zero + One + PartialEq,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) / ({})", self.numerator, self.denominator)
    }
}
