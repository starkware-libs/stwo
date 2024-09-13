use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg};

use itertools::Itertools;
use num_traits::{One, Zero};

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;

/// A monic monomial consists of a list of variables and their exponents.
#[derive(Debug, Hash, PartialEq, PartialOrd, Eq, Ord, Clone)]
pub struct Monomial {
    /// The variables of the monomial and their exponents.
    pub vars: BTreeMap<usize, usize>,
}

impl Monomial {
    pub fn degree(&self) -> usize {
        self.vars.values().sum()
    }

    fn default() -> Monomial {
        Monomial {
            vars: [(0, 0)].into(),
        }
    }
}

/// A polynomial consists of a list of monomials with coefficients.
#[derive(Debug, Hash, PartialEq, PartialOrd, Eq, Ord, Clone)]
pub struct Polynomial<F: From<BaseField>> {
    monomials: BTreeMap<Monomial, F>,
}

impl<F: One + From<BaseField>> From<Monomial> for Polynomial<F> {
    fn from(monomial: Monomial) -> Self {
        Self {
            monomials: [(monomial, F::one())].into(),
        }
    }
}

impl<F: From<BaseField>> From<F> for Polynomial<F>
where
    F: One + Clone,
{
    fn from(scalar: F) -> Self {
        Self {
            monomials: [(Monomial::default(), scalar)].into(),
        }
    }
}

impl<F: Zero + Add + Clone + From<BaseField>> Add for Polynomial<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut monomials = self.monomials;
        for (monomial, coef) in rhs.monomials {
            if let Some(existing) = monomials.get_mut(&monomial) {
                let res = existing.clone() + coef;
                if res.is_zero() {
                    monomials.remove(&monomial);
                } else {
                    *existing = res;
                }
            } else {
                monomials.insert(monomial, coef);
            }
        }
        Self { monomials }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Mul for Monomial {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut vars = self.vars;
        for (var, exp) in rhs.vars {
            if let Some(existing) = vars.get_mut(&var) {
                *existing += exp;
            } else {
                vars.insert(var, exp);
            }
        }
        Monomial { vars }
    }
}

impl<F> Mul for Polynomial<F>
where
    F: Clone + Mul<Output = F> + Add<Output = F> + AddAssign + From<BaseField>,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut monomials = BTreeMap::new();
        for (monomial1, coef1) in self.monomials {
            for (monomial2, coef2) in rhs.monomials.clone() {
                let monomial = monomial1.clone() * monomial2.clone();
                let coef = coef1.clone() * coef2.clone();
                if let Some(existing) = monomials.get_mut(&monomial) {
                    *existing += coef;
                } else {
                    monomials.insert(monomial, coef);
                }
            }
        }
        Self { monomials }
    }
}

impl<F> Mul<BaseField> for Polynomial<F>
where
    F: Clone + Mul<Output = F> + Add<Output = F> + AddAssign + From<BaseField>,
{
    type Output = Self;
    fn mul(self, rhs: BaseField) -> Self {
        Self {
            monomials: self
                .monomials
                .into_iter()
                .map(|(m, c)| (m, c * rhs.into()))
                .collect(),
        }
    }
}

impl<F, G> MulAssign<G> for Polynomial<F>
where
    F: Clone + Mul<Output = F> + Add<Output = F> + AddAssign + From<BaseField>,
    G: Into<Polynomial<F>>,
{
    fn mul_assign(&mut self, rhs: G) {
        *self = self.clone() * rhs.into();
    }
}

impl<F> Mul<SecureField> for Polynomial<F>
where
    F: Clone
        + Mul<SecureField, Output = SecureField>
        + Add<Output = F>
        + AddAssign
        + From<BaseField>,
{
    type Output = Polynomial<SecureField>;
    fn mul(self, rhs: SecureField) -> Self::Output {
        Self::Output {
            monomials: self
                .monomials
                .into_iter()
                .map(|(m, c)| (m, c * rhs))
                .collect(),
        }
    }
}

impl<F: Zero + Clone + From<BaseField>> Zero for Polynomial<F> {
    fn is_zero(&self) -> bool {
        self.monomials.is_empty()
    }
    fn zero() -> Self {
        Self {
            monomials: BTreeMap::new(),
        }
    }
}

impl<F: One + From<BaseField> + AddAssign + Clone + Add<Output = F>> One for Polynomial<F> {
    fn one() -> Self {
        Self {
            monomials: [(Monomial::default(), F::one())].into(),
        }
    }
}

impl<F: Neg<Output = F> + From<BaseField>> Neg for Polynomial<F> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            monomials: self
                .monomials
                .into_iter()
                .map(|(m, c)| (m, c.neg()))
                .collect(),
        }
    }
}

impl<F, G> AddAssign<G> for Polynomial<F>
where
    F: Clone + Add<Output = F> + Zero + From<BaseField>,
    G: Into<Polynomial<F>>,
{
    fn add_assign(&mut self, rhs: G) {
        self.monomials = (self.clone() + rhs.into()).monomials;
    }
}

impl<F> FieldExpOps for Polynomial<F>
where
    F: Zero + One + Clone + Add + Mul + AddAssign + From<BaseField>,
{
    fn inverse(&self) -> Self {
        assert!(!self.is_zero(), "0 has no inverse");
        let mut res = Self::from(BaseField::one()) / self.clone();
        res
    }
}

fn to_superscript(n: usize) -> String {
    if n == 0 {
        return std::char::from_u32(0x2070_u32).unwrap().to_string();
    }
    let mut n = n;
    let mut res = String::new();
    while n > 0 {
        let rem = n % 10;
        res.push(match rem {
            1 => std::char::from_u32(0x00b9).unwrap(),
            2 => std::char::from_u32(0x00b2).unwrap(),
            3 => std::char::from_u32(0x00b3).unwrap(),
            _ => std::char::from_u32(0x2070 + rem as u32).unwrap(),
        });
        n /= 10;
    }
    res.chars().rev().collect()
}

fn to_subscript(n: usize) -> String {
    if n == 0 {
        return std::char::from_u32(0x2080_u32).unwrap().to_string();
    }
    let mut n = n;
    let mut res = String::new();
    while n > 0 {
        let rem = n % 10;
        res.push(std::char::from_u32(0x2080 + rem as u32).unwrap());
        n /= 10;
    }
    res.chars().rev().collect()
}

impl Display for Monomial {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (var, exp) in self.vars.iter().sorted() {
            write!(
                f,
                "x{}{}",
                to_subscript(*var),
                if *exp == 1 {
                    "".to_string()
                } else {
                    to_superscript(*exp)
                }
            )?;
        }
        write!(f, "")
    }
}

impl<F> Display for Polynomial<F>
where
    F: Display + Zero + Add + Clone + From<BaseField>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut is_first = true;
        for (monomial, coef) in self.monomials.iter() {
            if !coef.is_zero() {
                if !is_first {
                    write!(f, " + ")?;
                } else {
                    is_first = false;
                }
                write!(f, "{}", coef)?;
                if !monomial.vars.is_empty() {
                    write!(f, "{}", monomial)?;
                }
            }
        }
        write!(f, "")
    }
}

#[cfg(test)]
mod tests {
    use super::{Monomial, Polynomial};
    use crate::core::fields::m31::M31;

    #[test]
    fn test_polynomial_display() {
        let monomial1 = Monomial {
            vars: (0..10).map(|i| (i, i + 1)).collect(),
        };
        let monomial2 = Monomial {
            vars: (0..10).map(|i| (i + 1, i + 1)).collect(),
        };
        let polynomial = Polynomial {
            monomials: [(monomial1.clone(), M31(18)), (monomial2.clone(), M31(3))].into(),
        };
        assert_eq!(
            polynomial.to_string(),
            "18x₀x₁²x₂³x₃⁴x₄⁵x₅⁶x₆⁷x₇⁸x₈⁹x₉¹⁰ + 3x₁x₂²x₃³x₄⁴x₅⁵x₆⁶x₇⁷x₈⁸x₉⁹x₁₀¹⁰"
        )
    }

    #[test]
    fn test_add_polynomials() {
        let monomial1 = Monomial {
            vars: [(0, 1), (1, 2)].into(),
        };
        let monomial2 = Monomial {
            vars: [(2, 1), (3, 2)].into(),
        };
        let monomial3 = Monomial {
            vars: [(4, 1), (5, 2)].into(),
        };
        let monomial4 = Monomial {
            vars: [(6, 1), (7, 2)].into(),
        };
        let poly1 = Polynomial::<M31> {
            monomials: [
                (monomial1.clone(), M31(1)),
                (monomial2.clone(), M31(2)),
                (monomial3.clone(), M31(8)),
            ]
            .into(),
        };
        let poly2 = Polynomial::<M31> {
            monomials: [
                (monomial2.clone(), M31(5)),
                (monomial3.clone(), -M31(8)),
                (monomial4.clone(), M31(6)),
            ]
            .into(),
        };

        assert_eq!(
            (poly1.clone() + poly2.clone()).monomials,
            [
                (monomial1.clone(), M31(1)),
                (monomial2.clone(), M31(7)),
                (monomial4.clone(), M31(6))
            ]
            .into()
        );
    }

    #[test]
    fn test_mul_monomials() {
        let monomial1 = Monomial {
            vars: [(0, 1), (1, 2)].into(),
        };
        let monomial2 = Monomial {
            vars: [(1, 1), (2, 2)].into(),
        };
        assert_eq!(
            monomial1.clone() * monomial2.clone(),
            Monomial {
                vars: [(0, 1), (1, 3), (2, 2)].into()
            }
        );
    }

    #[test]
    fn test_mul_polynomials() {
        let monomial1 = Monomial {
            vars: [(0, 1), (1, 2)].into(),
        };
        let monomial2 = Monomial {
            vars: [(2, 1), (3, 2)].into(),
        };
        let monomial3 = Monomial {
            vars: [(4, 1), (5, 2)].into(),
        };
        let poly1 = Polynomial::<M31> {
            monomials: [(monomial1.clone(), M31(1)), (monomial2.clone(), M31(2))].into(),
        };
        let poly2 = Polynomial::<M31> {
            monomials: [(monomial2.clone(), M31(5)), (monomial3.clone(), -M31(8))].into(),
        };

        assert_eq!(
            (poly1.clone() * poly2.clone()).monomials,
            [
                (monomial1.clone() * monomial2.clone(), M31(5)),
                (monomial1.clone() * monomial3.clone(), -M31(8)),
                (monomial2.clone() * monomial2.clone(), M31(10)),
                (monomial2.clone() * monomial3.clone(), -M31(16))
            ]
            .into()
        );
    }
}
