use std::borrow::{Borrow, Cow};
use std::iter::Sum;
use std::ops::Add;

use derivative::Derivative;
use num_traits::{One, Zero};

use super::gkr::{
    EqEvals, GkrBinaryGate, GkrBinaryLayer, GkrMask, GkrMultivariatePolyOracle, GkrOps, Layer,
    NotConstantPolyError,
};
use super::mle::{Mle, MleOps};
use super::sumcheck::MultivariatePolyOracle;
use super::utils::{eq, UnivariatePoly};
use crate::core::backend::{Column, CpuBackend};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{ExtensionOf, Field};

pub trait LogupOps: MleOps<BaseField> + MleOps<SecureField> + GkrOps + Sized + 'static {
    /// Generates the next GKR layer from the current one.
    fn next_layer(layer: &LogupTrace<Self>) -> LogupTrace<Self>;

    /// Returns univariate polynomial `f(t) = sum_x h(t, x)` for all `x` in the boolean hypercube.
    ///
    /// `claim` equals `f(0) + f(1)`.
    ///
    /// For more context see docs of [`MultivariatePolyOracle::sum_as_poly_in_first_variable()`].
    fn sum_as_poly_in_first_variable(
        h: &LogupOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField>;
}

#[derive(Derivative)]
#[derivative(Debug(bound = ""), Clone(bound = ""))]
pub enum LogupTrace<B: LogupOps> {
    /// All numerators implicitly equal "1".
    Singles { denominators: Mle<B, SecureField> },
    Multiplicities {
        numerators: Mle<B, BaseField>,
        denominators: Mle<B, SecureField>,
    },
    Generic {
        numerators: Mle<B, SecureField>,
        denominators: Mle<B, SecureField>,
    },
}

impl<B: LogupOps> LogupTrace<B> {
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        1 << self.n_variables()
    }

    /// Returns a copy of this trace with the [`CpuBackend`].
    ///
    /// This operation is expensive but can be useful for small traces that are difficult to handle
    /// depending on the backend. For example, the SIMD backend offloads to the CPU backend when
    /// trace length becomes smaller than the SIMD lane count.
    pub fn to_cpu(&self) -> LogupTrace<CpuBackend> {
        match self {
            LogupTrace::Singles { denominators } => LogupTrace::Singles {
                denominators: Mle::new(denominators.to_cpu()),
            },
            LogupTrace::Multiplicities {
                numerators,
                denominators,
            } => LogupTrace::Multiplicities {
                numerators: Mle::new(numerators.to_cpu()),
                denominators: Mle::new(denominators.to_cpu()),
            },
            LogupTrace::Generic {
                numerators,
                denominators,
            } => LogupTrace::Generic {
                numerators: Mle::new(numerators.to_cpu()),
                denominators: Mle::new(denominators.to_cpu()),
            },
        }
    }
}

impl<B: LogupOps> GkrBinaryLayer for LogupTrace<B> {
    type Backend = B;

    type MultivariatePolyOracle<'a> = LogupOracle<'a, B>;

    fn n_variables(&self) -> usize {
        match self {
            Self::Singles { denominators }
            | Self::Multiplicities { denominators, .. }
            | Self::Generic { denominators, .. } => denominators.n_variables(),
        }
    }

    fn next(&self) -> Layer<Self> {
        assert_ne!(0, self.n_variables());
        let next_layer = B::next_layer(self);

        if next_layer.n_variables() == 0 {
            Layer::Output(match next_layer {
                Self::Singles { denominators } => {
                    let numerator = SecureField::one();
                    let denominator = denominators.at(0);
                    vec![numerator, denominator]
                }
                Self::Multiplicities {
                    numerators,
                    denominators,
                } => {
                    let numerator = numerators.at(0).into();
                    let denominator = denominators.at(0);
                    vec![numerator, denominator]
                }
                Self::Generic {
                    numerators,
                    denominators,
                } => {
                    let numerator = numerators.at(0);
                    let denominator = denominators.at(0);
                    vec![numerator, denominator]
                }
            })
        } else {
            Layer::Internal(next_layer)
        }
    }

    fn into_multivariate_poly(
        self,
        lambda: SecureField,
        eq_evals: &EqEvals<Self::Backend>,
    ) -> Self::MultivariatePolyOracle<'_> {
        LogupOracle::new(Cow::Borrowed(eq_evals), lambda, self)
    }
}

/// Sumcheck oracle for a logup+GKR layer.
pub struct LogupOracle<'a, B: LogupOps> {
    /// `eq_evals` passed by [`GkrBinaryLayer::into_multivariate_poly()`].
    eq_evals: Cow<'a, EqEvals<B>>,
    eq_fixed_var_correction: SecureField,
    /// Used to perform a random linear combination of the multivariate polynomial for the
    /// numerators and denominators.
    lambda: SecureField,
    trace: LogupTrace<B>,
}

impl<'a, B: LogupOps> LogupOracle<'a, B> {
    pub fn new(eq_evals: Cow<'a, EqEvals<B>>, lambda: SecureField, trace: LogupTrace<B>) -> Self {
        Self {
            eq_evals,
            eq_fixed_var_correction: SecureField::one(),
            lambda,
            trace,
        }
    }

    pub fn lambda(&self) -> SecureField {
        self.lambda
    }

    pub fn eq_evals(&self) -> &EqEvals<B> {
        self.eq_evals.borrow()
    }

    pub fn eq_fixed_var_correction(&self) -> SecureField {
        self.eq_fixed_var_correction
    }

    pub fn trace(&self) -> &LogupTrace<B> {
        &self.trace
    }

    /// Returns a copy of this oracle with the [`CpuBackend`].
    ///
    /// This operation is expensive but can be useful for small oracles that are difficult to handle
    /// depending on the backend. For example, the SIMD backend offloads to the CPU backend when
    /// trace length becomes smaller than the SIMD lane count.
    pub fn to_cpu(&self) -> LogupOracle<'a, CpuBackend> {
        // TODO(andrew): This block is not ideal.
        let n_eq_evals = 1 << (self.n_variables() - 1);
        let eq_evals = Cow::Owned(EqEvals {
            evals: Mle::new((0..n_eq_evals).map(|i| self.eq_evals.at(i)).collect()),
            y: self.eq_evals.y.to_vec(),
        });

        LogupOracle {
            eq_evals,
            eq_fixed_var_correction: self.eq_fixed_var_correction,
            lambda: self.lambda,
            trace: self.trace.to_cpu(),
        }
    }
}

impl<'a, B: LogupOps> MultivariatePolyOracle for LogupOracle<'a, B> {
    fn n_variables(&self) -> usize {
        self.trace.n_variables() - 1
    }

    fn sum_as_poly_in_first_variable(&self, claim: SecureField) -> UnivariatePoly<SecureField> {
        B::sum_as_poly_in_first_variable(self, claim)
    }

    fn fix_first_variable(self, challenge: SecureField) -> Self {
        if self.n_variables() == 0 {
            return self;
        }

        let z0 = self.eq_evals.y()[self.eq_evals.y().len() - self.n_variables()];
        let eq_fixed_var_correction = self.eq_fixed_var_correction * eq(&[challenge], &[z0]);

        let trace = match self.trace {
            LogupTrace::Generic {
                numerators,
                denominators,
            } => LogupTrace::Generic {
                numerators: numerators.fix_first_variable(challenge),
                denominators: denominators.fix_first_variable(challenge),
            },
            LogupTrace::Multiplicities {
                numerators,
                denominators,
            } => LogupTrace::Generic {
                numerators: numerators.fix_first_variable(challenge),
                denominators: denominators.fix_first_variable(challenge),
            },
            LogupTrace::Singles { denominators } => LogupTrace::Singles {
                denominators: denominators.fix_first_variable(challenge),
            },
        };

        Self {
            eq_evals: self.eq_evals,
            eq_fixed_var_correction,
            lambda: self.lambda,
            trace,
        }
    }
}

impl<'a, B: LogupOps> GkrMultivariatePolyOracle for LogupOracle<'a, B> {
    fn try_into_mask(self) -> Result<GkrMask, NotConstantPolyError> {
        if self.n_variables() != 0 {
            return Err(NotConstantPolyError);
        }

        let columns = match self.trace {
            LogupTrace::Generic {
                numerators,
                denominators,
            } => {
                let numerators = numerators.to_cpu().try_into().unwrap();
                let denominators = denominators.to_cpu().try_into().unwrap();
                vec![numerators, denominators]
            }
            LogupTrace::Singles { denominators } => {
                let numerators = [SecureField::one(); 2];
                let denominators = denominators.to_cpu().try_into().unwrap();
                vec![numerators, denominators]
            }
            // Should only ever get called on `Singles` or `Generic`.
            LogupTrace::Multiplicities { .. } => unimplemented!(),
        };

        Ok(GkrMask::new(columns))
    }
}

/// Logup gate from <https://eprint.iacr.org/2023/1284.pdf> (section 3.1)
pub struct LogupGate;

impl GkrBinaryGate for LogupGate {
    fn eval(&self, even_row: &[SecureField], odd_row: &[SecureField]) -> Vec<SecureField> {
        assert_eq!(even_row.len(), 2);
        assert_eq!(odd_row.len(), 2);

        let a = Fraction::new(even_row[0], even_row[1]);
        let b = Fraction::new(odd_row[0], odd_row[1]);
        let c = a + b;

        vec![c.numerator, c.denominator]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Fraction<F> {
    pub numerator: F,
    pub denominator: SecureField,
}

impl<F> Fraction<F> {
    pub fn new(numerator: F, denominator: SecureField) -> Self {
        Self {
            numerator,
            denominator,
        }
    }
}

impl<F> Add for Fraction<F>
where
    F: Field,
    SecureField: ExtensionOf<F> + Field,
{
    type Output = Fraction<SecureField>;

    fn add(self, rhs: Self) -> Fraction<SecureField> {
        if self.numerator.is_one() && rhs.numerator.is_one() {
            Fraction {
                numerator: self.denominator + rhs.denominator,
                denominator: self.denominator * rhs.denominator,
            }
        } else {
            Fraction {
                numerator: rhs.denominator * self.numerator + self.denominator * rhs.numerator,
                denominator: self.denominator * rhs.denominator,
            }
        }
    }
}

impl Zero for Fraction<SecureField> {
    fn zero() -> Self {
        Self {
            numerator: SecureField::zero(),
            denominator: SecureField::one(),
        }
    }

    fn is_zero(&self) -> bool {
        self.numerator.is_zero() && !self.denominator.is_zero()
    }
}

impl Sum for Fraction<SecureField> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let first = iter.next().unwrap_or_else(Self::zero);
        iter.fold(first, |a, b| a + b)
    }
}
