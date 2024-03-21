use std::borrow::Cow;
use std::iter::{repeat, Sum};
use std::ops::Add;

use num_traits::{One, Zero};

use super::gkr::{BinaryTreeCircuit, GkrLayer, GkrOps, GkrSumcheckOracle};
use super::mle::{ColumnV2, Mle, MleOps, MleTrace};
use super::sumcheck::SumcheckOracle;
use super::utils::Polynomial;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{ExtensionOf, Field};

pub trait LogupOps: MleOps<BaseField> + MleOps<SecureField> + GkrOps + Sized {
    /// Generates the next GKR layer from the current one.
    fn next_layer(layer: &LogupTrace<Self>) -> LogupTrace<Self>;

    /// Evaluates the univariate round polynomial used in sumcheck at `0` and `1`.
    fn univariate_sum(
        oracle: &LogupOracle<'_, Self>,
        claim: SecureField,
    ) -> Polynomial<SecureField>;
}

#[derive(Debug, Clone)]
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
        1 << self.num_variables()
    }

    pub fn num_variables(&self) -> usize {
        match self {
            Self::Singles { denominators }
            | Self::Multiplicities { denominators, .. }
            | Self::Generic { denominators, .. } => denominators.num_variables(),
        }
    }
}

impl<B: LogupOps> GkrLayer for LogupTrace<B> {
    type Backend = B;
    type SumcheckOracle<'a> = LogupOracle<'a, B>;

    fn next(&self) -> Option<Self> {
        if self.len() == 1 {
            return None;
        }

        Some(B::next_layer(self))
    }

    fn into_sumcheck_oracle<'a>(
        self,
        lambda: SecureField,
        layer_assignment: &[SecureField],
        eq_evals: &'a B::EqEvals,
    ) -> LogupOracle<'a, B> {
        let num_variables = self.num_variables() - 1;

        LogupOracle {
            trace: self,
            eq_evals: Cow::Borrowed(eq_evals),
            num_variables,
            z: layer_assignment.to_vec(),
            r: Vec::new(),
            lambda,
        }
    }

    fn into_trace(self) -> MleTrace<B, SecureField> {
        let columns = match self {
            Self::Generic {
                numerators,
                denominators,
            } => vec![numerators, denominators],
            Self::Singles { denominators } => {
                let ones = repeat(SecureField::one());
                let numerators = Mle::new(ones.take(denominators.len()).collect());
                vec![numerators, denominators]
            }
            // `into_trace` should only ever be called on `Singles` or `Generic`.
            Self::Multiplicities { .. } => unimplemented!(),
        };

        MleTrace::new(columns)
    }
}

/// Sumcheck oracle for a logup+GKR layer.
pub struct LogupOracle<'a, B: LogupOps> {
    /// Multi-linear extension of the numerators and denominators
    trace: LogupTrace<B>,
    /// Evaluations of `eq_z(x_1, ..., x_n)` (see [`gen_eq_evals`] docs).
    eq_evals: Cow<'a, B::EqEvals>,
    /// The random point sampled during the GKR protocol for the sumcheck.
    // TODO: Better docs.
    z: Vec<SecureField>,
    r: Vec<SecureField>,
    /// Random value used to combine two sum-checks (one for numerators sumcheck and one for
    /// denominators sumcheck), into one.
    lambda: SecureField,
    num_variables: usize,
}

// TODO: Remove all these and change LogupOps to return two evaluations instead of polynomial.
impl<'a, B: LogupOps> LogupOracle<'a, B> {
    pub fn lambda(&self) -> SecureField {
        self.lambda
    }

    pub fn r(&self) -> &[SecureField] {
        &self.r
    }

    pub fn z(&self) -> &[SecureField] {
        &self.z
    }

    pub fn eq_evals(&self) -> &B::EqEvals {
        self.eq_evals.as_ref()
    }

    pub fn trace(&self) -> &LogupTrace<B> {
        &self.trace
    }
}

impl<'a, B: LogupOps> SumcheckOracle for LogupOracle<'a, B> {
    fn num_variables(&self) -> usize {
        self.num_variables
    }

    fn univariate_sum(&self, claim: SecureField) -> Polynomial<SecureField> {
        B::univariate_sum(self, claim)
    }

    fn fix_first(self, challenge: SecureField) -> Self {
        if self.num_variables == 0 {
            return self;
        }

        let trace = match self.trace {
            LogupTrace::Generic {
                numerators,
                denominators,
            } => LogupTrace::Generic {
                numerators: numerators.fix_first(challenge),
                denominators: denominators.fix_first(challenge),
            },
            LogupTrace::Multiplicities {
                numerators,
                denominators,
            } => LogupTrace::Generic {
                numerators: numerators.fix_first(challenge),
                denominators: denominators.fix_first(challenge),
            },
            LogupTrace::Singles { denominators } => LogupTrace::Singles {
                denominators: denominators.fix_first(challenge),
            },
        };

        let mut r = self.r;
        r.push(challenge);

        Self {
            trace,
            eq_evals: self.eq_evals,
            z: self.z,
            lambda: self.lambda,
            r,
            num_variables: self.num_variables - 1,
        }
    }
}

impl<'a, B: LogupOps> GkrSumcheckOracle for LogupOracle<'a, B> {
    type Backend = B;

    fn into_inputs(self) -> MleTrace<B, SecureField> {
        self.trace.into_trace()
    }
}

/// Logup circuit from <https://eprint.iacr.org/2023/1284.pdf> (section 3.1)
pub struct LogupCircuit;

impl BinaryTreeCircuit for LogupCircuit {
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
