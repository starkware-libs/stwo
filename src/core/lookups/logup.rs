use std::borrow::Cow;
use std::iter::{repeat, Sum};
use std::ops::Add;

use num_traits::{One, Zero};

use super::gkr::{BinaryTreeCircuit, GkrLayer, GkrOps, GkrSumcheckOracle};
use super::mle::{ColumnV2, Mle, MleOps, MleTrace};
use super::sumcheck::{SumcheckOracle, UnivariateEvals};
use super::utils::{eq, Polynomial};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{ExtensionOf, Field};

pub trait LogupOps: MleOps<BaseField> + MleOps<SecureField> + GkrOps + Sized {
    /// Generates the next GKR layer from the current one.
    fn next_layer(layer: &LogupTrace<Self>) -> LogupTrace<Self>;

    /// Evaluates the univariate round polynomial used in sumcheck at `0` and `1`.
    fn univariate_sum_evals(oracle: &LogupOracle<'_, Self>) -> UnivariateEvals;
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
        let UnivariateEvals {
            mut eval_at_0,
            mut eval_at_2,
        } = B::univariate_sum_evals(self);

        let z = self.z();
        let r = self.r();

        // We wanted to compute a sum of a multivariate polynomial
        // `eq((0^(k-1), x_k, .., x_n), (z_1, .., z_n)) * (..)` over
        // all `(x_k, ..., x_n)` in `{0, 1}^(n-k)`. Instead we computes a sum over
        // `eq((0^(k-2), x_k, .., x_n), (z_2, .., z_n)) * (..)`. The two multivariate sums differs
        // by a constant factor `eq((0), (z_1))` which is added back in here.
        //
        // The reason the factor is left out originally is for performance reasons. In the naive
        // version we want to precompute the evaluations of `eq((x_1, .., x_n), (z_1, .., z_n))`
        // ahead of time for all `(x_1, .., x_n)` in `{0, 1}^n`. Notice we only use half of
        // these evaluations (specifically those where `x_1` is zero). Each the term of the sum gets
        // multiplied by one of these evaluations. Notice all the terms of the sum contain a
        // constant factor `eq((x_1), (z_1))` (since x_1 equals zero). In the optimized
        // version we precompute the evaluations of `eq((x_2, .., x_n), (z_2, .., z_n))` which is
        // half the size (and takes half the work) of the original precomputation. We then add the
        // missing `eq((x_1), (z_1))` factor back here.
        //
        // TODO: Doc is a bit wordy it's not great have to explain all this but the optimization
        // is worthwhile. Consider modifying `gen_eq_evals()` so that it only returns the first
        // half. Would be just as optimized but prevent having to explain things here.
        eval_at_0 *= eq(&[SecureField::zero()], &[z[0]]);
        eval_at_2 *= eq(&[SecureField::zero()], &[z[0]]);

        // The evaluations on `0` and `2` are invalid. They were obtained by summing over the poly
        // `eq((0^(k-1), x_k, .., x_n), (z_1, .., z_n)) * (..)` but we require the sum to be taken
        // on `eq((r_1, ..., r_{k-1}, x_k, .., x_n), (z_1, .., z_n)) * (..)`. Conveniently
        // `eq((0^(k-1), x_k, .., x_n), (z_1, .., z_n))` and `eq((r_1, ..., r_{k-1}, x_k, .., x_n),
        // (z_1, .., z_n))` differ only by a constant factor `eq((r_1, ..., r_{k-1}), (z_1, ..,
        // z_{k-1})) / eq((0^(k-1)), (z_1, .., z_{k-1}))` for all values of `x`.
        // TODO: explain
        let k = r.len();
        let eq_correction_factor = eq(r, &z[0..k]) / eq(&vec![SecureField::zero(); k], &z[0..k]);

        // Our goal is to compute the sum of `eq((x_k, .., x_n), (z_k, .., z_n)) * h(x_k, .., x_n)`
        // over all possible values `(x_{k+1}, .., x_n)` in `{0, 1}^{n-1}`, effectively reducing the
        // sum to a univariate polynomial in `x_k`. Let this univariate polynomial be `f`. Our
        // method to is to evaluate `f` in `deg(f) + 1` points (which can be done efficiently) to
        // obtain the coefficient representation of `f` via interpolation.
        //
        // Although evaluating `f` is efficient, the runtime of the sumcheck prover is proportional
        // to how many points `f` needs to be evaluated on. To reduce the number of evaluations the
        // prover must perform we can reduce the degree of of the polynomial we need to interpolate.
        // This can be done by instead computing the sum over `eq((0, .., x_n), (z_k, .., z_n)) *
        // h(x_k, .., x_n)` denoted `simplified_sum` which has degree `deg(f) - 1`. We interpolate,
        // our lower degree polynomial, `simplified_sum` with one less evaluation and multiply it
        // afterwards by `eq((x_k), (z_k)) / eq((0), (z_k))` to obtain the original `f`. This idea
        // and algorithm is from <https://eprint.iacr.org/2024/108.pdf> (Section 3.2).
        let correction_factor_at = |x| eq(&[x], &[z[k]]) / eq(&[SecureField::zero()], &[z[k]]);

        let x0: SecureField = BaseField::zero().into();
        let x1 = BaseField::one().into();
        let x2 = BaseField::from(2).into();

        let mut y0 = eq_correction_factor * eval_at_0;
        let mut y1 = (claim - y0) / correction_factor_at(x1);
        let mut y2 = eq_correction_factor * eval_at_2;

        // We are interpolating a degree 2 function so need three evaluations.
        let simplified_univariate_sum =
            Polynomial::interpolate_lagrange(&[x0, x1, x2], &[y0, y1, y2]);

        let x3 = BaseField::from(3).into();
        let mut y3 = simplified_univariate_sum.eval(x3);

        // Correct all the evaluations (see comment above).
        y0 *= correction_factor_at(x0); // `y0 *= 1`
        y1 *= correction_factor_at(x1);
        y2 *= correction_factor_at(x2);
        y3 *= correction_factor_at(x3);

        Polynomial::interpolate_lagrange(&[x0, x1, x2, x3], &[y0, y1, y2, y3])
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
