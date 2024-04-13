use std::ops::Deref;

use derivative::Derivative;
use gkr::{BinaryTreeCircuit, EqEvals, GkrLayer, GkrOps, GkrSumcheckOracle};
use mle::{Mle, MleOps, MleTrace};
use num_traits::{One, Zero};
use sumcheck::SumcheckOracle;
use utils::{eq, UnivariatePolynomial};

use crate::core::backend::ColumnOps;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;

pub mod gkr;
pub mod mle;
pub mod sumcheck;
pub mod utils;

// TODO: Consider removing these super traits (especially GkrOps).
pub trait GrandProductOps: MleOps<SecureField> + GkrOps + Sized {
    /// Generates the next GKR layer from the current one.
    fn next_layer(layer: &GrandProductTrace<Self>) -> GrandProductTrace<Self>;

    /// Evaluates the univariate round polynomial used in sumcheck at `0` and `2`.
    // TODO: document
    fn univariate_sum_evals(oracle: &GrandProductOracle<'_, Self>) -> UnivariateEvals;
}

// TODO: Docs and consider naming the variants better.
#[derive(Derivative)]
#[derivative(Debug(bound = ""), Clone(bound = ""))]
pub struct GrandProductTrace<B: ColumnOps<SecureField>>(pub Mle<B, SecureField>);

impl<B: MleOps<SecureField>> GrandProductTrace<B> {
    pub fn new(column: Mle<B, SecureField>) -> Self {
        Self(column)
    }

    pub fn num_variables(&self) -> usize {
        self.0.num_variables()
    }
}

impl<B: ColumnOps<SecureField>> Deref for GrandProductTrace<B> {
    type Target = Mle<B, SecureField>;

    fn deref(&self) -> &Mle<B, SecureField> {
        &self.0
    }
}

impl<B: GrandProductOps + 'static> GkrLayer for GrandProductTrace<B> {
    type Backend = B;
    type SumcheckOracle<'a> = GrandProductOracle<'a, B>;

    fn num_variables(&self) -> usize {
        self.0.num_variables()
    }

    fn next(&self) -> Option<Self> {
        if self.0.num_variables() == 0 {
            return None;
        }

        Some(B::next_layer(self))
    }

    fn into_sumcheck_oracle(
        self,
        _lambda: SecureField,
        num_unused_variables: usize,
        eq_evals: &EqEvals<B>,
    ) -> GrandProductOracle<'_, B> {
        let num_variables = self.num_variables() - 1;

        GrandProductOracle {
            trace: self,
            eq_evals,
            num_variables,
            num_unused_variables,
            r: Vec::new(),
        }
    }

    fn into_trace(self) -> MleTrace<B, SecureField> {
        MleTrace::new(vec![self.0])
    }
}

/// Sumcheck oracle for a grand product + GKR layer.
pub struct GrandProductOracle<'a, B: GrandProductOps> {
    trace: GrandProductTrace<B>,
    eq_evals: &'a EqEvals<B>,
    // Randomness sampled through sum-check protocol used to adjust the round polynomials.
    r: Vec<SecureField>,
    num_variables: usize,
    num_unused_variables: usize,
}

// TODO: Remove all these and change LogupOps to return two evaluations instead of polynomial.
impl<'a, B: GrandProductOps> GrandProductOracle<'a, B> {
    pub fn r(&self) -> &[SecureField] {
        &self.r
    }

    pub fn eq_evals(&self) -> &EqEvals<B> {
        self.eq_evals
    }

    pub fn trace(&self) -> &GrandProductTrace<B> {
        &self.trace
    }
}

impl<'a, B: GrandProductOps> SumcheckOracle for GrandProductOracle<'a, B> {
    fn num_variables(&self) -> usize {
        self.num_variables + self.num_unused_variables
    }

    fn univariate_sum(&self, claim: SecureField) -> UnivariatePolynomial<SecureField> {
        if self.num_unused_variables != 0 {
            return UnivariatePolynomial::new(vec![
                claim / (SecureField::one() + SecureField::one()),
            ]);
        }

        let UnivariateEvals {
            eval_at_0,
            eval_at_2,
        } = B::univariate_sum_evals(self);

        let z = self.eq_evals().y();
        let r = self.r();

        // (1 - z0)
        // (1 - z0)

        // TODO: Document
        let n_skipped = z.len() - (r.len() + self.num_variables);
        let k = r.len();
        // let r = [vec![SecureField::one(); n_skipped], r.to_vec()].concat();

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
        // eval_at_0 *= eq(&[SecureField::zero()], &[z[0]]);
        // eval_at_2 *= eq(&[SecureField::zero()], &[z[0]]);

        // The evaluations on `0` and `2` are invalid. They were obtained by summing over the poly
        // `eq((0^(k-1), x_k, .., x_n), (z_1, .., z_n)) * (..)` but we require the sum to be taken
        // on `eq((r_1, ..., r_{k-1}, x_k, .., x_n), (z_1, .., z_n)) * (..)`. Conveniently
        // `eq((0^(k-1), x_k, .., x_n), (z_1, .., z_n))` and `eq((r_1, ..., r_{k-1}, x_k, .., x_n),
        // (z_1, .., z_n))` differ only by a constant factor `eq((r_1, ..., r_{k-1}), (z_1, ..,
        // z_{k-1})) / eq((0^(k-1)), (z_1, .., z_{k-1}))` for all values of `x`.
        // TODO: explain
        let eq_correction_factor = eq(&r[0..k], &z[n_skipped..n_skipped + k])
            / eq(
                &vec![SecureField::zero(); n_skipped + k],
                &z[0..n_skipped + k],
            );

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
        let correction_factor_at =
            |x| eq(&[x], &[z[n_skipped + k]]) / eq(&[SecureField::zero()], &[z[n_skipped + k]]);

        let x0: SecureField = BaseField::zero().into();
        let x1 = BaseField::one().into();
        let x2 = BaseField::from(2).into();

        let mut y0 = eq_correction_factor * eval_at_0;
        let mut y1 = (claim - y0) / correction_factor_at(x1);
        let mut y2 = eq_correction_factor * eval_at_2;

        // We are interpolating a degree 2 function so need three evaluations.
        let simplified_univariate_sum =
            UnivariatePolynomial::interpolate_lagrange(&[x0, x1, x2], &[y0, y1, y2]);

        let x3 = BaseField::from(3).into();
        let mut y3 = simplified_univariate_sum.eval_at_point(x3);

        // Correct all the evaluations (see comment above).
        y0 *= correction_factor_at(x0); // `y0 *= 1`
        y1 *= correction_factor_at(x1);
        y2 *= correction_factor_at(x2);
        y3 *= correction_factor_at(x3);

        UnivariatePolynomial::interpolate_lagrange(&[x0, x1, x2, x3], &[y0, y1, y2, y3])
    }

    fn fix_first(mut self, challenge: SecureField) -> Self {
        if self.num_variables == 0 {
            return self;
        }

        if self.num_unused_variables != 0 {
            self.num_unused_variables -= 1;
            return self;
        }

        let mut r = self.r;
        r.push(challenge);

        Self {
            trace: GrandProductTrace::new(self.trace.0.fix_first(challenge)),
            eq_evals: self.eq_evals,
            r,
            num_variables: self.num_variables - 1,
            num_unused_variables: 0,
        }
    }
}

impl<'a, B: GrandProductOps + 'static> GkrSumcheckOracle for GrandProductOracle<'a, B> {
    type Backend = B;

    fn into_inputs(self) -> MleTrace<B, SecureField> {
        self.trace.into_trace()
    }
}

/// Circuit for computing the grand product of a single column.
pub struct GrandProductCircuit;

impl BinaryTreeCircuit for GrandProductCircuit {
    fn eval(even_row: &[SecureField], odd_row: &[SecureField]) -> Vec<SecureField> {
        assert_eq!(even_row.len(), 1);
        assert_eq!(odd_row.len(), 1);
        vec![even_row[0] * odd_row[0]]
    }
}

pub struct UnivariateEvals {
    pub eval_at_0: SecureField,
    pub eval_at_2: SecureField,
}
