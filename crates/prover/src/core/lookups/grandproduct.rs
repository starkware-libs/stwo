use std::borrow::{Borrow, Cow};
use std::ops::Deref;

use derivative::Derivative;
use num_traits::One;

use super::gkr::{
    EqEvals, GkrBinaryGate, GkrBinaryLayer, GkrMask, GkrMultivariatePolyOracle, GkrOps, Layer,
    NotConstantPolyError,
};
use super::mle::{Mle, MleOps};
use super::sumcheck::MultivariatePolyOracle;
use super::utils::{eq, UnivariatePoly};
use crate::core::backend::{Column, ColumnOps, CpuBackend};
use crate::core::fields::qm31::SecureField;

pub trait GrandProductOps: MleOps<SecureField> + GkrOps + Sized + 'static {
    /// Generates the next GKR layer from the current one.
    fn next_layer(layer: &GrandProductTrace<Self>) -> GrandProductTrace<Self>;

    /// Returns univariate polynomial `f(t) = sum_x h(t, x)` for all `x` in the boolean hypercube.
    ///
    /// `claim` equals `f(0) + f(1)`.
    ///
    /// For more context see docs of [`MultivariatePolyOracle::sum_as_poly_in_first_variable()`].
    fn sum_as_poly_in_first_variable(
        h: &GrandProductOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField>;
}

#[derive(Derivative)]
#[derivative(Debug(bound = ""), Clone(bound = ""))]
pub struct GrandProductTrace<B: ColumnOps<SecureField>>(Mle<B, SecureField>);

impl<B: ColumnOps<SecureField>> GrandProductTrace<B> {
    pub fn new(column: Mle<B, SecureField>) -> Self {
        Self(column)
    }

    /// Returns a copy of this trace with the [`CpuBackend`].
    ///
    /// This operation is expensive but can be useful for small traces that are difficult to handle
    /// depending on the backend. For example, the SIMD backend offloads to the CPU backend when
    /// trace length becomes smaller than the SIMD lane count.
    pub fn to_cpu(&self) -> GrandProductTrace<CpuBackend> {
        GrandProductTrace::new(Mle::new(self.0.to_cpu()))
    }
}

impl<B: ColumnOps<SecureField>> Deref for GrandProductTrace<B> {
    type Target = Mle<B, SecureField>;

    fn deref(&self) -> &Mle<B, SecureField> {
        &self.0
    }
}

impl<B: GrandProductOps> GkrBinaryLayer for GrandProductTrace<B> {
    type Backend = B;

    type MultivariatePolyOracle<'a> = GrandProductOracle<'a, B>;

    fn n_variables(&self) -> usize {
        self.0.n_variables()
    }

    fn next(&self) -> Layer<Self> {
        assert_ne!(0, self.n_variables());
        let next_layer = B::next_layer(self);

        if next_layer.n_variables() == 0 {
            Layer::Output(next_layer.0.to_cpu())
        } else {
            Layer::Internal(next_layer)
        }
    }

    fn into_multivariate_poly(
        self,
        _: SecureField,
        eq_evals: &EqEvals<B>,
    ) -> GrandProductOracle<'_, B> {
        GrandProductOracle::new(Cow::Borrowed(eq_evals), self)
    }
}

/// Multivariate polynomial oracle.
pub struct GrandProductOracle<'a, B: GrandProductOps> {
    /// `eq_evals` passed by [`GkrBinaryLayer::into_multivariate_poly()`].
    eq_evals: Cow<'a, EqEvals<B>>,
    eq_fixed_var_correction: SecureField,
    trace: GrandProductTrace<B>,
}

impl<'a, B: GrandProductOps> GrandProductOracle<'a, B> {
    pub fn new(eq_evals: Cow<'a, EqEvals<B>>, trace: GrandProductTrace<B>) -> Self {
        Self {
            eq_evals,
            eq_fixed_var_correction: SecureField::one(),
            trace,
        }
    }

    pub fn eq_evals(&self) -> &EqEvals<B> {
        self.eq_evals.borrow()
    }

    pub fn eq_fixed_var_correction(&self) -> SecureField {
        self.eq_fixed_var_correction
    }

    pub fn trace(&self) -> &GrandProductTrace<B> {
        &self.trace
    }

    /// Returns a copy of this oracle with the [`CpuBackend`].
    ///
    /// This operation is expensive but can be useful for small oracles that are difficult to handle
    /// depending on the backend. For example, the SIMD backend offloads to the CPU backend when
    /// trace length becomes smaller than the SIMD lane count.
    pub fn to_cpu(&self) -> GrandProductOracle<'a, CpuBackend> {
        // TODO(andrew): This block is not ideal.
        let n_eq_evals = 1 << (self.n_variables() - 1);
        let eq_evals = Cow::Owned(EqEvals {
            evals: Mle::new((0..n_eq_evals).map(|i| self.eq_evals.at(i)).collect()),
            y: self.eq_evals.y.to_vec(),
        });

        GrandProductOracle {
            eq_evals,
            eq_fixed_var_correction: self.eq_fixed_var_correction,
            trace: self.trace.to_cpu(),
        }
    }
}

impl<'a, B: GrandProductOps> MultivariatePolyOracle for GrandProductOracle<'a, B> {
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

        Self {
            eq_evals: self.eq_evals,
            eq_fixed_var_correction,
            trace: GrandProductTrace::new(self.trace.0.fix_first_variable(challenge)),
        }
    }
}

impl<'a, B: GrandProductOps> GkrMultivariatePolyOracle for GrandProductOracle<'a, B> {
    fn try_into_mask(self) -> Result<GkrMask, NotConstantPolyError> {
        if self.n_variables() != 0 {
            return Err(NotConstantPolyError);
        }

        let evals = self.trace.0.to_cpu().try_into().unwrap();

        Ok(GkrMask::new(vec![evals]))
    }
}

/// A multiplication gate.
pub struct GrandProductGate;

impl GkrBinaryGate for GrandProductGate {
    fn eval(&self, a: &[SecureField], b: &[SecureField]) -> Vec<SecureField> {
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        vec![a[0] * b[0]]
    }
}
