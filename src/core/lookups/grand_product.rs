use std::borrow::Cow;
use std::fmt::Debug;
use std::ops::Deref;

use super::gkr::{BinaryTreeCircuit, GkrLayer, GkrOps, GkrSumcheckOracle};
use super::mle::{ColumnOpsV2, Mle, MleOps, MleTrace};
use super::sumcheck::SumcheckOracle;
use super::utils::Polynomial;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::mle::ColumnV2;

// TODO: Consider removing these super traits (especially GkrOps).
pub trait GrandProductOps: MleOps<SecureField> + GkrOps + Sized {
    /// Generates the next GKR layer from the current one.
    fn next_layer(layer: &GrandProductTrace<Self>) -> GrandProductTrace<Self>;

    /// Evaluates the univariate round polynomial used in sumcheck at `0` and `1`.
    fn univariate_sum(
        oracle: &GrandProductOracle<'_, Self>,
        claim: SecureField,
    ) -> Polynomial<SecureField>;
}

// TODO: Docs and consider naming the variants better.
pub struct GrandProductTrace<B: ColumnOpsV2<SecureField>>(Mle<B, SecureField>);

impl<B: ColumnOpsV2<SecureField>> Clone for GrandProductTrace<B> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<B: ColumnOpsV2<SecureField>> Debug for GrandProductTrace<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("GrandProductTrace").field(&self.0).finish()
    }
}

impl<B: MleOps<SecureField>> GrandProductTrace<B> {
    pub fn new(column: Mle<B, SecureField>) -> Self {
        Self(column)
    }

    pub fn num_variables(&self) -> usize {
        self.0.num_variables()
    }
}

impl<B: ColumnOpsV2<SecureField>> Deref for GrandProductTrace<B> {
    type Target = Mle<B, SecureField>;

    fn deref(&self) -> &Mle<B, SecureField> {
        &self.0
    }
}

impl<B: GrandProductOps> GkrLayer for GrandProductTrace<B> {
    type Backend = B;
    type SumcheckOracle<'a> = GrandProductOracle<'a, B>;

    fn next(&self) -> Option<Self> {
        if self.0.len() == 2 {
            return None;
        }

        Some(B::next_layer(self))
    }

    fn into_sumcheck_oracle<'a>(
        self,
        _lambda: SecureField,
        layer_assignment: &[SecureField],
        eq_evals: &'a B::EqEvals,
    ) -> GrandProductOracle<'a, B> {
        let num_variables = self.num_variables() - 1;

        GrandProductOracle {
            trace: self,
            eq_evals: Cow::Borrowed(eq_evals),
            num_variables,
            z: layer_assignment.to_vec(),
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
    /// Evaluations of `eq_z(x_1, ..., x_n)` (see [`gen_eq_evals`] docs).
    eq_evals: Cow<'a, B::EqEvals>,
    /// The random point sampled during the GKR protocol for the sumcheck.
    // TODO: Better docs.
    z: Vec<SecureField>,
    r: Vec<SecureField>,
    num_variables: usize,
}

// TODO: Remove all these and change LogupOps to return two evaluations instead of polynomial.
impl<'a, B: GrandProductOps> GrandProductOracle<'a, B> {
    pub fn r(&self) -> &[SecureField] {
        &self.r
    }

    pub fn z(&self) -> &[SecureField] {
        &self.z
    }

    pub fn eq_evals(&self) -> &B::EqEvals {
        self.eq_evals.as_ref()
    }

    pub fn trace(&self) -> &GrandProductTrace<B> {
        &self.trace
    }
}

impl<'a, B: GrandProductOps> SumcheckOracle for GrandProductOracle<'a, B> {
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

        let mut r = self.r;
        r.push(challenge);

        Self {
            trace: GrandProductTrace::new(self.trace.0.fix_first(challenge)),
            eq_evals: self.eq_evals,
            z: self.z,
            r,
            num_variables: self.num_variables - 1,
        }
    }
}

impl<'a, B: GrandProductOps> GkrSumcheckOracle for GrandProductOracle<'a, B> {
    type Backend = B;

    fn into_inputs(self) -> MleTrace<B, SecureField> {
        self.trace.into_trace()
    }
}

/// Circuit for computing the grand product of a single column.
pub struct GrandProductCircuit;

impl BinaryTreeCircuit for GrandProductCircuit {
    fn eval(&self, even_row: &[SecureField], odd_row: &[SecureField]) -> Vec<SecureField> {
        assert_eq!(even_row.len(), 1);
        assert_eq!(odd_row.len(), 1);

        let a = even_row[0];
        let b = odd_row[0];
        let c = a * b;

        vec![c]
    }
}
