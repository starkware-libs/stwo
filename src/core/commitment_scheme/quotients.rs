use crate::core::backend::Backend;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;
use crate::core::poly::circle::{CircleDomain, CircleEvaluation};
use crate::core::poly::BitReversedOrder;

pub trait QuotientOps: Backend {
    /// Accumulates the quotients of the columns at the given domain.
    /// For a column f(x), and a point sample (p,v), the quotient is
    ///   (f(x) - V0(x))/V1(x)
    /// where V0(p)=v, V0(conj(p))=conj(v), and V1 is a vanishing polynomial for p,conj(p).
    /// This ensures that if f(p)=v, then the quotient is a polynomial.
    /// The result is a linear combination of the quotients using powers of random_coeff.
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        samples: &[ColumnSampleBatch],
    ) -> SecureColumn<Self>;
}

/// A batch of column samplings at a point.
pub struct ColumnSampleBatch {
    /// The point at which the columns are sampled.
    pub point: CirclePoint<SecureField>,
    /// The sampled column indices and their values at the point.
    pub column_indices_and_values: Vec<(usize, SecureField)>,
}
