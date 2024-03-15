use crate::core::air::accumulation::ColumnAccumulator;
use crate::core::backend::{Backend, Col};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::CircleDomain;

pub trait QuotientOps: Backend {
    fn accumulate_quotients(
        domain: CircleDomain,
        accum: ColumnAccumulator<'_, Self>,
        columns: &[Col<Self, BaseField>],
        random_coeff: SecureField,
        openings: &[BatchedColumnOpenings],
    );
}

pub struct BatchedColumnOpenings {
    pub point: CirclePoint<SecureField>,
    pub column_indices_and_values: Vec<(usize, SecureField)>,
}
