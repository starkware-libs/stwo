use mle_eval::MleCoeffColumnOracle;

use crate::core::air::accumulation::PointEvaluationAccumulator;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;
use crate::core::ColumnVec;

pub mod accumulation;
pub mod mle_eval;

// TODO(andrew): Try come up with less verbose name.
pub trait AccumulatedMleCoeffColumnOracle {
    fn accumulate_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        acc: &mut PointEvaluationAccumulator,
    );
}

// TODO(andrew): Try come up with less verbose name.
#[derive(Clone)]
pub struct MleCoeffColumnOracleAccumulator<'a> {
    acc_coeff: SecureField,
    oracles: Vec<&'a dyn AccumulatedMleCoeffColumnOracle>,
}

impl<'a> MleCoeffColumnOracleAccumulator<'a> {
    pub fn new(acc_coeff: SecureField) -> Self {
        Self {
            acc_coeff,
            oracles: Vec::new(),
        }
    }

    pub fn accumulate<'b: 'a>(&mut self, oracle: &'b dyn AccumulatedMleCoeffColumnOracle) {
        self.oracles.push(oracle)
    }
}

impl<'a> MleCoeffColumnOracle for MleCoeffColumnOracleAccumulator<'a> {
    fn evaluate_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
    ) -> SecureField {
        let mut acc = PointEvaluationAccumulator::new(self.acc_coeff);
        for oracle in &self.oracles {
            oracle.accumulate_at_point(point, mask, &mut acc);
        }
        acc.finalize()
    }
}
