use std::collections::BTreeSet;
use std::iter::zip;

use num_traits::{One, Zero};
use rand::Rng;
use xor_reference_component::XorReferenceComponent;
use xor_table_component::XorTableComponent;

use crate::core::air::{
    Air, AirProver, AirTraceVerifier, AirTraceWriter, Component, ComponentProver,
};
use crate::core::backend::CpuBackend;
use crate::core::channel::Channel;
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::{BitReversedOrder, NaturalOrder};
use crate::core::{ColumnVec, InteractionElements};

// pub mod air;
pub mod multilinear_eval_at_point;
pub mod xor_reference_component;
pub mod xor_table_component;

const LOG_TRACE_LEN: u32 = u8::BITS + u8::BITS;

const TRACE_LEN: usize = 1 << LOG_TRACE_LEN;

pub struct XorAir;

impl Air for XorAir {
    fn components(&self) -> Vec<&dyn Component> {
        vec![&XorReferenceComponent, &XorTableComponent]
    }
}

impl AirTraceVerifier for XorAir {
    fn interaction_elements(
        &self,
        channel: &mut crate::core::channel::Blake2sChannel,
    ) -> InteractionElements {
        let ids = self
            .components()
            .iter()
            .flat_map(|c| c.interaction_element_ids())
            .collect::<BTreeSet<String>>();
        let elements = channel.draw_felts(ids.len());
        InteractionElements::new(zip(ids, elements).collect())
    }
}

impl AirTraceWriter<CpuBackend> for XorAir {
    fn interact(
        &self,
        _trace: &ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
        _elements: &InteractionElements,
    ) -> Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        todo!()
    }

    fn to_air_prover(&self) -> &impl AirProver<CpuBackend> {
        self
    }
}

impl AirProver<CpuBackend> for XorAir {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<CpuBackend>> {
        vec![&XorReferenceComponent, &XorTableComponent]
    }
}

/// Rectangular trace.
pub struct BaseTrace {
    /// 8-bit XOR LHS operands
    xor_lhs_column: Vec<BaseField>,
    /// 8-bit XOR RHS operands
    xor_rhs_column: Vec<BaseField>,
    /// 8-bit XOR results
    xor_res_column: Vec<BaseField>,
    /// Multiplicity of each 8-bit XOR.
    ///
    /// Index `i` stores the multiplicity of `(i & 0xFF) ^ ((i >> 8) & 0xFF)`.
    xor_multiplicities: Vec<BaseField>,
}

impl BaseTrace {
    /// Generates a random trace.
    pub fn gen_random<R: Rng>(rng: &mut R) -> Self {
        let mut xor_lhs_column = Vec::new();
        let mut xor_rhs_column = Vec::new();
        let mut xor_res_column = Vec::new();
        let mut xor_multiplicities = vec![BaseField::zero(); 256 * 256];

        // Fill trace with random XOR instances.
        for _ in 0..TRACE_LEN {
            let lhs = rng.gen::<u8>() as usize;
            let rhs = rng.gen::<u8>() as usize;
            let res = lhs ^ rhs;

            xor_lhs_column.push(BaseField::from(lhs));
            xor_rhs_column.push(BaseField::from(rhs));
            xor_res_column.push(BaseField::from(res));

            xor_multiplicities[lhs + (rhs << 8)] += BaseField::one();
        }

        Self {
            xor_lhs_column,
            xor_rhs_column,
            xor_res_column,
            xor_multiplicities,
        }
    }

    pub fn into_column_vec(
        self,
    ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        let trace_domain = CanonicCoset::new(LOG_TRACE_LEN).circle_domain();

        let xor_lhs_column = CircleEvaluation::<CpuBackend, BaseField, NaturalOrder>::new(
            trace_domain,
            self.xor_lhs_column,
        );
        let xor_rhs_column = CircleEvaluation::<CpuBackend, BaseField, NaturalOrder>::new(
            trace_domain,
            self.xor_rhs_column,
        );
        let xor_res_column = CircleEvaluation::<CpuBackend, BaseField, NaturalOrder>::new(
            trace_domain,
            self.xor_res_column,
        );
        let xor_multiplicities = CircleEvaluation::<CpuBackend, BaseField, NaturalOrder>::new(
            trace_domain,
            self.xor_multiplicities,
        );

        vec![
            xor_lhs_column.bit_reverse(),
            xor_rhs_column.bit_reverse(),
            xor_res_column.bit_reverse(),
            xor_multiplicities.bit_reverse(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    // use super::air::XorAir;
    use super::BaseTrace;
    use crate::core::prover::prove;
    use crate::core::test_utils::test_channel;
    use crate::examples::xor::XorAir;

    #[test]
    fn xor_lookup_example() {
        let mut rng = SmallRng::seed_from_u64(0);
        let base_trace = BaseTrace::gen_random(&mut rng);

        let _proof = prove(&XorAir, &mut test_channel(), base_trace.into_column_vec());

        todo!()
    }
}
