use std::collections::BTreeSet;
use std::iter::zip;

use gkr_lookup_component::prover::{GkrLookupProverComponent, GkrProofNotConstructedError};
use gkr_lookup_component::verifier::GkrLookupVerifierComponent;
use num_traits::{One, Zero};
use rand::Rng;
use xor_reference_component::XorReferenceComponent;
use xor_table_component::XorTableComponent;

use crate::core::air::{
    Air, AirProver, AirTraceVerifier, AirTraceWriter, Component, ComponentProver,
};
use crate::core::backend::CpuBackend;
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::fields::m31::BaseField;
use crate::core::lookups::gkr_verifier::GkrBatchProof;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::{BitReversedOrder, NaturalOrder};
use crate::core::{ColumnVec, InteractionElements};

pub mod gkr_lookup_component;
pub mod mle_eval_component;
pub mod xor_reference_component;
pub mod xor_table_component;

const LOG_TRACE_LEN: u32 = u8::BITS + u8::BITS;

const TRACE_LEN: usize = 1 << LOG_TRACE_LEN;

pub struct XorAirVerifier {
    gkr_lookup_verifier: GkrLookupVerifierComponent<'static>,
}

impl XorAirVerifier {
    pub fn new(gkr_proof: GkrBatchProof) -> Self {
        Self {
            gkr_lookup_verifier: GkrLookupVerifierComponent::new(
                gkr_proof,
                vec![&XorReferenceComponent, &XorTableComponent],
            ),
        }
    }
}

impl Air for XorAirVerifier {
    fn components(&self) -> Vec<&dyn Component> {
        vec![&self.gkr_lookup_verifier]
    }
}

impl AirTraceVerifier for XorAirVerifier {
    fn interact(&self, channel: &mut Blake2sChannel) -> InteractionElements {
        let ids = self
            .components()
            .iter()
            .flat_map(|c| c.interaction_element_ids())
            .collect::<BTreeSet<String>>();
        let elements = channel.draw_felts(ids.len());
        let interaction_elements = InteractionElements::new(zip(ids, elements).collect());
        self.gkr_lookup_verifier
            .verify_gkr_and_generate_mle_eval_components(channel, &interaction_elements);
        interaction_elements
    }
}

pub struct XorAirProver {
    gkr_lookup_prover: GkrLookupProverComponent<'static, CpuBackend>,
}

impl XorAirProver {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            gkr_lookup_prover: GkrLookupProverComponent::new(vec![
                &XorReferenceComponent,
                &XorTableComponent,
            ]),
        }
    }

    pub fn try_into_gkr_proof(self) -> Result<GkrBatchProof, GkrProofNotConstructedError> {
        self.gkr_lookup_prover.try_into_gkr_proof()
    }
}

impl Air for XorAirProver {
    fn components(&self) -> Vec<&dyn Component> {
        vec![&self.gkr_lookup_prover]
    }
}

impl AirTraceVerifier for XorAirProver {
    fn interact(&self, channel: &mut crate::core::channel::Blake2sChannel) -> InteractionElements {
        let ids = self
            .components()
            .iter()
            .flat_map(|c| c.interaction_element_ids())
            .collect::<BTreeSet<String>>();
        let elements = channel.draw_felts(ids.len());
        InteractionElements::new(zip(ids, elements).collect())
    }
}

impl AirTraceWriter<CpuBackend> for XorAirProver {
    fn write_interaction_trace(
        &self,
        channel: &mut Blake2sChannel,
        trace: &ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
        elements: &InteractionElements,
    ) -> Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        let trace = trace.iter().collect();
        self.gkr_lookup_prover
            .write_interaction_trace(channel, &trace, elements)
    }

    fn to_air_prover(&self) -> &impl AirProver<CpuBackend> {
        self
    }
}

impl AirProver<CpuBackend> for XorAirProver {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<CpuBackend>> {
        vec![&self.gkr_lookup_prover]
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
    use itertools::Itertools;
    use num_traits::Zero;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    // use super::air::XorAir;
    use super::BaseTrace;
    use crate::core::backend::CpuBackend;
    use crate::core::circle::CirclePoint;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::poly::circle::{CanonicCoset, CirclePoly};
    use crate::core::prover::{prove, verify};
    use crate::core::test_utils::test_channel;
    use crate::examples::xor::{XorAirProver, XorAirVerifier};

    #[test]
    fn xor_lookup_example() {
        let mut rng = SmallRng::seed_from_u64(0);
        let base_trace = BaseTrace::gen_random(&mut rng);

        let xor_air_prover = XorAirProver::new();

        let stark_proof = prove(
            &xor_air_prover,
            &mut test_channel(),
            base_trace.into_column_vec(),
        )
        .unwrap();

        let gkr_proof = xor_air_prover.try_into_gkr_proof().unwrap();

        let xor_air_verifier = XorAirVerifier::new(gkr_proof);

        let verification_result = verify(stark_proof, &xor_air_verifier, &mut test_channel());

        println!("verified: {:?}", verification_result);
        todo!()
    }

    #[test]
    fn circle_sum() {
        let mut rng = SmallRng::seed_from_u64(0);
        let coeffs = (0..1 << 8).map(|_| rng.gen()).collect_vec();
        let poly = CirclePoly::<CpuBackend>::new(coeffs);
        let sum_domain = CanonicCoset::new(8).coset();

        let mut sum_on_domain = BaseField::zero();
        for p in sum_domain {
            sum_on_domain += poly.eval_at_point(p.into_ef()).0 .0;
        }

        println!("{}", sum_on_domain);
        let eval_at_0_0 = poly.eval_at_point(CirclePoint {
            x: SecureField::zero(),
            y: SecureField::zero(),
        });
        let doubler = BaseField::from(1 << 8);
        println!("{}", eval_at_0_0);
        println!("{}", eval_at_0_0 / doubler);
        println!("{}", eval_at_0_0 * doubler);
        println!("{}", poly.coeffs[0] / doubler);
        println!("{}", poly.coeffs[0] * doubler);
    }
}
