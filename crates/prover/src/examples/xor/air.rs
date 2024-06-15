use crate::core::air::{Air, AirProver, Component, ComponentProver};
use crate::core::backend::{Backend, CpuBackend};
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::{ColumnVec, ComponentVec, InteractionElements};
use crate::examples::xor::unordered_xor_component::UnorderedXorComponent;
use crate::examples::xor::xor_table_component::XorTableComponent;

trait LookupAir: Air {
    fn lookup_mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> ColumnVec<Vec<CirclePoint<SecureField>>>;
}

trait LookupAirVerifier: LookupAir {
    fn verify_lookup(
        lookup_mask: ColumnVec<Vec<CirclePoint<SecureField>>>,
        interaction_elements: &InteractionElements
    )
}

trait LookupAirProver<B: Backend>: LookupAir {
    fn 
}

pub struct XorAir;

impl Air for XorAir {
    fn components(&self) -> Vec<&dyn Component> {
        vec![&UnorderedXorComponent, &XorTableComponent]
    }
}

impl AirProver<CpuBackend> for XorAir {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<CpuBackend>> {
        todo!()
    }
}
