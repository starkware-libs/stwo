use super::component::FibonacciComponent;
use crate::core::air::{Air, ComponentVisitor};
use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;

pub struct MultiFibonacciAir {
    pub components: Vec<FibonacciComponent>,
}

impl MultiFibonacciAir {
    pub fn new(n_components: usize, log_size: u32, claim: BaseField) -> Self {
        let mut components = Vec::new();
        for _ in 0..n_components {
            components.push(FibonacciComponent::new(log_size, claim));
        }
        Self { components }
    }
}

impl Air<CPUBackend> for MultiFibonacciAir {
    fn visit_components<V: ComponentVisitor<CPUBackend>>(&self, v: &mut V) {
        for component in self.components.iter() {
            v.visit(component);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MultiFibonacciAir;
    use crate::commitment_scheme::blake2_hash::Blake2sHasher;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::IntoSlice;
    use crate::core::prover::{prove, verify};
    use crate::fibonacci::Fibonacci;
    use crate::m31;

    #[test]
    fn test_multi_fibonacci() {
        let (log_size, n_components, fib) = (5, 16, Fibonacci::new(5, m31!(443693538)));
        let air = MultiFibonacciAir::new(n_components, log_size, fib.claim);
        let prover_channel =
            &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[fib.claim])));
        let trace = vec![fib.get_trace(); n_components];
        let proof = prove(&air, prover_channel, trace);
        let verifier_channel =
            &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[fib.claim])));
        assert!(verify(proof, &air, verifier_channel));
    }
}
