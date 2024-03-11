use super::component::FibonacciComponent;
use crate::core::air::{Air, ComponentVisitor};
use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;

pub struct MultiFibonacciAir {
    pub components: Vec<FibonacciComponent>,
}

impl MultiFibonacciAir {
    pub fn new(log_n_components: u32, log_size: u32, claim: BaseField) -> Self {
        let n_components = 1 << log_n_components;
        let mut components = Vec::with_capacity(n_components);
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
    use crate::commitment_scheme::blake2_hash::{Blake2sHasher, HASH_COUNTER};
    use crate::commitment_scheme::hasher::Hasher;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::IntoSlice;
    use crate::core::prover::{prove, verify};
    use crate::fibonacci::Fibonacci;
    use crate::m31;

    #[test]
    fn test_multi_fibonacci() {
        let params = vec![(5, 4, Fibonacci::new(5, m31!(443693538)))];
        for (log_size, log_n_components, fib) in params {
            let air = MultiFibonacciAir::new(log_n_components, log_size, fib.claim);
            let channel =
                &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[fib.claim])));
            let n_components = 1 << log_n_components;
            let trace = vec![fib.get_trace(); n_components];
            let proof = prove(&air, channel, trace);
            unsafe {
                HASH_COUNTER = 0;
            }
            let channel =
                &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[fib.claim])));
            assert!(verify(proof, &air, channel));
            unsafe {
                println!(
                    "Number of hashes for {} columns of size {}: {}",
                    n_components,
                    1 << log_size,
                    HASH_COUNTER
                );
            }
        }
    }
}
