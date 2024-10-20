use num_traits::One;

use crate::constraint_framework::constant_columns::gen_is_first;
use crate::constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, TraceLocationAllocator,
};
use crate::core::backend::simd::SimdBackend;
use crate::core::channel::Poseidon252Channel;
use crate::core::fields::m31::BaseField;
use crate::core::pcs::{CommitmentSchemeProver, PcsConfig};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use crate::core::poly::NaturalOrder;
use crate::core::prover::{prove, StarkProof};
use crate::core::utils::coset_order_to_circle_domain_order;
use crate::core::vcs::poseidon252_merkle::{Poseidon252MerkleChannel, Poseidon252MerkleHasher};

// pub struct FibClaim {
//     log_n: u32,
// }

pub type FibComponent<const N_COLUMNS: usize> = FrameworkComponent<FibEval<N_COLUMNS>>;

pub struct FibEval<const N_COLUMNS: usize> {
    log_size: u32,
}

impl<const N_COLUMNS: usize> FrameworkEval for FibEval<N_COLUMNS> {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let [is_first, is_second] = eval.next_interaction_mask(1, [0, -1]);
        let is_first_or_second = is_first + is_second;
        let is_not_first_or_second = E::F::from(BaseField::one()) - is_first_or_second.clone();

        for _ in 0..N_COLUMNS {
            let [curr, prev, prev_prev] = eval.next_interaction_mask(0, [0, -1, -2]);
            eval.add_constraint((curr.clone() - prev - prev_prev) * is_not_first_or_second.clone());
            eval.add_constraint((curr - E::F::from(BaseField::one())) * is_first_or_second.clone());
        }

        eval
    }
}

pub fn prove_fib<const N_COLUMNS: usize>(
    log_size: u32,
    config: PcsConfig,
) -> StarkProof<Poseidon252MerkleHasher> {
    assert!(log_size > 0);
    let mut fib_values = vec![BaseField::one(), BaseField::one()];
    for i in 2..1 << log_size {
        fib_values.push(fib_values[i - 1] + fib_values[i - 2]);
    }

    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_size + config.fri_config.log_blowup_factor + 1)
            .circle_domain()
            .half_coset,
    );

    let mut channel = Poseidon252Channel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<_, Poseidon252MerkleChannel>::new(config, &twiddles);

    // Fib values trace.
    let domain = CanonicCoset::new(log_size).circle_domain();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(vec![
            CircleEvaluation::<SimdBackend, BaseField, NaturalOrder>::new(
                domain,
                coset_order_to_circle_domain_order(&fib_values)
                    .into_iter()
                    .collect(),
            )
            .bit_reverse();
            N_COLUMNS
        ]);
    tree_builder.commit(&mut channel);

    // Constants trace ("is_first").
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(vec![gen_is_first(log_size)]);
    tree_builder.commit(&mut channel);

    // Prove constraints.
    let component = FibComponent::new(
        &mut TraceLocationAllocator::default(),
        FibEval::<N_COLUMNS> { log_size },
    );
    prove(&[&component], &mut channel, &mut commitment_scheme).unwrap()
}

pub fn verify_fib() {}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use itertools::Itertools;
    use starknet_ff::FieldElement;

    use super::prove_fib;
    use crate::constraint_framework::TraceLocationAllocator;
    use crate::core::air::Component;
    use crate::core::channel::Poseidon252Channel;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fri::{FriConfig, FriLayerProof, FriProof};
    use crate::core::pcs::{CommitmentSchemeProof, CommitmentSchemeVerifier, PcsConfig};
    use crate::core::poly::line::LinePoly;
    use crate::core::prover::{verify, StarkProof};
    use crate::core::vcs::poseidon252_merkle::{Poseidon252MerkleChannel, Poseidon252MerkleHasher};
    use crate::core::vcs::prover::MerkleDecommitment;
    use crate::examples::simple_fib::{FibComponent, FibEval};

    #[test]
    fn test_fib_prove() {
        const N_COLUMNS: usize = 2;

        let log_size = 10;

        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(5, 4, 64),
        };

        // Prove.
        let proof = prove_fib::<N_COLUMNS>(log_size, config);

        // Verify.
        let component = FibComponent::<N_COLUMNS>::new(
            &mut TraceLocationAllocator::default(),
            FibEval { log_size },
        );
        let mut channel = Poseidon252Channel::default();
        let mut commitment_scheme =
            CommitmentSchemeVerifier::<Poseidon252MerkleChannel>::new(config);

        // Decommit.
        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = component.trace_log_degree_bounds();
        // Trace columns.
        commitment_scheme.commit(proof.commitments[0], &sizes[0], &mut channel);
        // Constant columns.
        commitment_scheme.commit(proof.commitments[1], &sizes[1], &mut channel);

        serialize_proof_cairo1(&proof);
        // println!("{}", );

        verify(&[&component], &mut channel, &mut commitment_scheme, proof).unwrap();
    }

    fn serialize_proof_cairo1(proof: &StarkProof<Poseidon252MerkleHasher>) {
        trait CairoSerialize {
            fn serialize(&self) -> String;
        }

        impl CairoSerialize for BaseField {
            fn serialize(&self) -> String {
                format!("m31({self})")
            }
        }

        impl CairoSerialize for SecureField {
            fn serialize(&self) -> String {
                let [c0, c1, c2, c3] = self.to_m31_array().map(|v| v.0);
                format!("qm31({c0}, {c1}, {c2}, {c3})")
            }
        }

        impl CairoSerialize for MerkleDecommitment<Poseidon252MerkleHasher> {
            fn serialize(&self) -> String {
                let Self {
                    hash_witness,
                    column_witness,
                } = self;
                format!(
                    "MerkleDecommitment {{ hash_witness: {}, column_witness: {} }}",
                    hash_witness.serialize(),
                    column_witness.serialize()
                )
            }
        }

        impl CairoSerialize for LinePoly {
            fn serialize(&self) -> String {
                let Self { coeffs, log_size } = self;
                format!(
                    "LinePoly {{ coeffs: {}, log_size: {log_size} }}",
                    coeffs.serialize()
                )
            }
        }

        impl CairoSerialize for FriLayerProof<Poseidon252MerkleHasher> {
            fn serialize(&self) -> String {
                let Self {
                    evals_subset,
                    decommitment,
                    commitment,
                } = self;
                format!(
                    "FriLayerProof {{ evals_subset: {}, decommitment: {}, commitment: {} }}",
                    evals_subset.serialize(),
                    decommitment.serialize(),
                    commitment.serialize()
                )
            }
        }

        impl CairoSerialize for FriProof<Poseidon252MerkleHasher> {
            fn serialize(&self) -> String {
                let Self {
                    inner_layers,
                    last_layer_poly,
                } = self;
                format!(
                    "FriProof {{ inner_layers: {}, last_layer_poly: {} }}",
                    inner_layers.serialize(),
                    last_layer_poly.serialize()
                )
            }
        }

        impl CairoSerialize for FieldElement {
            fn serialize(&self) -> String {
                format!("{self}_felt252")
            }
        }

        impl CairoSerialize for CommitmentSchemeProof<Poseidon252MerkleHasher> {
            fn serialize(&self) -> String {
                let Self {
                    sampled_values,
                    decommitments,
                    queried_values,
                    proof_of_work,
                    fri_proof,
                } = self;
                format!("CommitmentSchemeProof {{ sampled_values: {}, decommitments: {}, queried_values: {}, proof_of_work: {}, fri_proof: {} }}", sampled_values.serialize(), decommitments.serialize(), queried_values.serialize(), proof_of_work, fri_proof.serialize())
            }
        }

        impl CairoSerialize for StarkProof<Poseidon252MerkleHasher> {
            fn serialize(&self) -> String {
                let Self {
                    commitments,
                    commitment_scheme_proof,
                } = self;
                format!(
                    "StarkProof {{ commitments: {}, commitment_scheme_proof: {} }}",
                    commitments.serialize(),
                    commitment_scheme_proof.serialize()
                )
            }
        }

        impl<T: CairoSerialize> CairoSerialize for Vec<T> {
            fn serialize(&self) -> String {
                format!(
                    "array![{}]",
                    self.iter().map(CairoSerialize::serialize).join(",")
                )
            }
        }

        let ser_proof = format!(
            r#"
            use stwo_cairo_verifier::fields::m31::m31;
            use stwo_cairo_verifier::fields::qm31::qm31;
            use stwo_cairo_verifier::vcs::verifier::MerkleDecommitment;
            use stwo_cairo_verifier::poly::line::LinePoly;
            use stwo_cairo_verifier::fri::{{FriLayerProof, FriProof}};
            use stwo_cairo_verifier::pcs::verifier::CommitmentSchemeProof;
            use stwo_cairo_verifier::verifier::StarkProof;
            
            pub fn proof() -> StarkProof {{ 
                {} 
            }}"#,
            proof.serialize()
        );

        let mut file = File::create("proof.cairo").unwrap();
        file.write_all(ser_proof.as_bytes()).unwrap();
    }
}
