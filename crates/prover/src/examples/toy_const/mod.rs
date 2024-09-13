mod constraints;
mod gen;

#[cfg(test)]
mod tests {
    use itertools::chain;

    use crate::constraint_framework::constant_columns::StaticTree;
    use crate::constraint_framework::{FrameworkComponent, TraceLocationAllocator};
    use crate::core::backend::simd::SimdBackend;
    use crate::core::channel::Blake2sChannel;
    use crate::core::pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig};
    use crate::core::poly::circle::{CanonicCoset, PolyOps};
    use crate::core::prover::{prove, verify};
    use crate::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use crate::examples::toy_const::constraints::{Add1Eval, Add2Eval};
    use crate::examples::toy_const::gen::{gen_add_1_trace, gen_add_2_trace, gen_const_1_trace};

    #[test]
    fn test_toy_const() {
        const LOG_N_INSTANCES: u32 = 6;
        let config = PcsConfig::default();
        // Precompute twiddles.
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(LOG_N_INSTANCES + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        // Setup protocol.
        let prover_channel = &mut Blake2sChannel::default();
        let commitment_scheme =
            &mut CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(
                config, &twiddles,
            );
        let tree_span_provider = &mut TraceLocationAllocator::default();
        tree_span_provider.static_table_offsets = StaticTree::add1(LOG_N_INSTANCES).locations;

        // Constant Trace.
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(gen_const_1_trace(LOG_N_INSTANCES));
        tree_builder.commit(prover_channel);

        // Trace.
        let add_1_trace = gen_add_1_trace(LOG_N_INSTANCES);
        let add_2_trace = gen_add_2_trace(LOG_N_INSTANCES);
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(chain![add_1_trace, add_2_trace]);
        tree_builder.commit(prover_channel);

        // Prove constraints.
        let add_1_component = FrameworkComponent::<Add1Eval>::new(
            tree_span_provider,
            Add1Eval {
                log_size: LOG_N_INSTANCES,
            },
        );
        let add_2_component = FrameworkComponent::<Add2Eval>::new(
            tree_span_provider,
            Add2Eval {
                log_size: LOG_N_INSTANCES,
            },
        );

        let proof = prove::<SimdBackend, Blake2sMerkleChannel>(
            &[&add_1_component, &add_2_component],
            prover_channel,
            commitment_scheme,
        )
        .unwrap();

        // Verify.
        let verifier_channel = &mut Blake2sChannel::default();
        let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = vec![6; 4];
        commitment_scheme.commit(proof.commitments[0], &[6], verifier_channel);
        commitment_scheme.commit(proof.commitments[1], &sizes, verifier_channel);
        verify(
            &[&add_1_component, &add_2_component],
            verifier_channel,
            commitment_scheme,
            proof,
        )
        .unwrap();
    }
}
