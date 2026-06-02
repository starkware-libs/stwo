use itertools::Itertools;
use num_traits::One;
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::{PcsConfig, TreeSubspan};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::Column;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::{prove, CommitmentSchemeProver};
use stwo_constraint_framework::logup::LookupElements;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    assert_constraints_on_polys, relation, EvalAtRow, FrameworkComponent, FrameworkEval,
    LogupTraceGenerator, RelationEntry, TraceLocationAllocator,
};
use tracing::{span, Level};

pub type PlonkComponent = FrameworkComponent<PlonkEval>;

// TODO(alont): Rename this and all other `LookupElements` types to `Relation`.
relation!(PlonkLookupElements, 2);

#[derive(Clone)]
pub struct PlonkEval {
    pub log_n_rows: u32,
    pub lookup_elements: PlonkLookupElements,
    pub claimed_sum: SecureField,
    pub base_trace_location: TreeSubspan,
    pub interaction_trace_location: TreeSubspan,
    pub constants_trace_location: TreeSubspan,
}

impl FrameworkEval for PlonkEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let a_wire = eval.get_preprocessed_column(Plonk::new("wire_a".to_string()).id());
        let b_wire = eval.get_preprocessed_column(Plonk::new("wire_b".to_string()).id());
        // Note: c_wire could also be implicit: (self.eval.point() - M31_CIRCLE_GEN.into_ef()).x.
        //   A constant column is easier though.
        let c_wire = eval.get_preprocessed_column(Plonk::new("wire_c".to_string()).id());
        let op = eval.get_preprocessed_column(Plonk::new("op".to_string()).id());

        let mult = eval.next_trace_mask();
        let a_val = eval.next_trace_mask();
        let b_val = eval.next_trace_mask();
        let c_val = eval.next_trace_mask();

        eval.add_constraint(
            c_val.clone() - op.clone() * (a_val.clone() + b_val.clone())
                + (E::F::one() - op) * a_val.clone() * b_val.clone(),
        );

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::one(),
            &[a_wire, a_val],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::one(),
            &[b_wire, b_val],
        ));

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            (-mult).into(),
            &[c_wire, c_val],
        ));

        eval.finalize_logup_in_pairs();
        eval
    }
}

#[derive(Clone)]
pub struct PlonkCircuitTrace {
    pub mult: BaseColumn,
    pub a_wire: BaseColumn,
    pub b_wire: BaseColumn,
    pub c_wire: BaseColumn,
    pub op: BaseColumn,
    pub a_val: BaseColumn,
    pub b_val: BaseColumn,
    pub c_val: BaseColumn,
}
pub fn gen_trace(
    log_size: u32,
    circuit: &PlonkCircuitTrace,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let _span = span!(Level::INFO, "Generation").entered();

    let domain = CanonicCoset::new(log_size).circle_domain();
    [
        &circuit.mult,
        &circuit.a_val,
        &circuit.b_val,
        &circuit.c_val,
    ]
    .into_iter()
    .map(|eval| CircleEvaluation::new(domain, eval.clone()))
    .collect()
}

pub fn gen_interaction_trace(
    log_size: u32,
    circuit: &PlonkCircuitTrace,
    lookup_elements: &LookupElements<2>,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let _span = span!(Level::INFO, "Generate interaction trace").entered();
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    let mut col_gen = logup_gen.new_col();
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let q0: PackedSecureField =
            lookup_elements.combine(&[circuit.a_wire.data[vec_row], circuit.a_val.data[vec_row]]);
        let q1: PackedSecureField =
            lookup_elements.combine(&[circuit.b_wire.data[vec_row], circuit.b_val.data[vec_row]]);
        col_gen.write_frac(vec_row, q0 + q1, q0 * q1);
    }
    col_gen.finalize_col();

    let mut col_gen = logup_gen.new_col();
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let p = -circuit.mult.data[vec_row];
        let q: PackedSecureField =
            lookup_elements.combine(&[circuit.c_wire.data[vec_row], circuit.c_val.data[vec_row]]);
        col_gen.write_frac(vec_row, p.into(), q);
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
}

#[allow(unused)]
pub fn prove_fibonacci_plonk(
    log_n_rows: u32,
    config: PcsConfig,
) -> (PlonkComponent, StarkProof<Blake2sMerkleHasher>) {
    assert!(log_n_rows >= LOG_N_LANES);

    // Prepare a fibonacci circuit.
    let mut fib_values = vec![BaseField::one(), BaseField::one()];
    for _ in 0..(1 << log_n_rows) {
        fib_values.push(fib_values[fib_values.len() - 1] + fib_values[fib_values.len() - 2]);
    }
    let range = 0..(1 << log_n_rows);
    let mut circuit = PlonkCircuitTrace {
        mult: range.clone().map(|_| 2.into()).collect(),
        a_wire: range.clone().map(|i| i.into()).collect(),
        b_wire: range.clone().map(|i| (i + 1).into()).collect(),
        c_wire: range.clone().map(|i| (i + 2).into()).collect(),
        op: range.clone().map(|_| 1.into()).collect(),
        a_val: range.clone().map(|i| fib_values[i]).collect(),
        b_val: range.clone().map(|i| fib_values[i + 1]).collect(),
        c_val: range.clone().map(|i| fib_values[i + 2]).collect(),
    };
    circuit.mult.set((1 << log_n_rows) - 1, 0.into());
    circuit.mult.set((1 << log_n_rows) - 2, 1.into());

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + config.fri_config.log_blowup_factor + 1)
            .circle_domain()
            .half_coset,
    );
    span.exit();

    // Setup protocol.
    let channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<_, Blake2sMerkleChannel>::new(config, &twiddles);
    commitment_scheme.set_store_polynomials_coefficients();

    // Preprocessed trace.
    let span = span!(Level::INFO, "Constant").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    let mut constant_trace = [
        circuit.a_wire.clone(),
        circuit.b_wire.clone(),
        circuit.c_wire.clone(),
        circuit.op.clone(),
    ]
    .into_iter()
    .map(|col| {
        CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(
            CanonicCoset::new(log_n_rows).circle_domain(),
            col,
        )
    })
    .collect_vec();
    let constants_trace_location = tree_builder.extend_evals(constant_trace);
    tree_builder.commit(channel);
    span.exit();

    // Trace.
    let span = span!(Level::INFO, "Trace").entered();
    let trace = gen_trace(log_n_rows, &circuit);
    let mut tree_builder = commitment_scheme.tree_builder();
    let base_trace_location = tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();

    // Draw lookup element.
    let lookup_elements = PlonkLookupElements::draw(channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let (trace, claimed_sum) = gen_interaction_trace(log_n_rows, &circuit, &lookup_elements.0);
    let mut tree_builder = commitment_scheme.tree_builder();
    let interaction_trace_location = tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();
    // Prove constraints.
    let component = PlonkComponent::new(
        &mut TraceLocationAllocator::default(),
        PlonkEval {
            log_n_rows,
            lookup_elements,
            claimed_sum,
            base_trace_location,
            interaction_trace_location,
            constants_trace_location,
        },
        claimed_sum,
    );

    // Sanity check. Remove for production.
    let trace_polys = commitment_scheme.trees.as_ref().map(|t| {
        t.polynomials
            .iter()
            .map(|p| p.coeffs.clone().unwrap())
            .collect_vec()
    });
    let component_eval = component.clone();
    assert_constraints_on_polys(
        &trace_polys,
        CanonicCoset::new(log_n_rows),
        |assert_eval| {
            component_eval.evaluate(assert_eval);
        },
        claimed_sum,
    );

    let proof = prove(&[&component], channel, commitment_scheme).unwrap();

    (component, proof)
}

/// Preprocessed columns for describing a plonk circuit.
/// Each plonk gate is described by input wires `a_wire`, `b_wire`, output wire `c_wire`, and
/// operation `op`.  
#[derive(Debug)]
pub struct Plonk {
    pub name: String,
}
impl Plonk {
    pub const fn new(name: String) -> Self {
        Self { name }
    }

    pub fn id(&self) -> PreProcessedColumnId {
        PreProcessedColumnId {
            id: format!("preprocessed_plonk_{}", self.name),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::env;

    use num_traits::Zero;
    use stwo::core::air::Component;
    use stwo::core::channel::Blake2sChannel;
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::fri::FriConfig;
    use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeSubspan, TreeVec};
    use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
    use stwo::core::verifier::verify;
    use stwo::core::zk::{
        build_zk_air_metadata_from_privacy_provider, zk_singleton_column_ranges, ZkAirId,
        ZkAirMetadataBuildError, ZkAirPrivacyProvider, ZkColumnRange, ZkDependencyKind,
        ZkDependencyMetadataCompleteness, ZkPrivacyDependency, ZkPrivacyInferenceMode,
        ZkPrivacyReason, ZkPrivateColumnUsage, ZkPrivateRoot, ZkTraceTreeScope,
    };
    use stwo_constraint_framework::TraceLocationAllocator;

    use crate::plonk::{prove_fibonacci_plonk, PlonkComponent, PlonkEval, PlonkLookupElements};

    struct PlonkZkPrivacyProvider {
        component: PlonkComponent,
        log_n_rows: u32,
    }

    impl PlonkZkPrivacyProvider {
        fn new(log_n_rows: u32) -> Self {
            Self {
                component: plonk_component_for_metadata(log_n_rows),
                log_n_rows,
            }
        }
    }

    fn plonk_component_for_metadata(log_n_rows: u32) -> PlonkComponent {
        let dummy_trace_location = TreeSubspan {
            tree_index: 0,
            col_start: 0,
            col_end: 0,
        };

        PlonkComponent::new(
            &mut TraceLocationAllocator::default(),
            PlonkEval {
                log_n_rows,
                lookup_elements: PlonkLookupElements::dummy(),
                claimed_sum: SecureField::zero(),
                base_trace_location: dummy_trace_location,
                interaction_trace_location: dummy_trace_location,
                constants_trace_location: dummy_trace_location,
            },
            SecureField::zero(),
        )
    }

    impl ZkAirPrivacyProvider for PlonkZkPrivacyProvider {
        fn air_id(&self) -> ZkAirId {
            ZkAirId(b"stwo.examples.plonk.private-logup.blocked.zk.v1".to_vec())
        }

        fn component_column_log_sizes(&self) -> TreeVec<Vec<u32>> {
            self.component.trace_log_degree_bounds()
        }

        fn max_constraint_log_degree_bound(&self) -> u32 {
            self.component.max_constraint_log_degree_bound()
        }

        fn trace_tree_scopes(&self) -> Vec<ZkTraceTreeScope> {
            vec![
                ZkTraceTreeScope::Preprocessed,
                ZkTraceTreeScope::OriginalTrace,
                ZkTraceTreeScope::InteractionTrace {
                    interaction_index: 0,
                },
            ]
        }

        fn public_roots(&self) -> Vec<ZkColumnRange> {
            let log_sizes = self.component.trace_log_degree_bounds();
            if log_sizes[0].is_empty() {
                vec![]
            } else {
                vec![ZkColumnRange::new(0, 0, log_sizes[0].len())]
            }
        }

        fn private_roots(&self) -> Vec<ZkPrivateRoot> {
            let log_sizes = self.component.trace_log_degree_bounds();
            zk_singleton_column_ranges(1, log_sizes[1].len())
                .into_iter()
                .map(|range| ZkPrivateRoot {
                    range,
                    usage: ZkPrivateColumnUsage::OrdinaryWitness,
                    reason: ZkPrivacyReason::Witness,
                })
                .collect()
        }

        fn dependency_edges(&self) -> Vec<ZkPrivacyDependency> {
            let log_sizes = self.component.trace_log_degree_bounds();
            if log_sizes[1].is_empty() || log_sizes[2].is_empty() {
                return vec![];
            }

            vec![ZkPrivacyDependency {
                from: ZkColumnRange::new(1, 0, log_sizes[1].len()),
                to: ZkColumnRange::new(2, 0, log_sizes[2].len()),
                kind: ZkDependencyKind::LogUpRunningSum,
            }]
        }

        fn dependency_metadata_completeness(&self) -> ZkDependencyMetadataCompleteness {
            ZkDependencyMetadataCompleteness::CompleteTraceAndInteractionClosure
        }

        fn application_domain(&self) -> &[u8] {
            b"stwo.examples.plonk.zk-public-statement.v1"
        }

        fn application_statement(&self) -> Vec<u8> {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(b"fibonacci-plonk-public-circuit-private-logup-blocked-v2");
            bytes.extend_from_slice(&self.log_n_rows.to_le_bytes());
            for label in [
                b"wire_a".as_slice(),
                b"wire_b".as_slice(),
                b"wire_c".as_slice(),
                b"op".as_slice(),
            ] {
                bytes.extend_from_slice(&(label.len() as u64).to_le_bytes());
                bytes.extend_from_slice(label);
            }
            let log_sizes = self.component.trace_log_degree_bounds();
            bytes.extend_from_slice(&(log_sizes.0.len() as u64).to_le_bytes());
            for tree_log_sizes in &log_sizes.0 {
                bytes.extend_from_slice(&(tree_log_sizes.len() as u64).to_le_bytes());
                for log_size in tree_log_sizes {
                    bytes.extend_from_slice(&log_size.to_le_bytes());
                }
            }
            bytes
        }
    }

    #[test_log::test]
    fn test_simd_plonk_prove() {
        // Get from environment variable:
        let log_n_instances = env::var("LOG_N_INSTANCES")
            .unwrap_or_else(|_| "10".to_string())
            .parse::<u32>()
            .unwrap();
        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(5, 4, 64, 1),
            lifting_log_size: None,
        };

        // Prove.
        let (component, proof) = prove_fibonacci_plonk(log_n_instances, config);

        // Verify.
        // TODO: Create Air instance independently.
        let channel = &mut Blake2sChannel::default();
        let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

        // Decommit.
        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = component.trace_log_degree_bounds();

        // Preprocessed columns.
        commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);

        // Trace columns.
        commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);
        // Draw lookup element.
        let lookup_elements = PlonkLookupElements::draw(channel);
        assert_eq!(lookup_elements, component.lookup_elements);
        // Interaction columns.
        commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

        verify(&[&component], channel, commitment_scheme, proof).unwrap();
    }

    #[test]
    fn test_plonk_zk_provider_uses_component_trace_bounds_and_air_bounds() {
        let log_n_rows = 10;
        let provider = PlonkZkPrivacyProvider::new(log_n_rows);
        let component = plonk_component_for_metadata(log_n_rows);
        let log_sizes = component.trace_log_degree_bounds();

        assert_eq!(provider.component_column_log_sizes().0, log_sizes.0);
        assert_eq!(
            provider.max_constraint_log_degree_bound(),
            component.max_constraint_log_degree_bound()
        );
        assert_eq!(
            provider.trace_tree_scopes(),
            vec![
                ZkTraceTreeScope::Preprocessed,
                ZkTraceTreeScope::OriginalTrace,
                ZkTraceTreeScope::InteractionTrace {
                    interaction_index: 0,
                },
            ]
        );
        assert_eq!(
            provider.public_roots(),
            vec![ZkColumnRange::new(0, 0, log_sizes[0].len())]
        );

        let private_roots = provider.private_roots();
        assert_eq!(private_roots.len(), log_sizes[1].len());
        for (column, root) in private_roots.iter().enumerate() {
            assert_eq!(root.range, ZkColumnRange::new(1, column, column + 1));
            assert!(root.range.is_singleton());
            assert_eq!(root.usage, ZkPrivateColumnUsage::OrdinaryWitness);
            assert_eq!(root.reason, ZkPrivacyReason::Witness);
        }

        assert_eq!(
            provider.dependency_edges(),
            vec![ZkPrivacyDependency {
                from: ZkColumnRange::new(1, 0, log_sizes[1].len()),
                to: ZkColumnRange::new(2, 0, log_sizes[2].len()),
                kind: ZkDependencyKind::LogUpRunningSum,
            }]
        );
    }

    #[test]
    fn test_plonk_zk_provider_fails_closed_on_private_logup_claims() {
        let provider = PlonkZkPrivacyProvider::new(10);

        assert!(matches!(
            build_zk_air_metadata_from_privacy_provider(
                &provider,
                PcsConfig::default().fri_config.log_blowup_factor,
                ZkPrivacyInferenceMode::FailClosed,
            ),
            Err(ZkAirMetadataBuildError::IncompleteLogupClaimMetadata)
        ));
    }
}
