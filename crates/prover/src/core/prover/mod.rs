use itertools::Itertools;
use thiserror::Error;
use tracing::{span, Level};

use super::air::AirProver;
use super::backend::Backend;
use super::fields::secure_column::SECURE_EXTENSION_DEGREE;
use super::fri::FriVerificationError;
use super::pcs::{CommitmentSchemeProof, TreeVec};
use super::poly::circle::{CanonicCoset, MAX_CIRCLE_DOMAIN_LOG_SIZE};
use super::poly::twiddles::TwiddleTree;
use super::proof_of_work::ProofOfWorkVerificationError;
use super::{ColumnVec, InteractionElements, LookupValues};
use crate::core::air::{Air, AirExt, AirProverExt};
use crate::core::backend::CpuBackend;
use crate::core::channel::{Blake2sChannel, Channel as ChannelTrait};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier};
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::vcs::blake2_hash::Blake2sHasher;
use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use crate::core::vcs::hasher::Hasher;
use crate::core::vcs::ops::MerkleOps;
use crate::core::vcs::verifier::MerkleVerificationError;
use crate::trace_generation::{AirTraceGenerator, AirTraceVerifier};

type Channel = Blake2sChannel;
type ChannelHasher = Blake2sHasher;
type MerkleHasher = Blake2sMerkleHasher;

pub const LOG_BLOWUP_FACTOR: u32 = 1;
pub const LOG_LAST_LAYER_DEGREE_BOUND: u32 = 0;
pub const PROOF_OF_WORK_BITS: u32 = 12;
pub const N_QUERIES: usize = 3;

pub const BASE_TRACE: usize = 0;
pub const INTERACTION_TRACE: usize = 1;

#[derive(Debug)]
pub struct StarkProof {
    pub commitments: TreeVec<<ChannelHasher as Hasher>::Hash>,
    pub lookup_values: LookupValues,
    pub commitment_scheme_proof: CommitmentSchemeProof,
}

#[derive(Debug)]
pub struct AdditionalProofData {
    pub composition_polynomial_oods_value: SecureField,
    pub composition_polynomial_random_coeff: SecureField,
    pub oods_point: CirclePoint<SecureField>,
    pub oods_quotients: Vec<CircleEvaluation<CpuBackend, SecureField, BitReversedOrder>>,
}

pub fn evaluate_and_commit_on_trace<B: Backend + MerkleOps<MerkleHasher>>(
    air: &impl AirTraceGenerator<B>,
    channel: &mut Channel,
    twiddles: &TwiddleTree<B>,
    trace: ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
) -> Result<(CommitmentSchemeProver<B>, InteractionElements), ProvingError> {
    let mut commitment_scheme = CommitmentSchemeProver::new(LOG_BLOWUP_FACTOR);
    // TODO(spapini): Remove clone.
    let span = span!(Level::INFO, "Trace").entered();
    commitment_scheme.commit_on_evals(trace.clone(), channel, twiddles);
    span.exit();

    let interaction_elements = air.interaction_elements(channel);
    let interaction_trace = air.interact(&trace, &interaction_elements);
    if !interaction_trace.is_empty() {
        let _span = span!(Level::INFO, "Interaction").entered();
        commitment_scheme.commit_on_evals(interaction_trace, channel, twiddles);
    }

    Ok((commitment_scheme, interaction_elements))
}

// TODO(spapini): Rename to prove, and rename prover to prove_and_commit.
pub fn prove_without_commit<B: Backend + MerkleOps<MerkleHasher>>(
    air: &impl AirProver<B>,
    channel: &mut Channel,
    interaction_elements: &InteractionElements,
    twiddles: &TwiddleTree<B>,
    commitment_scheme: &mut CommitmentSchemeProver<B>,
) -> Result<StarkProof, ProvingError> {
    let component_traces = air.component_traces(&commitment_scheme.trees);
    let lookup_values = air.lookup_values(&component_traces);

    // Evaluate and commit on composition polynomial.
    let random_coeff = channel.draw_felt();

    let span = span!(Level::INFO, "Composition").entered();
    let span1 = span!(Level::INFO, "Generation").entered();
    let composition_polynomial_poly = air.compute_composition_polynomial(
        random_coeff,
        &component_traces,
        interaction_elements,
        &lookup_values,
    );
    span1.exit();

    commitment_scheme.commit(composition_polynomial_poly.to_vec(), channel, twiddles);
    span.exit();

    // Draw OODS point.
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);

    // Get mask sample points relative to oods point.
    let sample_points = air.mask_points(oods_point);

    // Prove the trace and composition OODS values, and retrieve them.
    let commitment_scheme_proof = commitment_scheme.prove_values(sample_points, channel, twiddles);

    // Evaluate composition polynomial at OODS point and check that it matches the trace OODS
    // values. This is a sanity check.
    // TODO(spapini): Save clone.
    let (trace_oods_values, composition_oods_value) =
        sampled_values_to_mask(air, &commitment_scheme_proof.sampled_values).unwrap();

    if composition_oods_value
        != air.eval_composition_polynomial_at_point(
            oods_point,
            &trace_oods_values,
            random_coeff,
            interaction_elements,
            &lookup_values,
        )
    {
        return Err(ProvingError::ConstraintsNotSatisfied);
    }

    Ok(StarkProof {
        commitments: commitment_scheme.roots(),
        lookup_values,
        commitment_scheme_proof,
    })
}

pub fn prove<B: Backend + MerkleOps<MerkleHasher>>(
    air: &impl AirTraceGenerator<B>,
    channel: &mut Channel,
    trace: ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
) -> Result<StarkProof, ProvingError> {
    // Check that traces are not too big.
    for (i, trace) in trace.iter().enumerate() {
        if trace.domain.log_size() + LOG_BLOWUP_FACTOR > MAX_CIRCLE_DOMAIN_LOG_SIZE {
            return Err(ProvingError::MaxTraceDegreeExceeded {
                trace_index: i,
                degree: trace.domain.log_size(),
            });
        }
    }

    // Check that the composition polynomial is not too big.
    // TODO(AlonH): Get traces log degree bounds from trace writer.
    let composition_polynomial_log_degree_bound = air.composition_log_degree_bound();
    if composition_polynomial_log_degree_bound + LOG_BLOWUP_FACTOR > MAX_CIRCLE_DOMAIN_LOG_SIZE {
        return Err(ProvingError::MaxCompositionDegreeExceeded {
            degree: composition_polynomial_log_degree_bound,
        });
    }

    let span = span!(Level::INFO, "Precompute twiddle").entered();
    let twiddles = B::precompute_twiddles(
        CanonicCoset::new(composition_polynomial_log_degree_bound + LOG_BLOWUP_FACTOR)
            .circle_domain()
            .half_coset,
    );
    span.exit();

    let (mut commitment_scheme, interaction_elements) =
        evaluate_and_commit_on_trace(air, channel, &twiddles, trace)?;

    let air = air.to_air_prover();
    channel.mix_felts(
        &air.lookup_values(&air.component_traces(&commitment_scheme.trees))
            .0
            .values()
            .map(|v| SecureField::from(*v))
            .collect_vec(),
    );

    prove_without_commit(
        &air,
        channel,
        &interaction_elements,
        &twiddles,
        &mut commitment_scheme,
    )
}

pub fn verify(
    proof: StarkProof,
    air: &(impl Air + AirTraceVerifier),
    channel: &mut Channel,
) -> Result<(), VerificationError> {
    // Read trace commitment.
    let mut commitment_scheme = CommitmentSchemeVerifier::new();
    let column_log_sizes = air.column_log_sizes();
    commitment_scheme.commit(
        proof.commitments[BASE_TRACE],
        &column_log_sizes[BASE_TRACE],
        channel,
    );
    let interaction_elements = air.interaction_elements(channel);

    if air.n_interaction_phases() == 2 {
        commitment_scheme.commit(
            proof.commitments[INTERACTION_TRACE],
            &column_log_sizes[INTERACTION_TRACE],
            channel,
        );
    }

    channel.mix_felts(
        &proof
            .lookup_values
            .0
            .values()
            .map(|v| SecureField::from(*v))
            .collect_vec(),
    );
    verify_without_commit(
        air,
        channel,
        &interaction_elements,
        &mut commitment_scheme,
        proof,
    )
}

pub fn verify_without_commit(
    air: &impl Air,
    channel: &mut Blake2sChannel,
    interaction_elements: &InteractionElements,
    commitment_scheme: &mut CommitmentSchemeVerifier,
    proof: StarkProof,
) -> Result<(), VerificationError> {
    let random_coeff = channel.draw_felt();

    // Read composition polynomial commitment.
    commitment_scheme.commit(
        *proof.commitments.last().unwrap(),
        &[air.composition_log_degree_bound(); SECURE_EXTENSION_DEGREE],
        channel,
    );

    // Draw OODS point.
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);

    // Get mask sample points relative to oods point.
    let sample_points = air.mask_points(oods_point);

    // TODO(spapini): Save clone.
    let (trace_oods_values, composition_oods_value) = sampled_values_to_mask(
        air,
        &proof.commitment_scheme_proof.sampled_values,
    )
    .map_err(|_| {
        VerificationError::InvalidStructure("Unexpected sampled_values structure".to_string())
    })?;

    if composition_oods_value
        != air.eval_composition_polynomial_at_point(
            oods_point,
            &trace_oods_values,
            random_coeff,
            interaction_elements,
            &proof.lookup_values,
        )
    {
        return Err(VerificationError::OodsNotMatching);
    }

    commitment_scheme.verify_values(sample_points, proof.commitment_scheme_proof, channel)
}

#[allow(clippy::type_complexity)]
/// Structures the tree-wise sampled values into component-wise OODS values and a composition
/// polynomial OODS value.
fn sampled_values_to_mask(
    air: &impl Air,
    sampled_values: &TreeVec<ColumnVec<Vec<SecureField>>>,
) -> Result<(Vec<TreeVec<Vec<Vec<SecureField>>>>, SecureField), InvalidOodsSampleStructure> {
    let mut sampled_values = sampled_values.as_ref();
    let composition_values = sampled_values.pop().ok_or(InvalidOodsSampleStructure)?;

    let mut sample_iters = sampled_values.map(|tree_value| tree_value.iter());
    let trace_oods_values = air
        .components()
        .iter()
        .map(|component| {
            component
                .mask_points(CirclePoint::zero())
                .zip(sample_iters.as_mut())
                .map(|(mask_per_tree, tree_iter)| {
                    tree_iter.take(mask_per_tree.len()).cloned().collect_vec()
                })
        })
        .collect_vec();

    let composition_oods_value = SecureField::from_partial_evals(
        composition_values
            .iter()
            .flatten()
            .cloned()
            .collect_vec()
            .try_into()
            .map_err(|_| InvalidOodsSampleStructure)?,
    );

    Ok((trace_oods_values, composition_oods_value))
}

/// Error when the sampled values have an invalid structure.
#[derive(Clone, Copy, Debug)]
pub struct InvalidOodsSampleStructure;

#[derive(Clone, Copy, Debug, Error)]
pub enum ProvingError {
    #[error(
        "Trace column {trace_index} log degree bound ({degree}) exceeded max log degree ({}).",
        MAX_CIRCLE_DOMAIN_LOG_SIZE - LOG_BLOWUP_FACTOR
    )]
    MaxTraceDegreeExceeded { trace_index: usize, degree: u32 },
    #[error(
        "Composition polynomial log degree bound ({degree}) exceeded max log degree ({}).",
        MAX_CIRCLE_DOMAIN_LOG_SIZE - LOG_BLOWUP_FACTOR
    )]
    MaxCompositionDegreeExceeded { degree: u32 },
    #[error("Constraints not satisfied.")]
    ConstraintsNotSatisfied,
}

#[derive(Clone, Debug, Error)]
pub enum VerificationError {
    #[error("Proof has invalid structure: {0}.")]
    InvalidStructure(String),
    #[error(transparent)]
    Merkle(#[from] MerkleVerificationError),
    #[error(
        "The composition polynomial OODS value does not match the trace OODS values
    (DEEP-ALI failure)."
    )]
    OodsNotMatching,
    #[error(transparent)]
    Fri(#[from] FriVerificationError),
    #[error(transparent)]
    ProofOfWork(#[from] ProofOfWorkVerificationError),
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
    use crate::core::air::{Air, AirProver, Component, ComponentProver, ComponentTrace};
    use crate::core::backend::cpu::CpuCircleEvaluation;
    use crate::core::backend::CpuBackend;
    use crate::core::channel::Blake2sChannel;
    use crate::core::circle::{CirclePoint, CirclePointIndex, Coset};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::{
        CanonicCoset, CircleDomain, CircleEvaluation, MAX_CIRCLE_DOMAIN_LOG_SIZE,
    };
    use crate::core::poly::BitReversedOrder;
    use crate::core::prover::{prove, ProvingError};
    use crate::core::test_utils::test_channel;
    use crate::core::{ColumnVec, InteractionElements, LookupValues};
    use crate::qm31;
    use crate::trace_generation::registry::ComponentGenerationRegistry;
    use crate::trace_generation::{AirTraceGenerator, AirTraceVerifier, ComponentTraceGenerator};

    #[derive(Clone)]
    struct TestAir<C: ComponentProver<CpuBackend>> {
        component: C,
    }

    impl Air for TestAir<TestComponent> {
        fn components(&self) -> Vec<&dyn Component> {
            vec![&self.component]
        }
    }

    impl AirTraceVerifier for TestAir<TestComponent> {
        fn interaction_elements(&self, _channel: &mut Blake2sChannel) -> InteractionElements {
            InteractionElements::default()
        }
    }

    impl AirTraceGenerator<CpuBackend> for TestAir<TestComponent> {
        fn interact(
            &self,
            _trace: &ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
            _elements: &InteractionElements,
        ) -> Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
            vec![]
        }

        fn to_air_prover(&self) -> impl AirProver<CpuBackend> {
            self.clone()
        }

        fn composition_log_degree_bound(&self) -> u32 {
            self.component.max_constraint_log_degree_bound()
        }
    }

    impl AirProver<CpuBackend> for TestAir<TestComponent> {
        fn prover_components(&self) -> Vec<&dyn ComponentProver<CpuBackend>> {
            vec![&self.component]
        }
    }

    #[derive(Clone)]
    struct TestComponent {
        log_size: u32,
        max_constraint_log_degree_bound: u32,
    }

    impl Component for TestComponent {
        fn n_constraints(&self) -> usize {
            0
        }

        fn max_constraint_log_degree_bound(&self) -> u32 {
            self.max_constraint_log_degree_bound
        }

        fn n_interaction_phases(&self) -> u32 {
            1
        }

        fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
            TreeVec::new(vec![vec![self.log_size]])
        }

        fn mask_points(
            &self,
            point: CirclePoint<SecureField>,
        ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
            TreeVec::new(vec![vec![vec![point]]])
        }

        fn evaluate_constraint_quotients_at_point(
            &self,
            _point: CirclePoint<SecureField>,
            _mask: &TreeVec<Vec<Vec<SecureField>>>,
            evaluation_accumulator: &mut PointEvaluationAccumulator,
            _interaction_elements: &InteractionElements,
            _lookup_values: &LookupValues,
        ) {
            evaluation_accumulator.accumulate(qm31!(0, 0, 0, 1))
        }
    }

    impl ComponentTraceGenerator<CpuBackend> for TestComponent {
        type Component = Self;
        type Inputs = ();

        fn add_inputs(&mut self, _inputs: &Self::Inputs) {}

        fn write_trace(
            _component_id: &str,
            _registry: &mut ComponentGenerationRegistry,
        ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
            vec![]
        }

        fn write_interaction_trace(
            &self,
            _trace: &ColumnVec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
            _elements: &InteractionElements,
        ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
            vec![]
        }

        fn component(&self) -> Self::Component {
            self.clone()
        }
    }

    impl ComponentProver<CpuBackend> for TestComponent {
        fn evaluate_constraint_quotients_on_domain(
            &self,
            _trace: &ComponentTrace<'_, CpuBackend>,
            _evaluation_accumulator: &mut DomainEvaluationAccumulator<CpuBackend>,
            _interaction_elements: &InteractionElements,
            _lookup_values: &LookupValues,
        ) {
            // Does nothing.
        }

        fn lookup_values(&self, _trace: &ComponentTrace<'_, CpuBackend>) -> LookupValues {
            LookupValues::default()
        }
    }

    // Ignored because it takes too long and too much memory (in the CI) to run.
    #[test]
    #[ignore]
    fn test_trace_too_big() {
        const LOG_DOMAIN_SIZE: u32 = MAX_CIRCLE_DOMAIN_LOG_SIZE;
        let air = TestAir {
            component: TestComponent {
                log_size: LOG_DOMAIN_SIZE,
                max_constraint_log_degree_bound: LOG_DOMAIN_SIZE,
            },
        };
        let domain = CircleDomain::new(Coset::new(
            CirclePointIndex::generator(),
            LOG_DOMAIN_SIZE - 1,
        ));
        let values = vec![BaseField::zero(); 1 << LOG_DOMAIN_SIZE];
        let trace = vec![CpuCircleEvaluation::new(domain, values)];

        let proof_error = prove(&air, &mut test_channel(), trace).unwrap_err();
        assert!(matches!(
            proof_error,
            ProvingError::MaxTraceDegreeExceeded {
                trace_index: 0,
                degree: LOG_DOMAIN_SIZE
            }
        ));
    }

    #[test]
    fn test_composition_polynomial_too_big() {
        const COMPOSITION_POLYNOMIAL_DEGREE: u32 = MAX_CIRCLE_DOMAIN_LOG_SIZE;
        const LOG_DOMAIN_SIZE: u32 = 5;
        let air = TestAir {
            component: TestComponent {
                log_size: LOG_DOMAIN_SIZE,
                max_constraint_log_degree_bound: COMPOSITION_POLYNOMIAL_DEGREE,
            },
        };
        let domain = CircleDomain::new(Coset::new(
            CirclePointIndex::generator(),
            LOG_DOMAIN_SIZE - 1,
        ));
        let values = vec![BaseField::zero(); 1 << LOG_DOMAIN_SIZE];
        let trace = vec![CpuCircleEvaluation::new(domain, values)];

        let proof_error = prove(&air, &mut test_channel(), trace).unwrap_err();
        assert!(matches!(
            proof_error,
            ProvingError::MaxCompositionDegreeExceeded {
                degree: COMPOSITION_POLYNOMIAL_DEGREE
            }
        ));
    }

    #[test]
    fn test_constraints_not_satisfied() {
        const LOG_DOMAIN_SIZE: u32 = 5;
        let air = TestAir {
            component: TestComponent {
                log_size: LOG_DOMAIN_SIZE,
                max_constraint_log_degree_bound: LOG_DOMAIN_SIZE + 1,
            },
        };
        let domain = CanonicCoset::new(LOG_DOMAIN_SIZE).circle_domain();
        let values = vec![BaseField::zero(); 1 << LOG_DOMAIN_SIZE];
        let trace = vec![CpuCircleEvaluation::new(domain, values)];

        let proof = prove(&air, &mut test_channel(), trace).unwrap_err();
        assert!(matches!(proof, ProvingError::ConstraintsNotSatisfied));
    }
}
