use itertools::Itertools;
use tracing::{span, Level};

use super::{AirTraceGenerator, AirTraceVerifier, BASE_TRACE, INTERACTION_TRACE};
use crate::core::air::{Air, AirExt, AirProverExt};
use crate::core::backend::Backend;
use crate::core::channel::Channel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, MAX_CIRCLE_DOMAIN_LOG_SIZE};
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::BitReversedOrder;
use crate::core::prover::{
    prove, verify, ProvingError, StarkProof, VerificationError, LOG_BLOWUP_FACTOR,
};
use crate::core::vcs::ops::{MerkleHasher, MerkleOps};
use crate::core::{ColumnVec, InteractionElements};

pub fn commit_and_prove<B, H, C>(
    air: &impl AirTraceGenerator<B>,
    channel: &mut C,
    trace: ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
) -> Result<StarkProof<H>, ProvingError>
where
    B: Backend + MerkleOps<H>,
    C: Channel,
    H: MerkleHasher<Hash = C::Digest>,
{
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
    let composition_polynomial_log_degree_bound = air.composition_log_degree_bound();
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

    prove(&air, channel, &interaction_elements, &mut commitment_scheme)
}

pub fn evaluate_and_commit_on_trace<'a, B, H, C>(
    air: &impl AirTraceGenerator<B>,
    channel: &mut C,
    twiddles: &'a TwiddleTree<B>,
    trace: ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
) -> Result<(CommitmentSchemeProver<'a, B, H>, InteractionElements), ProvingError>
where
    B: Backend + MerkleOps<H>,
    C: Channel,
    H: MerkleHasher<Hash = C::Digest>,
{
    let mut commitment_scheme = CommitmentSchemeProver::new(LOG_BLOWUP_FACTOR, twiddles);
    // TODO(spapini): Remove clone.
    let span = span!(Level::INFO, "Trace").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace.clone());
    tree_builder.commit(channel);
    span.exit();

    let interaction_elements = air.interaction_elements(channel);
    let interaction_trace = air.interact(&trace, &interaction_elements);
    // TODO(spapini): Make this symmetric with verify, once the TraceGenerator traits support
    // retrieveing the column log sizes.
    if !interaction_trace.is_empty() {
        let _span = span!(Level::INFO, "Interaction").entered();
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(interaction_trace);
        tree_builder.commit(channel);
    }

    Ok((commitment_scheme, interaction_elements))
}

pub fn commit_and_verify<H, C>(
    proof: StarkProof<H>,
    air: &(impl Air + AirTraceVerifier),
    channel: &mut C,
) -> Result<(), VerificationError>
where
    C: Channel,
    H: MerkleHasher<Hash = C::Digest>,
{
    // Read trace commitment.
    let mut commitment_scheme = CommitmentSchemeVerifier::new();

    // TODO(spapini): Retrieve column_log_sizes from AirTraceVerifier, and remove the dependency on
    // Air.
    let column_log_sizes = air.column_log_sizes();
    commitment_scheme.commit(
        proof.commitments[BASE_TRACE],
        &column_log_sizes[BASE_TRACE],
        channel,
    );
    let interaction_elements = air.interaction_elements(channel);

    if air.column_log_sizes().len() == 2 {
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
    air.verify_lookups(&proof.lookup_values)?;
    verify(
        air,
        channel,
        &interaction_elements,
        &mut commitment_scheme,
        proof,
    )
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
    use crate::core::air::{Air, AirProver, Component, ComponentProver, ComponentTrace};
    use crate::core::backend::cpu::CpuCircleEvaluation;
    use crate::core::backend::CpuBackend;
    use crate::core::channel::Channel;
    use crate::core::circle::{CirclePoint, CirclePointIndex, Coset};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::{
        CanonicCoset, CircleDomain, CircleEvaluation, MAX_CIRCLE_DOMAIN_LOG_SIZE,
    };
    use crate::core::poly::BitReversedOrder;
    use crate::core::prover::{ProvingError, VerificationError};
    use crate::core::test_utils::test_channel;
    use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
    use crate::core::{ColumnVec, InteractionElements, LookupValues};
    use crate::qm31;
    use crate::trace_generation::registry::ComponentGenerationRegistry;
    use crate::trace_generation::{
        commit_and_prove, AirTraceGenerator, AirTraceVerifier, ComponentTraceGenerator,
    };

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
        fn interaction_elements(&self, _channel: &mut impl Channel) -> InteractionElements {
            InteractionElements::default()
        }

        fn verify_lookups(&self, _lookup_values: &LookupValues) -> Result<(), VerificationError> {
            Ok(())
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

        let proof_error =
            commit_and_prove::<_, Blake2sMerkleHasher, _>(&air, &mut test_channel(), trace)
                .unwrap_err();
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

        let proof_error =
            commit_and_prove::<_, Blake2sMerkleHasher, _>(&air, &mut test_channel(), trace)
                .unwrap_err();
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

        let proof = commit_and_prove::<_, Blake2sMerkleHasher, _>(&air, &mut test_channel(), trace)
            .unwrap_err();
        assert!(matches!(proof, ProvingError::ConstraintsNotSatisfied));
    }
}
