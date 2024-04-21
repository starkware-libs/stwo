use itertools::Itertools;
use thiserror::Error;
use tracing::{span, Level};

use super::air::AirProver;
use super::backend::Backend;
use super::fri::FriVerificationError;
use super::pcs::{CommitmentSchemeProof, TreeVec};
use super::poly::circle::{CanonicCoset, SecureCirclePoly, MAX_CIRCLE_DOMAIN_LOG_SIZE};
use super::proof_of_work::ProofOfWorkVerificationError;
use super::ColumnVec;
use crate::core::air::{Air, AirExt, AirProverExt};
use crate::core::backend::CPUBackend;
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
use crate::core::ComponentVec;

type Channel = Blake2sChannel;
type ChannelHasher = Blake2sHasher;
type MerkleHasher = Blake2sMerkleHasher;

pub const LOG_BLOWUP_FACTOR: u32 = 1;
pub const LOG_LAST_LAYER_DEGREE_BOUND: u32 = 0;
pub const PROOF_OF_WORK_BITS: u32 = 12;
pub const N_QUERIES: usize = 3;

#[derive(Debug)]
pub struct StarkProof {
    pub commitments: TreeVec<<ChannelHasher as Hasher>::Hash>,
    pub commitment_scheme_proof: CommitmentSchemeProof,
}

#[derive(Debug)]
pub struct AdditionalProofData {
    pub composition_polynomial_oods_value: SecureField,
    pub composition_polynomial_random_coeff: SecureField,
    pub oods_point: CirclePoint<SecureField>,
    pub oods_quotients: Vec<CircleEvaluation<CPUBackend, SecureField, BitReversedOrder>>,
}

pub fn prove<B: Backend + MerkleOps<MerkleHasher>>(
    air: &impl AirProver<B>,
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

    // Evaluate and commit on trace.
    // TODO(spapini): Commit on trace outside.
    let span = span!(Level::INFO, "Trace interpolation").entered();
    let trace_polys = trace
        .clone()
        .into_iter()
        .map(|poly| poly.interpolate_with_twiddles(&twiddles))
        .collect();
    span.exit();

    let mut commitment_scheme = CommitmentSchemeProver::new(LOG_BLOWUP_FACTOR);
    let span = span!(Level::INFO, "Trace commitment").entered();
    commitment_scheme.commit(trace_polys, channel, &twiddles);
    span.exit();

    let interaction_elements = air.interaction_elements(channel);
    let interaction_traces = air.interact(&trace, &interaction_elements);
    let interaction_trace_polys = interaction_traces
        .0
        .into_iter()
        .flat_map(|trace| {
            trace
                .into_iter()
                .map(|poly| poly.interpolate_with_twiddles(&twiddles))
        })
        .collect_vec();
    let n_interaction_traces = interaction_trace_polys.len();
    if n_interaction_traces > 0 {
        commitment_scheme.commit(interaction_trace_polys, channel, &twiddles);
    }

    // Evaluate and commit on composition polynomial.
    let random_coeff = channel.draw_felt();

    let span = span!(Level::INFO, "Composition generation").entered();
    let composition_polynomial_poly = air.compute_composition_polynomial(
        random_coeff,
        &air.component_traces(&commitment_scheme.trees),
        &interaction_elements,
    );
    span.exit();

    let span = span!(Level::INFO, "Composition commitment").entered();
    commitment_scheme.commit(composition_polynomial_poly.to_vec(), channel, &twiddles);
    span.exit();

    // Draw OODS point.
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);

    // Get mask sample points relative to oods point.
    let sample_points = air.mask_points(oods_point);

    // TODO(spapini): Change when we support multiple interactions.
    // First tree - trace.
    let mut sample_points = TreeVec::new(vec![sample_points.flatten()]);
    if n_interaction_traces > 0 {
        // Second tree - interaction trace.
        sample_points.push(vec![vec![oods_point]; n_interaction_traces]);
    }
    // Third tree - composition polynomial.
    sample_points.push(vec![vec![oods_point]; 4]);

    // Prove the trace and composition OODS values, and retrieve them.
    let commitment_scheme_proof = commitment_scheme.prove_values(sample_points, channel, &twiddles);

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
            &interaction_elements,
        )
    {
        return Err(ProvingError::ConstraintsNotSatisfied);
    }

    Ok(StarkProof {
        commitments: commitment_scheme.roots(),
        commitment_scheme_proof,
    })
}

pub fn verify(
    proof: StarkProof,
    air: &impl Air,
    channel: &mut Channel,
) -> Result<(), VerificationError> {
    // Read trace commitment.
    let mut commitment_scheme = CommitmentSchemeVerifier::new();
    commitment_scheme.commit(proof.commitments[0], air.column_log_sizes(), channel);
    let interaction_elements = air.interaction_elements(channel);

    if !interaction_elements
        .iter()
        .flatten()
        .collect_vec()
        .is_empty()
    {
        commitment_scheme.commit(
            proof.commitments[1],
            air.column_log_sizes()[..1].to_vec(),
            channel,
        );
    }

    let random_coeff = channel.draw_felt();

    // Read composition polynomial commitment.
    commitment_scheme.commit(
        *proof.commitments.last().unwrap(),
        vec![air.composition_log_degree_bound(); 4],
        channel,
    );

    // Draw OODS point.
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);

    // Get mask sample points relative to oods point.
    let trace_sample_points = air.mask_points(oods_point);

    // TODO(spapini): Change when we support multiple interactions.
    // First tree - trace.
    let mut sample_points = TreeVec::new(vec![trace_sample_points.flatten()]);
    if !interaction_elements
        .iter()
        .flatten()
        .collect_vec()
        .is_empty()
    {
        // Second tree - interaction trace.
        sample_points.push(vec![vec![oods_point]; 1]);
    }
    // Third tree - composition polynomial.
    sample_points.push(vec![vec![oods_point]; 4]);

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
            &interaction_elements,
        )
    {
        return Err(VerificationError::OodsNotMatching);
    }

    commitment_scheme.verify_values(sample_points, proof.commitment_scheme_proof, channel)
}

/// Structures the tree-wise sampled values into component-wise OODS values and a composition
/// polynomial OODS value.
fn sampled_values_to_mask(
    air: &impl Air,
    sampled_values: &TreeVec<ColumnVec<Vec<SecureField>>>,
) -> Result<(ComponentVec<Vec<SecureField>>, SecureField), InvalidOodsSampleStructure> {
    // Retrieve sampled mask values for each component.
    let flat_trace_values = &mut sampled_values
        .first()
        .ok_or(InvalidOodsSampleStructure)?
        .iter();
    let mut trace_oods_values = vec![];
    air.components().iter().for_each(|c| {
        trace_oods_values.push(
            flat_trace_values
                .take(c.mask_points(CirclePoint::zero()).len())
                .cloned()
                .collect_vec(),
        )
    });

    if sampled_values.len() > 2 {
        let interaction_values = &mut sampled_values
            .get(1)
            .ok_or(InvalidOodsSampleStructure)?
            .iter();

        air.components()
            .iter()
            .zip_eq(&mut trace_oods_values)
            .for_each(|(_component, values)| {
                // TODO(AlonH): Implement n_interaction_columns() for component.
                values.extend(interaction_values.take(1).cloned().collect_vec())
            });
    }

    let composition_partial_sampled_values =
        sampled_values.last().ok_or(InvalidOodsSampleStructure)?;
    let composition_oods_value = SecureCirclePoly::<CPUBackend>::eval_from_partial_evals(
        composition_partial_sampled_values
            .iter()
            .flatten()
            .cloned()
            .collect_vec()
            .try_into()
            .map_err(|_| InvalidOodsSampleStructure)?,
    );

    Ok((ComponentVec(trace_oods_values), composition_oods_value))
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
    use crate::core::backend::cpu::CPUCircleEvaluation;
    use crate::core::backend::CPUBackend;
    use crate::core::channel::Blake2sChannel;
    use crate::core::circle::{CirclePoint, CirclePointIndex, Coset};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::poly::circle::{
        CanonicCoset, CircleDomain, CircleEvaluation, MAX_CIRCLE_DOMAIN_LOG_SIZE,
    };
    use crate::core::poly::BitReversedOrder;
    use crate::core::prover::{prove, ProvingError};
    use crate::core::test_utils::test_channel;
    use crate::core::ColumnVec;
    use crate::qm31;

    struct TestAir<C: ComponentProver<CPUBackend>> {
        component: C,
    }

    impl Air for TestAir<TestComponent> {
        fn components(&self) -> Vec<&dyn Component> {
            vec![&self.component]
        }
    }

    impl AirProver<CPUBackend> for TestAir<TestComponent> {
        fn prover_components(&self) -> Vec<&dyn ComponentProver<CPUBackend>> {
            vec![&self.component]
        }
    }

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

        fn trace_log_degree_bounds(&self) -> Vec<u32> {
            vec![self.log_size]
        }

        fn mask_points(
            &self,
            point: CirclePoint<SecureField>,
        ) -> crate::core::ColumnVec<Vec<CirclePoint<SecureField>>> {
            vec![vec![point]]
        }

        fn evaluate_constraint_quotients_at_point(
            &self,
            _point: CirclePoint<SecureField>,
            _mask: &crate::core::ColumnVec<Vec<SecureField>>,
            evaluation_accumulator: &mut PointEvaluationAccumulator,
            _interaction_elements: &[BaseField],
        ) {
            evaluation_accumulator.accumulate(qm31!(0, 0, 0, 1))
        }

        fn interaction_elements(&self, _channel: &mut Blake2sChannel) -> Vec<BaseField> {
            vec![]
        }
    }

    impl ComponentProver<CPUBackend> for TestComponent {
        fn evaluate_constraint_quotients_on_domain(
            &self,
            _trace: &ComponentTrace<'_, CPUBackend>,
            _evaluation_accumulator: &mut DomainEvaluationAccumulator<CPUBackend>,
            _interaction_elements: &[BaseField],
        ) {
            // Does nothing.
        }

        fn interact(
            &self,
            _trace: &ColumnVec<&CircleEvaluation<CPUBackend, BaseField, BitReversedOrder>>,
            _elements: &[BaseField],
        ) -> ColumnVec<CircleEvaluation<CPUBackend, BaseField, BitReversedOrder>> {
            // Returns empty trace.
            vec![]
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
        let trace = vec![CPUCircleEvaluation::new(domain, values)];

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
        let trace = vec![CPUCircleEvaluation::new(domain, values)];

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
        let trace = vec![CPUCircleEvaluation::new(domain, values)];

        let proof = prove(&air, &mut test_channel(), trace).unwrap_err();
        assert!(matches!(proof, ProvingError::ConstraintsNotSatisfied));
    }
}
