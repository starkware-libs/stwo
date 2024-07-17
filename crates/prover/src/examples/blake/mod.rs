use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Sub};
use std::simd::u32x16;

use bytemuck::cast_slice_mut;
use itertools::{chain, multizip, Itertools};
use num_traits::Zero;
use round::BlakeRoundComponent;
use round_gen::BlakeRoundInput;
use scheduler::BlakeSchedulerComponent;
use scheduler_gen::BlakeInput;
use table::TableComponent;
use tracing::{span, Level};

use crate::constraint_framework::constant_cols::gen_is_first;
use crate::constraint_framework::logup::LookupElements;
use crate::core::air::accumulation::{ColumnAccumulator, DomainEvaluationAccumulator};
use crate::core::air::{Air, AirProver, Component, ComponentProver, ComponentTrace};
use crate::core::backend::simd::column::BaseFieldVec;
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES, N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Column, ColumnOps};
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::circle::Coset;
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{FieldExpOps, FieldOps, IntoSlice};
use crate::core::pcs::CommitmentSchemeProver;
use crate::core::poly::circle::{CanonicCoset, CircleDomain, CircleEvaluation, PolyOps};
use crate::core::prover::{prove_without_commit, StarkProof, LOG_BLOWUP_FACTOR};
use crate::core::vcs::blake2_hash::Blake2sHasher;
use crate::core::vcs::blake2s_ref;
use crate::core::vcs::hasher::Hasher;
use crate::core::InteractionElements;

// Blake3.
pub const N_ROUNDS: usize = 7;

mod round;
mod round_constraints;
mod round_gen;
mod scheduler;
mod scheduler_constraints;
mod scheduler_gen;
mod table;

#[derive(Clone, Copy, Debug)]
struct Fu32<F>
where
    F: FieldExpOps
        + Copy
        + Debug
        + AddAssign<F>
        + Add<F, Output = F>
        + Sub<F, Output = F>
        + Mul<BaseField, Output = F>,
{
    l: F,
    h: F,
}
impl<F> Fu32<F>
where
    F: FieldExpOps
        + Copy
        + Debug
        + AddAssign<F>
        + Add<F, Output = F>
        + Sub<F, Output = F>
        + Mul<BaseField, Output = F>,
{
    fn to_felts(self) -> [F; 2] {
        [self.l, self.h]
    }
}

pub struct BlakeAir {
    pub scheduler_component: BlakeSchedulerComponent,
    pub round_component: BlakeRoundComponent,
    pub xor_component: TableComponent,
}

impl Air for BlakeAir {
    fn components(&self) -> Vec<&dyn Component> {
        vec![
            &self.scheduler_component,
            &self.round_component,
            &self.xor_component,
        ]
    }
}

impl AirProver<SimdBackend> for BlakeAir {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<SimdBackend>> {
        vec![
            &self.scheduler_component,
            &self.round_component,
            &self.xor_component,
        ]
    }
}

fn to_felts(x: u32x16) -> [PackedBaseField; 2] {
    [
        unsafe { PackedBaseField::from_simd_unchecked(x & u32x16::splat(0xffff)) },
        unsafe { PackedBaseField::from_simd_unchecked(x >> 16) },
    ]
}

#[allow(unused)]
pub fn prove_blake(log_size: u32) -> (BlakeAir, StarkProof) {
    const XOR_LOG_SIZE: u32 = 20;
    assert!(log_size >= LOG_N_LANES);

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let log_max_rows = (log_size + 3).max(XOR_LOG_SIZE);
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_max_rows + 1 + LOG_BLOWUP_FACTOR)
            .circle_domain()
            .half_coset,
    );
    span.exit();

    // Prepare inputs.
    let blake_inputs = (0..(1 << (log_size - LOG_N_LANES)))
        .map(|i| {
            let v = [u32x16::from_array(std::array::from_fn(|j| (i + 2 * j) as u32)); 16];
            let m = [u32x16::from_array(std::array::from_fn(|j| (i + 2 * j + 1) as u32)); 16];
            BlakeInput { v, m }
        })
        .collect::<Vec<_>>();

    // Setup protocol.
    let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
    let commitment_scheme = &mut CommitmentSchemeProver::new(LOG_BLOWUP_FACTOR);

    // Scheulder trace.
    let span = span!(Level::INFO, "Scheduler Generation").entered();
    let (scheduler_trace, scheduler_lookup_data, round_inputs) =
        scheduler_gen::gen_trace(log_size, &blake_inputs);

    // Round trace.
    span.exit();
    let span = span!(Level::INFO, "Round Generation").entered();
    let mut xor_mults = BaseFieldVec::zeros(1 << 24);
    let (round_trace, round_lookup_data) = round_gen::gen_trace(
        log_size + 3,
        &round_inputs,
        cast_slice_mut(&mut xor_mults.data[..]),
    );
    let n_padded_rounds = (1 << (log_size + 3 - LOG_N_LANES)) - round_trace.len();
    let round_trace0 = round_trace.clone();

    // Xor table trace.
    let a_col: BaseFieldVec = (0..(1 << 24))
        .map(|i| BaseField::from_u32_unchecked((i >> 12) as u32))
        .collect();
    let b_col: BaseFieldVec = (0..(1 << 24))
        .map(|i| BaseField::from_u32_unchecked((i & 0xfff) as u32))
        .collect();
    let c_col: BaseFieldVec = (0..(1 << 24))
        .map(|i| BaseField::from_u32_unchecked(((i >> 12) ^ (i & 0xfff)) as u32))
        .collect();
    let (xor_trace, xor_lookup_data) = table::generate_trace(
        XOR_LOG_SIZE,
        vec![a_col.clone(), b_col.clone(), c_col.clone()],
        xor_mults,
    );

    span.exit();
    let span = span!(Level::INFO, "Trace Commitment").entered();
    commitment_scheme.commit_on_evals(
        chain![scheduler_trace, round_trace, xor_trace,].collect_vec(),
        channel,
        &twiddles,
    );

    // Draw lookup element.
    let blake_lookup_elements = LookupElements::draw(channel, 2 * 16 * 3);
    let round_lookup_elements = LookupElements::draw(channel, 2 * 16 * 3);
    let xor_lookup_elements = LookupElements::draw(channel, 3);

    // Interaction trace.
    span.exit();
    let span = span!(Level::INFO, "Scheduler Interaction Generation").entered();
    let (scheduler_trace, scheduler_claimed_sum) = scheduler_gen::gen_interaction_trace(
        log_size,
        scheduler_lookup_data,
        &round_lookup_elements,
        &blake_lookup_elements,
    );

    span.exit();
    let span = span!(Level::INFO, "Round Interaction Generation").entered();
    let (round_trace, round_claimed_sum) = round_gen::gen_interaction_trace(
        log_size + 3,
        round_lookup_data,
        &xor_lookup_elements,
        &round_lookup_elements,
    );
    let round_trace1 = round_trace.clone();

    span.exit();
    let span = span!(Level::INFO, "Table Interaction Generation").entered();
    let (xor_trace, xor_claimed_sum) =
        table::gen_interaction_trace(XOR_LOG_SIZE, xor_lookup_data, &xor_lookup_elements);
    let round_trace1 = round_trace.clone();

    span.exit();
    let span = span!(Level::INFO, "Interaction Commitment").entered();
    commitment_scheme.commit_on_evals(
        chain![scheduler_trace, round_trace, xor_trace,].collect_vec(),
        channel,
        &twiddles,
    );

    // Constant trace.
    span.exit();
    let span = span!(Level::INFO, "Constant Trace Generation").entered();
    commitment_scheme.commit_on_evals(
        chain![
            [
                gen_is_first(log_size),
                gen_is_first(log_size + 3),
                gen_is_first(XOR_LOG_SIZE),
            ],
            multizip((
                a_col.data.chunks(1 << (XOR_LOG_SIZE - LOG_N_LANES)),
                b_col.data.chunks(1 << (XOR_LOG_SIZE - LOG_N_LANES)),
                c_col.data.chunks(1 << (XOR_LOG_SIZE - LOG_N_LANES))
            ))
            .flat_map(|(a, b, c)| {
                [
                    CircleEvaluation::new(
                        CanonicCoset::new(XOR_LOG_SIZE).circle_domain(),
                        BaseFieldVec::new(a.to_vec()),
                    ),
                    CircleEvaluation::new(
                        CanonicCoset::new(XOR_LOG_SIZE).circle_domain(),
                        BaseFieldVec::new(b.to_vec()),
                    ),
                    CircleEvaluation::new(
                        CanonicCoset::new(XOR_LOG_SIZE).circle_domain(),
                        BaseFieldVec::new(c.to_vec()),
                    ),
                ]
            })
        ]
        .collect_vec(),
        channel,
        &twiddles,
    );
    span.exit();

    // // Sanity check.
    // let scheduler_traces = TreeVec::new(vec![
    //     scheduler_trace0,
    //     scheduler_trace1,
    //     vec![gen_is_first(log_size)],
    // ]);
    // let scheduler_trace_polys =
    //     scheduler_traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

    // assert_constraints(
    //     &scheduler_trace_polys,
    //     CanonicCoset::new(log_size),
    //     |mut eval| {
    //         let [is_first] = eval.next_interaction_mask(2, [0]);
    //         BlakeSchedulerEval {
    //             eval,
    //             blake_lookup_elements,
    //             round_lookup_elements,
    //             logup: LogupAtRow::new(1, scheduler_claimed_sum, is_first),
    //         }
    //         .eval();
    //     },
    // );

    // let round_traces = TreeVec::new(vec![
    //     round_trace0,
    //     round_trace1,
    //     vec![gen_is_first(log_size + 3)],
    // ]);
    // let round_trace_polys =
    //     round_traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

    // assert_constraints(
    //     &round_trace_polys,
    //     CanonicCoset::new(log_size + 3),
    //     |mut eval| {
    //         let [is_first] = eval.next_interaction_mask(2, [0]);
    //         BlakeRoundEval {
    //             eval,
    //             xor_lookup_elements,
    //             logup: LogupAtRow::new(1, round_claimed_sum, is_first),
    //         }
    //         .eval();
    //     },
    // );

    // Check claimed sum.
    let to_felts = |x: u32| {
        [
            BaseField::from_u32_unchecked(x & 0xffff),
            BaseField::from_u32_unchecked(x >> 16),
        ]
    };
    println!("round_claimed_sum:{round_claimed_sum}");
    println!("xor_claimed_sum:{xor_claimed_sum}");
    let total_claimed_sum = scheduler_claimed_sum + round_claimed_sum + xor_claimed_sum;
    let mut expected_claimed_sum = -blake_inputs
        .iter()
        .flat_map(|inp| {
            (0..N_LANES).map(|i| {
                let v0 = inp.v.each_ref().map(|x| x[i]);
                let m = inp.m.each_ref().map(|x| x[i]);
                let mut v = v0;
                for r in 0..N_ROUNDS {
                    blake2s_ref::round(&mut v, m, r);
                }
                blake_lookup_elements
                    .combine::<BaseField, SecureField>(
                        &chain![
                            v0.into_iter().flat_map(to_felts),
                            v.into_iter().flat_map(to_felts),
                            m.into_iter().flat_map(to_felts),
                        ]
                        .collect_vec(),
                    )
                    .inverse()
            })
        })
        .fold(SecureField::zero(), |acc, x| acc + x);
    expected_claimed_sum = SecureField::zero();

    // Add round padding.
    {
        let padded_input = BlakeRoundInput::default();
        let n_padded_rounds = (1 << (log_size + 3)) - N_ROUNDS * (1 << log_size);
        let v0 = [0; 16];
        let m = [0; 16];
        let mut v = v0;
        blake2s_ref::round(&mut v, m, 0);
        expected_claimed_sum += -round_lookup_elements
            .combine::<BaseField, SecureField>(
                &chain![
                    v0.into_iter().flat_map(to_felts),
                    v.into_iter().flat_map(to_felts),
                    m.into_iter().flat_map(to_felts),
                ]
                .collect_vec(),
            )
            .inverse()
            * BaseField::from(n_padded_rounds);
    }

    assert_eq!(total_claimed_sum, expected_claimed_sum);

    // Prove constraints.
    let scheduler_component = BlakeSchedulerComponent {
        log_size,
        blake_lookup_elements,
        round_lookup_elements: round_lookup_elements.clone(),
        claimed_sum: scheduler_claimed_sum, // TODO: This is not correct.
    };
    let round_component = BlakeRoundComponent {
        log_size: log_size + 3,
        xor_lookup_elements: xor_lookup_elements.clone(),
        round_lookup_elements,
        claimed_sum: round_claimed_sum,
    };
    let xor_component = TableComponent {
        log_size: XOR_LOG_SIZE,
        n_cols: 3,
        n_reps: 1 << (24 - XOR_LOG_SIZE),
        lookup_elements: xor_lookup_elements,
        claimed_sum: xor_claimed_sum,
    };
    let air = BlakeAir {
        scheduler_component,
        round_component,
        xor_component,
    };
    let proof = prove_without_commit::<SimdBackend>(
        &air,
        channel,
        &InteractionElements::default(),
        &twiddles,
        commitment_scheme,
    )
    .unwrap();

    (air, proof)
}

// TODO(spapini): Move to a common module.
struct DomainEvalHelper<'a> {
    eval_domain: CircleDomain,
    trace_domain: Coset,
    trace: &'a ComponentTrace<'a, SimdBackend>,
    denom_inv: BaseFieldVec,
    accum: ColumnAccumulator<'a, SimdBackend>,
}
impl<'a> DomainEvalHelper<'a> {
    fn new(
        trace_log_size: u32,
        eval_log_size: u32,
        trace: &'a ComponentTrace<'a, SimdBackend>,
        evaluation_accumulator: &'a mut DomainEvaluationAccumulator<SimdBackend>,
        constraint_log_degree_bound: u32,
        n_constraints: usize,
    ) -> Self {
        assert_eq!(
            eval_log_size, constraint_log_degree_bound,
            "Extension not yet supported in generic evaluator"
        );
        let eval_domain = CanonicCoset::new(eval_log_size).circle_domain();
        let row_log_size = trace_log_size;

        // Denoms.
        let trace_domain = CanonicCoset::new(row_log_size).coset;
        let span = span!(Level::INFO, "Constraint eval denominators").entered();

        let mut denoms =
            BaseFieldVec::from_iter(eval_domain.iter().map(|p| coset_vanishing(trace_domain, p)));
        <SimdBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut denoms);
        let mut denom_inv = BaseFieldVec::zeros(denoms.len());
        <SimdBackend as FieldOps<BaseField>>::batch_inverse(&denoms, &mut denom_inv);

        span.exit();

        let [mut accum] =
            evaluation_accumulator.columns([(constraint_log_degree_bound, n_constraints)]);
        accum.random_coeff_powers.reverse();

        Self {
            eval_domain,
            trace_domain,
            trace,
            denom_inv,
            accum,
        }
    }
    fn finalize_row(&mut self, vec_row: usize, row_res: PackedSecureField) {
        unsafe {
            self.accum.col.set_packed(
                vec_row,
                self.accum.col.packed_at(vec_row) + row_res * self.denom_inv.data[vec_row],
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::env;

    use crate::constraint_framework::logup::LookupElements;
    use crate::core::air::AirExt;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::IntoSlice;
    use crate::core::pcs::CommitmentSchemeVerifier;
    use crate::core::prover::verify_without_commit;
    use crate::core::vcs::blake2_hash::Blake2sHasher;
    use crate::core::vcs::hasher::Hasher;
    use crate::core::InteractionElements;
    use crate::examples::blake::prove_blake;

    #[test_log::test]
    fn test_simd_blake_prove() {
        // Note: To see time measurement, run test with
        //   RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info RUST_BACKTRACE=1 RUSTFLAGS="
        //   -C target-cpu=native -C target-feature=+avx512f -C opt-level=3" cargo test
        //   test_simd_blake_prove -- --nocapture

        // Get from environment variable:
        let log_n_instances = env::var("LOG_N_INSTANCES")
            .unwrap_or_else(|_| "10".to_string())
            .parse::<u32>()
            .unwrap();

        // Prove.
        let (air, proof) = prove_blake(log_n_instances);

        // Verify.
        // TODO: Create Air instance independently.
        let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
        let commitment_scheme = &mut CommitmentSchemeVerifier::new();

        // Decommit.
        let sizes = air.column_log_sizes();
        // Trace columns.
        commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
        // Draw lookup element.
        let _blake_lookup_elements = LookupElements::draw(channel, 2 * 16 * 3);
        let _round_lookup_elements = LookupElements::draw(channel, 2 * 16 * 3);
        let xor_lookup_elements = LookupElements::draw(channel, 3);
        assert_eq!(xor_lookup_elements, air.round_component.xor_lookup_elements);
        // TODO(spapini): Check claimed sum against first and last instances.
        // Interaction columns.
        commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);
        // Constant columns.
        commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

        verify_without_commit(
            &air,
            channel,
            &InteractionElements::default(),
            commitment_scheme,
            proof,
        )
        .unwrap();
    }
}
