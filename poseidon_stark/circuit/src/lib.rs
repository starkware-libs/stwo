//! AIR for Poseidon2 hash function from <https://eprint.iacr.org/2023/323.pdf>.
//!
//! Poseidon2 is a cryptographic hash function used in Starknet.
//! This implements the STARK constraints for proving Poseidon hash computations.

use std::ops::{Add, AddAssign, Mul, Sub};

use itertools::Itertools;
use num_traits::One;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::{
    relation, EvalAtRow, FrameworkComponent, FrameworkEval, LogupTraceGenerator, Relation,
    RelationEntry,
};
use tracing::{span, Level};

pub const N_LOG_INSTANCES_PER_ROW: usize = 3; // (16 inputs + 100+ intermediate + 16 output) * 8 - kolumny
pub const N_INSTANCES_PER_ROW: usize = 1 << N_LOG_INSTANCES_PER_ROW;
pub const N_STATE: usize = 16;
const N_PARTIAL_ROUNDS: usize = 14;
const N_HALF_FULL_ROUNDS: usize = 4;
const FULL_ROUNDS: usize = 2 * N_HALF_FULL_ROUNDS;
const N_COLUMNS_PER_REP: usize = N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS;
pub const N_COLUMNS: usize = N_INSTANCES_PER_ROW * N_COLUMNS_PER_REP;
const LOG_EXPAND: u32 = 2;

// TODO(shahars): Use poseidon's real constants.
const EXTERNAL_ROUND_CONSTS: [[BaseField; N_STATE]; 2 * N_HALF_FULL_ROUNDS] =
    [[BaseField::from_u32_unchecked(1234); N_STATE]; 2 * N_HALF_FULL_ROUNDS];
const INTERNAL_ROUND_CONSTS: [BaseField; N_PARTIAL_ROUNDS] =
    [BaseField::from_u32_unchecked(1234); N_PARTIAL_ROUNDS];

pub type PoseidonComponent = FrameworkComponent<PoseidonEval>;

relation!(PoseidonElements, N_STATE);

#[derive(Clone)]
pub struct PoseidonEval {
    pub log_n_rows: u32,
    pub lookup_elements: PoseidonElements,
    pub claimed_sum: SecureField,
}

impl FrameworkEval for PoseidonEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + LOG_EXPAND
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        eval_poseidon_constraints(&mut eval, &self.lookup_elements);
        eval
    }
}

#[inline(always)]
/// Applies the M4 MDS matrix described in <https://eprint.iacr.org/2023/323.pdf> 5.1.
fn apply_m4<F>(x: [F; 4]) -> [F; 4]
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    let t0 = x[0].clone() + x[1].clone();
    let t02 = t0.clone() + t0.clone();
    let t1 = x[2].clone() + x[3].clone();
    let t12 = t1.clone() + t1.clone();
    let t2 = x[1].clone() + x[1].clone() + t1.clone();
    let t3 = x[3].clone() + x[3].clone() + t0.clone();
    let t4 = t12.clone() + t12.clone() + t3.clone();
    let t5 = t02.clone() + t02.clone() + t2.clone();
    let t6 = t3.clone() + t5.clone();
    let t7 = t2.clone() + t4.clone();
    [t6, t5, t7, t4]
}

/// Applies the external round matrix.
/// See <https://eprint.iacr.org/2023/323.pdf> 5.1 and Appendix B.
fn apply_external_round_matrix<F>(state: &mut [F; 16])
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    // Applies circ(2M4, M4, M4, M4).
    for i in 0..4 {
        [
            state[4 * i],
            state[4 * i + 1],
            state[4 * i + 2],
            state[4 * i + 3],
        ] = apply_m4([
            state[4 * i].clone(),
            state[4 * i + 1].clone(),
            state[4 * i + 2].clone(),
            state[4 * i + 3].clone(),
        ]);
    }
    for j in 0..4 {
        let s =
            state[j].clone() + state[j + 4].clone() + state[j + 8].clone() + state[j + 12].clone();
        for i in 0..4 {
            state[4 * i + j] += s.clone();
        }
    }
}

// Applies the internal round matrix.
//   mu_i = 2^{i+1} + 1.
// See <https://eprint.iacr.org/2023/323.pdf> 5.2.
fn apply_internal_round_matrix<F>(state: &mut [F; 16])
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    let sum = state[1..]
        .iter()
        .cloned()
        .fold(state[0].clone(), |acc, s| acc + s);
    state.iter_mut().enumerate().for_each(|(i, s)| {
        *s = s.clone() * BaseField::from_u32_unchecked(1 << (i + 1)) + sum.clone();
    });
}

fn pow5<F: FieldExpOps>(x: F) -> F {
    let x2 = x.clone() * x.clone();
    let x4 = x2.clone() * x2.clone();
    x4 * x.clone()
}

pub fn eval_poseidon_constraints<E: EvalAtRow>(eval: &mut E, lookup_elements: &PoseidonElements) {
    for _ in 0..N_INSTANCES_PER_ROW {
        let mut state: [_; N_STATE] = std::array::from_fn(|_| eval.next_trace_mask());
        //  vec[u32;16]
        let initial_state = state.clone();

        // 4 full rounds.
        (0..N_HALF_FULL_ROUNDS).for_each(|round| {
            (0..N_STATE).for_each(|i| {
                state[i] += EXTERNAL_ROUND_CONSTS[round][i];
            });
            apply_external_round_matrix(&mut state);
            state = std::array::from_fn(|i| pow5(state[i].clone()));
            state.iter_mut().for_each(|s| {
                let m = eval.next_trace_mask();
                eval.add_constraint(s.clone() - m.clone());
                *s = m;
            });
        });

        // Partial rounds.
        (0..N_PARTIAL_ROUNDS).for_each(|round| {
            state[0] += INTERNAL_ROUND_CONSTS[round];
            apply_internal_round_matrix(&mut state);
            state[0] = pow5(state[0].clone());
            let m = eval.next_trace_mask();
            eval.add_constraint(state[0].clone() - m.clone());
            state[0] = m;
        });

        // 4 full rounds.
        (0..N_HALF_FULL_ROUNDS).for_each(|round| {
            (0..N_STATE).for_each(|i| {
                state[i] += EXTERNAL_ROUND_CONSTS[round + N_HALF_FULL_ROUNDS][i];
            });
            apply_external_round_matrix(&mut state);
            state = std::array::from_fn(|i| pow5(state[i].clone()));
            state.iter_mut().for_each(|s| {
                let m = eval.next_trace_mask();
                eval.add_constraint(s.clone() - m.clone());
                *s = m;
            });
        });

        // Provide state lookups.
        eval.add_to_relation(RelationEntry::new(
            lookup_elements,
            E::EF::one(),
            &initial_state,
        ));
        eval.add_to_relation(RelationEntry::new(lookup_elements, -E::EF::one(), &state));
    }

    eval.finalize_logup_in_pairs();
}

pub struct LookupData {
    pub initial_state: [[BaseColumn; N_STATE]; N_INSTANCES_PER_ROW],
    pub final_state: [[BaseColumn; N_STATE]; N_INSTANCES_PER_ROW],
}

/// Generuje trace dla Poseidon2 hash
///
/// Dla każdego hash instance (8 per wiersz) tworzy 158 kolumn:
/// - Kolumny 0-15:    Initial state (16 elementów)
/// - Kolumny 16-79:   Po pierwszych 4 full rounds (4×16 = 64 elementy)
/// - Kolumny 80-93:   Po 14 partial rounds (14×1 = 14 elementów, tylko state[0])
/// - Kolumny 94-157:  Po ostatnich 4 full rounds (4×16 = 64 elementy)
///
/// Total: 158 kolumn × 8 instancji = 1264 kolumny
pub fn gen_trace(
    log_size: u32,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    LookupData,
) {
    let _span = span!(Level::INFO, "Generation").entered();
    assert!(log_size >= LOG_N_LANES);
    // Alokuj wszystkie kolumny trace (1264 dla 8 instancji × 158 kolumn każda)
    let mut trace = (0..N_COLUMNS)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();
    let mut lookup_data = LookupData {
        initial_state: std::array::from_fn(|_| {
            std::array::from_fn(|_| BaseColumn::zeros(1 << log_size))
        }),
        final_state: std::array::from_fn(|_| {
            std::array::from_fn(|_| BaseColumn::zeros(1 << log_size))
        }),
    };

    for vec_index in 0..(1 << (log_size - LOG_N_LANES)) {
        let mut col_index = 0; // ⭐ WSKAŹNIK KOLUMNY - śledzi którą kolumnę trace wypełniamy
        for rep_i in 0..N_INSTANCES_PER_ROW {
            // ========================================
            // 1️⃣ INITIAL STATE (kolumny 0-15)
            // ========================================
            let mut state: [_; N_STATE] = std::array::from_fn(|state_i| {
                PackedBaseField::from_array(std::array::from_fn(|i| {
                    BaseField::from_u32_unchecked((vec_index * 16 + i + state_i + rep_i) as u32)
                }))
            });
            // ⭐⭐⭐ ZAPISZ INITIAL STATE DO TRACE (kolumny 0-15) ⭐⭐⭐
            state.iter().copied().for_each(|s| {
                trace[col_index].data[vec_index] = s; // Zapisujemy do kolejnych kolumn trace
                col_index += 1; // col_index: 0→1→2...→15
            });
            // Teraz col_index = 16

            // Kopia initial state do lookup_data (dla LogUp)
            lookup_data.initial_state[rep_i]
                .iter_mut()
                .zip(state)
                .for_each(|(res, state_i)| res.data[vec_index] = state_i);

            // ========================================
            // 2️⃣ PIERWSZE 4 FULL ROUNDS (kolumny 16-79)
            // ========================================
            (0..N_HALF_FULL_ROUNDS).for_each(|round| {
                // round = 0,1,2,3
                // Krok 1: Dodaj stałe rundy
                (0..N_STATE).for_each(|i| {
                    state[i] += PackedBaseField::broadcast(EXTERNAL_ROUND_CONSTS[round][i]);
                });
                // Krok 2: MDS matrix (miksowanie wszystkich elementów)
                apply_external_round_matrix(&mut state);
                // Krok 3: S-box (x^5 dla wszystkich 16 elementów)
                state = std::array::from_fn(|i| pow5(state[i]));
                // ⭐⭐⭐ ZAPISZ STAN PO RUNDZIE DO TRACE ⭐⭐⭐
                state.iter().copied().for_each(|s| {
                    trace[col_index].data[vec_index] = s;
                    col_index += 1;
                });
                // Po rundzie 0: col_index = 32 (kolumny 16-31)
                // Po rundzie 1: col_index = 48 (kolumny 32-47)
                // Po rundzie 2: col_index = 64 (kolumny 48-63)
                // Po rundzie 3: col_index = 80 (kolumny 64-79)
            });

            // ========================================
            // 3️⃣ 14 PARTIAL ROUNDS (kolumny 80-93)
            // ========================================
            (0..N_PARTIAL_ROUNDS).for_each(|round| {
                // round = 0..13
                // Krok 1: Dodaj stałą (TYLKO do state[0])
                state[0] += PackedBaseField::broadcast(INTERNAL_ROUND_CONSTS[round]);
                // Krok 2: MDS matrix (dla WSZYSTKICH elementów mimo wszystko)
                apply_internal_round_matrix(&mut state);
                // Krok 3: S-box TYLKO dla state[0] (OSZCZĘDNOŚĆ! Tylko 1 zamiast 16)
                state[0] = pow5(state[0]);
                // ⭐⭐⭐ ZAPISZ TYLKO state[0] DO TRACE ⭐⭐⭐
                trace[col_index].data[vec_index] = state[0]; // Tylko pierwszy element!
                col_index += 1; // col_index: 80→81→82...→93
            });
            // Teraz col_index = 94

            // ========================================
            // 4️⃣ OSTATNIE 4 FULL ROUNDS (kolumny 94-157)
            // ========================================
            (0..N_HALF_FULL_ROUNDS).for_each(|round| {
                // round = 0,1,2,3
                // Krok 1: Dodaj stałe (z offsetem +4, bo to rundy 4,5,6,7)
                (0..N_STATE).for_each(|i| {
                    state[i] += PackedBaseField::broadcast(
                        EXTERNAL_ROUND_CONSTS[round + N_HALF_FULL_ROUNDS][i],
                    );
                });
                // Krok 2: MDS matrix
                apply_external_round_matrix(&mut state);
                // Krok 3: S-box (x^5 dla wszystkich)
                state = std::array::from_fn(|i| pow5(state[i]));
                // ⭐⭐⭐ ZAPISZ STAN PO RUNDZIE DO TRACE ⭐⭐⭐
                state.iter().copied().for_each(|s| {
                    trace[col_index].data[vec_index] = s;
                    col_index += 1;
                });
                // Po rundzie 0 (=round 5): col_index = 110 (kolumny 94-109)
                // Po rundzie 1 (=round 6): col_index = 126 (kolumny 110-125)
                // Po rundzie 2 (=round 7): col_index = 142 (kolumny 126-141)
                // Po rundzie 3 (=round 8): col_index = 158 (kolumny 142-157) ← FINAL STATE!
            });

            // ========================================
            // 5️⃣ FINAL STATE (zapisz do lookup_data)
            // ========================================
            // state teraz zawiera FINAL STATE (16 elementów po wszystkich rundach)
            lookup_data.final_state[rep_i]
                .iter_mut()
                .zip(state)
                .for_each(|(res, state_i)| res.data[vec_index] = state_i);
            // Ostatnie 16 elementów trace (kolumny 142-157) to FINAL STATE = OUTPUT HASHA!
        }
    }
    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace = trace
        .into_iter()
        .map(|eval| CircleEvaluation::new(domain, eval))
        .collect();
    (trace, lookup_data)
}

pub fn gen_interaction_trace(
    log_size: u32,
    lookup_data: LookupData,
    lookup_elements: &PoseidonElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let _span = span!(Level::INFO, "Generate interaction trace").entered();
    let mut logup_gen = unsafe { LogupTraceGenerator::uninitialized(log_size) };

    #[allow(clippy::needless_range_loop)]
    for rep_i in 0..N_INSTANCES_PER_ROW {
        let frac_at_row = |vec_row: usize| {
            let denom0: PackedSecureField = lookup_elements.combine(
                &lookup_data.initial_state[rep_i]
                    .each_ref()
                    .map(|s| s.data[vec_row]),
            );
            let denom1: PackedSecureField = lookup_elements.combine(
                &lookup_data.final_state[rep_i]
                    .each_ref()
                    .map(|s| s.data[vec_row]),
            );
            (denom1 - denom0, denom0 * denom1)
        };
        let range = 0..1 << (log_size - LOG_N_LANES);
        logup_gen.col_from_iter(range.map(frac_at_row));
    }

    logup_gen.finalize_last()
}
