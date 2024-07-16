use itertools::Itertools;
use num_traits::Zero;
use tracing::{span, Level};

use super::DomainEvalHelper;
use crate::constraint_framework::logup::{LogupAtRow, LogupTraceGenerator, LookupElements};
use crate::constraint_framework::{DomainEvaluator, EvalAtRow, InfoEvaluator, PointEvaluator};
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentProver, ComponentTrace};
use crate::core::backend::simd::column::BaseFieldVec;
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::circle::CirclePoint;
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::{ColumnVec, InteractionElements, LookupValues};

pub struct TableComponent {
    pub log_size: u32,
    pub n_cols: usize,
    pub n_reps: usize,
    pub lookup_elements: LookupElements,
    pub claimed_sum: SecureField,
}

fn table_info(n_cols: usize, n_reps: usize) -> InfoEvaluator {
    let mut counter = TableEval {
        eval: InfoEvaluator::default(),
        n_cols,
        n_reps,
        lookup_elements: &LookupElements::dummy(n_cols),
        logup: LogupAtRow::new(1, SecureField::zero(), BaseField::zero()),
    };
    // Constant is_first column.
    counter.eval.next_interaction_mask(2, [0]);
    counter.eval()
}

impl Component for TableComponent {
    fn n_constraints(&self) -> usize {
        table_info(self.n_cols, self.n_reps).n_constraints
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    fn n_interaction_phases(&self) -> u32 {
        1
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::new(
            table_info(self.n_cols, self.n_reps)
                .mask_offsets
                .iter()
                .map(|tree_masks| vec![self.log_size; tree_masks.len()])
                .collect(),
        )
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let info = table_info(self.n_cols, self.n_reps);
        let trace_step = CanonicCoset::new(self.log_size).step();
        info.mask_offsets.map(|tree_mask| {
            tree_mask
                .iter()
                .map(|col_mask| {
                    col_mask
                        .iter()
                        .map(|off| point + trace_step.mul_signed(*off).into_ef())
                        .collect()
                })
                .collect()
        })
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        let constraint_zero_domain = CanonicCoset::new(self.log_size).coset;
        let denom = coset_vanishing(constraint_zero_domain, point);
        let denom_inverse = denom.inverse();
        let mut eval = PointEvaluator::new(mask.as_ref(), evaluation_accumulator, denom_inverse);
        let [is_first] = eval.next_interaction_mask(2, [0]);
        let blake_eval = TableEval {
            eval,
            n_cols: self.n_cols,
            n_reps: self.n_reps,
            lookup_elements: &self.lookup_elements,
            logup: LogupAtRow::new(1, self.claimed_sum, is_first),
        };
        blake_eval.eval();
    }
}

impl ComponentProver<SimdBackend> for TableComponent {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, SimdBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<SimdBackend>,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        println!("Trace: {:?}", trace.evals.as_ref().map(|x| x.len()));
        println!(
            "Expected: {:?}",
            self.trace_log_degree_bounds().map(|x| x.len())
        );
        let mut domain_eval = DomainEvalHelper::new(
            self.log_size,
            self.log_size + 1,
            trace,
            evaluation_accumulator,
            self.max_constraint_log_degree_bound(),
            self.n_constraints(),
        );

        // TODO:
        let _span = span!(Level::INFO, "Constraint pointwise eval").entered();
        for vec_row in 0..(1 << (domain_eval.eval_domain.log_size() - LOG_N_LANES)) {
            let mut eval = DomainEvaluator::new(
                &domain_eval.trace.evals,
                vec_row,
                &domain_eval.accum.random_coeff_powers,
                domain_eval.trace_domain.log_size,
                domain_eval.eval_domain.log_size(),
            );
            // Constant column is_first.
            let [is_first] = eval.next_interaction_mask(2, [0]);
            let logup = LogupAtRow::new(1, self.claimed_sum, is_first);
            let table_eval = TableEval {
                eval,
                n_cols: self.n_cols,
                n_reps: self.n_reps,
                lookup_elements: &self.lookup_elements,
                logup,
            };
            let eval = table_eval.eval();
            domain_eval.finalize_row(vec_row, eval.row_res);
        }
    }

    fn lookup_values(
        &self,
        _trace: &crate::core::air::ComponentTrace<'_, SimdBackend>,
    ) -> LookupValues {
        LookupValues::default()
    }
}

pub struct TableEval<'a, E: EvalAtRow> {
    pub eval: E,
    pub n_cols: usize,
    pub n_reps: usize,
    pub lookup_elements: &'a LookupElements,
    pub logup: LogupAtRow<2, E>,
}
impl<'a, E: EvalAtRow> TableEval<'a, E> {
    pub fn eval(mut self) -> E {
        for _ in 0..self.n_reps {
            let multiplicity = -self.eval.next_trace_mask();
            let cols = (0..self.n_cols)
                .map(|_| self.eval.next_interaction_mask(2, [0])[0])
                .collect_vec();

            self.logup.push_lookup(
                &mut self.eval,
                multiplicity.into(),
                &cols,
                self.lookup_elements,
            );
        }
        self.logup.finalize(&mut self.eval);

        self.eval
    }
}

pub struct TableLookupData {
    pub inputs: Vec<Vec<BaseFieldVec>>,
    pub mults: Vec<BaseFieldVec>,
}

pub fn generate_trace(
    log_size: u32,
    inputs: Vec<BaseFieldVec>,
    mults: BaseFieldVec,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    TableLookupData,
) {
    let n_reps = mults.len() >> log_size;
    assert_eq!(mults.len(), n_reps << log_size);
    // Split input to `n_reps` chunks.
    let mut inputs = inputs
        .into_iter()
        .map(|input| {
            input
                .data
                .chunks(1 << (log_size - LOG_N_LANES) as usize)
                .map(|chunk| BaseFieldVec {
                    data: chunk.to_vec(),
                    length: chunk.len() << LOG_N_LANES,
                })
                .collect_vec()
                .into_iter()
        })
        .collect_vec();

    // Transpose inputs.
    let inputs = (0..n_reps)
        .map(|_| inputs.iter_mut().map(|it| it.next().unwrap()).collect_vec())
        .collect_vec();

    // Split mults to `n_reps` chunks.
    let mults = mults
        .data
        .chunks(1 << (log_size - LOG_N_LANES) as usize)
        .map(|chunk| BaseFieldVec::new(chunk.to_vec()))
        .collect_vec();

    (
        mults
            .iter()
            .map(|mult| {
                CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), mult.clone())
            })
            .collect_vec(),
        TableLookupData { inputs, mults },
    )
}

pub fn gen_interaction_trace(
    log_size: u32,
    lookup_data: TableLookupData,
    lookup_elements: &LookupElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    println!(
        "xor lookup_data: {:?}x{:?}",
        lookup_data.inputs.len(),
        lookup_data.inputs[0].len()
    );
    let _span = span!(Level::INFO, "Generate table trace").entered();
    let mut logup_gen = LogupTraceGenerator::new(log_size);
    for [(inputs0, mults0), (inputs1, mults1)] in lookup_data
        .inputs
        .iter()
        .zip(&lookup_data.mults)
        .array_chunks::<2>()
    {
        let mut col_gen = logup_gen.new_col();

        #[allow(clippy::needless_range_loop)]
        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            // TODO(spapini): Get rid of the allocation here, by preallocating a buffer.
            //   Or getting an iterator in comine.
            let p0: PackedSecureField =
                lookup_elements.combine(&inputs0.iter().map(|l| l.data[vec_row]).collect_vec());
            let p1: PackedSecureField =
                lookup_elements.combine(&inputs1.iter().map(|l| l.data[vec_row]).collect_vec());
            let num = p1 * mults0.data[vec_row] + p0 * mults1.data[vec_row];
            let denom = p0 * p1;
            col_gen.write_frac(vec_row, -num, denom);
        }
        col_gen.finalize_col();
    }

    logup_gen.finalize()
}
