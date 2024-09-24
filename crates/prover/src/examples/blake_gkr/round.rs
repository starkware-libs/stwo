use std::array;

use itertools::{chain, Itertools};
use num_traits::One;
use tracing::{span, Level};

use super::air::InvalidClaimError;
use super::gkr_lookups::accumulation::{MleClaimAccumulator, MleCollection};
use super::gkr_lookups::AccumulatedMleCoeffColumnOracle;
use crate::constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, PointEvaluator};
use crate::core::air::accumulation::PointEvaluationAccumulator;
use crate::core::backend::simd::column::SecureColumn;
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr_prover::Layer;
use crate::core::lookups::gkr_verifier::{LogUpArtifactInstance, LookupArtifactInstance};
use crate::core::lookups::mle::Mle;
use crate::core::pcs::TreeVec;
use crate::core::ColumnVec;
use crate::examples::blake::round::{BlakeRoundLookupData, RoundElements, TraceGenerator};
use crate::examples::blake::{BlakeXorElements, Fu32, STATE_SIZE};

pub type BlakeRoundComponent = FrameworkComponent<BlakeRoundEval>;

pub struct BlakeRoundEval {
    pub log_size: u32,
    pub xor_lookup_elements: BlakeXorElements,
    pub round_lookup_elements: RoundElements,
}

impl FrameworkEval for BlakeRoundEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }
    fn evaluate<E: EvalAtRow>(&self, eval: E) -> E {
        const MLE_COEFF_COL_EVAL: bool = false;
        let blake_eval = BlakeRoundConstraintEval::<E, MLE_COEFF_COL_EVAL> {
            eval,
            xor_lookup_elements: &self.xor_lookup_elements,
            round_lookup_elements: &self.round_lookup_elements,
            mle_coeff_col_evals: None,
        };
        blake_eval.eval()
    }
}

impl AccumulatedMleCoeffColumnOracle for BlakeRoundComponent {
    fn accumulate_at_point(
        &self,
        _point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        acc: &mut PointEvaluationAccumulator,
    ) {
        // Create dummy point evaluator just to extract the value we need from the mask
        let mut _accumulator = PointEvaluationAccumulator::new(SecureField::one());
        let eval = PointEvaluator::new(
            mask.sub_tree(self.trace_locations()),
            &mut _accumulator,
            SecureField::one(),
        );

        let mut mle_coef_col_evals = Vec::new();

        const MLE_COEFF_COL_EVAL: bool = true;
        BlakeRoundConstraintEval::<_, MLE_COEFF_COL_EVAL> {
            eval,
            xor_lookup_elements: &self.xor_lookup_elements,
            round_lookup_elements: &self.round_lookup_elements,
            mle_coeff_col_evals: Some(&mut mle_coef_col_evals),
        }
        .eval();

        for eval in mle_coef_col_evals {
            acc.accumulate(eval)
        }
    }
}

const INV16: BaseField = BaseField::from_u32_unchecked(1 << 15);
const TWO: BaseField = BaseField::from_u32_unchecked(2);

pub struct BlakeRoundConstraintEval<'a, E: EvalAtRow, const MLE_COEFF_COL_EVAL: bool> {
    pub eval: E,
    pub xor_lookup_elements: &'a BlakeXorElements,
    pub round_lookup_elements: &'a RoundElements,
    pub mle_coeff_col_evals: Option<&'a mut Vec<E::EF>>,
}
impl<'a, E: EvalAtRow, const MLE_COEFF_COL_EVAL: bool>
    BlakeRoundConstraintEval<'a, E, MLE_COEFF_COL_EVAL>
{
    pub fn eval(mut self) -> E {
        let mut v: [Fu32<E::F>; STATE_SIZE] = array::from_fn(|_| self.next_u32());
        let input_v = v;
        let m: [Fu32<E::F>; STATE_SIZE] = array::from_fn(|_| self.next_u32());

        self.g(v.get_many_mut([0, 4, 8, 12]).unwrap(), m[0], m[1]);
        self.g(v.get_many_mut([1, 5, 9, 13]).unwrap(), m[2], m[3]);
        self.g(v.get_many_mut([2, 6, 10, 14]).unwrap(), m[4], m[5]);
        self.g(v.get_many_mut([3, 7, 11, 15]).unwrap(), m[6], m[7]);
        self.g(v.get_many_mut([0, 5, 10, 15]).unwrap(), m[8], m[9]);
        self.g(v.get_many_mut([1, 6, 11, 12]).unwrap(), m[10], m[11]);
        self.g(v.get_many_mut([2, 7, 8, 13]).unwrap(), m[12], m[13]);
        self.g(v.get_many_mut([3, 4, 9, 14]).unwrap(), m[14], m[15]);

        if MLE_COEFF_COL_EVAL {
            self.mle_coeff_col_evals.as_mut().unwrap().push(
                self.round_lookup_elements.combine(
                    &chain![
                        input_v.iter().copied().flat_map(Fu32::to_felts),
                        v.iter().copied().flat_map(Fu32::to_felts),
                        m.iter().copied().flat_map(Fu32::to_felts)
                    ]
                    .collect_vec(),
                ),
            );
        }

        self.eval
    }
    fn next_u32(&mut self) -> Fu32<E::F> {
        let l = self.eval.next_trace_mask();
        let h = self.eval.next_trace_mask();
        Fu32 { l, h }
    }
    fn g(&mut self, v: [&mut Fu32<E::F>; 4], m0: Fu32<E::F>, m1: Fu32<E::F>) {
        let [a, b, c, d] = v;

        *a = self.add3_u32_unchecked(*a, *b, m0);
        *d = self.xor_rotr16_u32(*a, *d);
        *c = self.add2_u32_unchecked(*c, *d);
        *b = self.xor_rotr_u32(*b, *c, 12);
        *a = self.add3_u32_unchecked(*a, *b, m1);
        *d = self.xor_rotr_u32(*a, *d, 8);
        *c = self.add2_u32_unchecked(*c, *d);
        *b = self.xor_rotr_u32(*b, *c, 7);
    }

    /// Adds two u32s, returning the sum.
    /// Assumes a, b are properly range checked.
    /// The caller is responsible for checking:
    /// res.{l,h} not in [2^16, 2^17) or in [-2^16,0)
    fn add2_u32_unchecked(&mut self, a: Fu32<E::F>, b: Fu32<E::F>) -> Fu32<E::F> {
        let sl = self.eval.next_trace_mask();
        let sh = self.eval.next_trace_mask();

        let carry_l = (a.l + b.l - sl) * E::F::from(INV16);
        self.eval.add_constraint(carry_l * carry_l - carry_l);

        let carry_h = (a.h + b.h + carry_l - sh) * E::F::from(INV16);
        self.eval.add_constraint(carry_h * carry_h - carry_h);

        Fu32 { l: sl, h: sh }
    }

    /// Adds three u32s, returning the sum.
    /// Assumes a, b, c are properly range checked.
    /// Caller is responsible for checking:
    /// res.{l,h} not in [2^16, 3*2^16) or in [-2^17,0)
    fn add3_u32_unchecked(&mut self, a: Fu32<E::F>, b: Fu32<E::F>, c: Fu32<E::F>) -> Fu32<E::F> {
        let sl = self.eval.next_trace_mask();
        let sh = self.eval.next_trace_mask();

        let carry_l = (a.l + b.l + c.l - sl) * E::F::from(INV16);
        self.eval
            .add_constraint(carry_l * (carry_l - E::F::one()) * (carry_l - E::F::from(TWO)));

        let carry_h = (a.h + b.h + c.h + carry_l - sh) * E::F::from(INV16);
        self.eval
            .add_constraint(carry_h * (carry_h - E::F::one()) * (carry_h - E::F::from(TWO)));

        Fu32 { l: sl, h: sh }
    }

    /// Splits a felt at r.
    /// Caller is responsible for checking that the ranges of h * 2^r and l don't overlap.
    fn split_unchecked(&mut self, a: E::F, r: u32) -> (E::F, E::F) {
        let h = self.eval.next_trace_mask();
        let l = a - h * E::F::from(BaseField::from_u32_unchecked(1 << r));
        (l, h)
    }

    /// Checks that a, b are in range, and computes their xor rotated right by `r` bits.
    /// Guarantees that all elements are in range.
    fn xor_rotr_u32(&mut self, a: Fu32<E::F>, b: Fu32<E::F>, r: u32) -> Fu32<E::F> {
        let (all, alh) = self.split_unchecked(a.l, r);
        let (ahl, ahh) = self.split_unchecked(a.h, r);
        let (bll, blh) = self.split_unchecked(b.l, r);
        let (bhl, bhh) = self.split_unchecked(b.h, r);

        // These also guarantee that all elements are in range.
        let xorll = self.xor(r, all, bll);
        let xorlh = self.xor(16 - r, alh, blh);
        let xorhl = self.xor(r, ahl, bhl);
        let xorhh = self.xor(16 - r, ahh, bhh);

        Fu32 {
            l: xorhl * E::F::from(BaseField::from_u32_unchecked(1 << (16 - r))) + xorlh,
            h: xorll * E::F::from(BaseField::from_u32_unchecked(1 << (16 - r))) + xorhh,
        }
    }

    /// Checks that a, b are in range, and computes their xor rotated right by 16 bits.
    /// Guarantees that all elements are in range.
    fn xor_rotr16_u32(&mut self, a: Fu32<E::F>, b: Fu32<E::F>) -> Fu32<E::F> {
        let (all, alh) = self.split_unchecked(a.l, 8);
        let (ahl, ahh) = self.split_unchecked(a.h, 8);
        let (bll, blh) = self.split_unchecked(b.l, 8);
        let (bhl, bhh) = self.split_unchecked(b.h, 8);

        // These also guarantee that all elements are in range.
        let xorll = self.xor(8, all, bll);
        let xorlh = self.xor(8, alh, blh);
        let xorhl = self.xor(8, ahl, bhl);
        let xorhh = self.xor(8, ahh, bhh);

        Fu32 {
            l: xorhh * E::F::from(BaseField::from_u32_unchecked(1 << 8)) + xorhl,
            h: xorlh * E::F::from(BaseField::from_u32_unchecked(1 << 8)) + xorll,
        }
    }

    /// Checks that a, b are in [0, 2^w) and computes their xor.
    fn xor(&mut self, w: u32, a: E::F, b: E::F) -> E::F {
        // TODO: Separate lookups by w.
        let c = self.eval.next_trace_mask();

        if MLE_COEFF_COL_EVAL {
            let lookup_elements = self.xor_lookup_elements.get(w);
            self.mle_coeff_col_evals
                .as_mut()
                .unwrap()
                .push(lookup_elements.combine(&[a, b, c]));
        }

        c
    }
}

pub struct RoundLookupArtifact {
    pub round: LogUpArtifactInstance,
    pub xors: Vec<LogUpArtifactInstance>,
}

impl RoundLookupArtifact {
    pub fn new_from_iter(mut iter: impl Iterator<Item = LookupArtifactInstance>) -> Self {
        let xors = (0..n_xor_lookups())
            .map(|_| match iter.next() {
                Some(LookupArtifactInstance::LogUp(artifact)) => artifact,
                _ => panic!(),
            })
            .collect();

        let round = match iter.next() {
            Some(LookupArtifactInstance::LogUp(artifact)) => artifact,
            _ => panic!(),
        };

        Self { round, xors }
    }

    pub fn accumulate_mle_eval_iop_claims(&self, acc: &mut MleClaimAccumulator) {
        let Self { round, xors } = self;

        for xor in xors {
            acc.accumulate(xor.input_n_variables, xor.input_denominators_claim);
        }

        acc.accumulate(round.input_n_variables, round.input_denominators_claim);
    }

    pub fn verify_succinct_mle_claims(&self) -> Result<(), InvalidClaimError> {
        let Self { round, xors } = self;

        for xor_artifact in xors {
            if !xor_artifact.input_numerators_claim.is_one() {
                return Err(InvalidClaimError);
            }
        }

        if !round.input_numerators_claim.is_one() {
            return Err(InvalidClaimError);
        }

        Ok(())
    }
}

/// Returns an ordered list of all XOR lookup types the round component uses.
fn n_xor_lookups() -> usize {
    // Create a dummy trace to extract the structural xor lookup information.
    let mut trace_generator = TraceGenerator::new(LOG_N_LANES);
    let mut row = trace_generator.gen_row(0);
    row.generate(Default::default(), Default::default());
    trace_generator.xor_lookups.len()
}

pub fn generate_lookup_instances(
    log_size: u32,
    lookup_data: BlakeRoundLookupData,
    xor_lookup_elements: &BlakeXorElements,
    round_lookup_elements: &RoundElements,
    collection_for_univariate_iop: &mut MleCollection<SimdBackend>,
) -> Vec<Layer<SimdBackend>> {
    let _span = span!(Level::INFO, "Generate round interaction trace").entered();
    let size = 1 << log_size;
    let mut round_lookup_layers = Vec::new();

    for (w, l) in &lookup_data.xor_lookups {
        let lookup_elements = xor_lookup_elements.get(*w);
        let mut denominators = Mle::<SimdBackend, SecureField>::new(SecureColumn::zeros(size));
        for vec_row in 0..1 << (log_size - LOG_N_LANES) {
            let denom = lookup_elements.combine(&l.each_ref().map(|l| l.data[vec_row]));
            denominators.data[vec_row] = denom;
        }
        collection_for_univariate_iop.push(denominators.clone());
        round_lookup_layers.push(Layer::LogUpSingles { denominators });
    }

    // Blake round lookup.
    let mut round_denominators = Mle::<SimdBackend, SecureField>::new(SecureColumn::zeros(size));
    for vec_row in 0..1 << (log_size - LOG_N_LANES) {
        let denom = round_lookup_elements
            .combine(&lookup_data.round_lookup.each_ref().map(|l| l.data[vec_row]));
        round_denominators.data[vec_row] = denom;
    }
    collection_for_univariate_iop.push(round_denominators.clone());
    round_lookup_layers.push(Layer::LogUpSingles {
        denominators: round_denominators,
    });

    round_lookup_layers
}
