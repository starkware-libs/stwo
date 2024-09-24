use std::array;
use std::iter::zip;
use std::simd::u32x16;

use itertools::Itertools;
use num_traits::{One, Zero};
use tracing::{span, Level};

use super::air::InvalidClaimError;
use super::gkr_lookups::accumulation::{MleClaimAccumulator, MleCollection};
use super::gkr_lookups::AccumulatedMleCoeffColumnOracle;
use crate::constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, PointEvaluator};
use crate::core::air::accumulation::PointEvaluationAccumulator;
use crate::core::backend::simd::column::SecureColumn;
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;
use crate::core::lookups::gkr_prover::Layer;
use crate::core::lookups::gkr_verifier::{LogUpArtifactInstance, LookupArtifactInstance};
use crate::core::lookups::mle::Mle;
use crate::core::pcs::TreeVec;
use crate::core::ColumnVec;
use crate::examples::blake::xor_table::{column_bits, limb_bits, XorElements, XorTableLookupData};
use crate::examples::blake::BlakeXorElements;

/// Component that evaluates the xor table.
pub type XorTableComponent<const ELEM_BITS: u32, const EXPAND_BITS: u32> =
    FrameworkComponent<XorTableEval<ELEM_BITS, EXPAND_BITS>>;

/// Evaluates the xor table.
pub struct XorTableEval<const ELEM_BITS: u32, const EXPAND_BITS: u32> {
    pub lookup_elements: XorElements,
}

impl<const ELEM_BITS: u32, const EXPAND_BITS: u32> FrameworkEval
    for XorTableEval<ELEM_BITS, EXPAND_BITS>
{
    fn log_size(&self) -> u32 {
        column_bits::<ELEM_BITS, EXPAND_BITS>()
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        column_bits::<ELEM_BITS, EXPAND_BITS>() + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let _ = eval_xor_table_multiplicity_cols::<E, ELEM_BITS, EXPAND_BITS>(&mut eval);
        eval
    }
}

impl<const ELEM_BITS: u32, const EXPAND_BITS: u32> AccumulatedMleCoeffColumnOracle
    for XorTableComponent<ELEM_BITS, EXPAND_BITS>
{
    fn accumulate_at_point(
        &self,
        _point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        acc: &mut PointEvaluationAccumulator,
    ) {
        // Create dummy point evaluator just to extract the value we need from the mask
        let mut _accumulator = PointEvaluationAccumulator::new(SecureField::one());
        let mut eval = PointEvaluator::new(
            mask.sub_tree(self.trace_locations()),
            &mut _accumulator,
            SecureField::one(),
        );

        for eval in eval_xor_table_multiplicity_cols::<_, ELEM_BITS, EXPAND_BITS>(&mut eval) {
            acc.accumulate(eval)
        }
    }
}

fn eval_xor_table_multiplicity_cols<E: EvalAtRow, const ELEM_BITS: u32, const EXPAND_BITS: u32>(
    eval: &mut E,
) -> Vec<E::F> {
    (0..1 << (2 * EXPAND_BITS))
        .map(|_| eval.next_trace_mask())
        .collect()
}

pub struct XorLookupArtifacts {
    xor12: XorLookupArtifact<12, 4>,
    xor9: XorLookupArtifact<9, 2>,
    xor8: XorLookupArtifact<8, 2>,
    xor7: XorLookupArtifact<7, 2>,
    xor4: XorLookupArtifact<4, 0>,
}

impl XorLookupArtifacts {
    pub fn new_from_iter(mut iter: impl Iterator<Item = LookupArtifactInstance>) -> Self {
        Self {
            xor12: XorLookupArtifact::new_from_iter(&mut iter),
            xor9: XorLookupArtifact::new_from_iter(&mut iter),
            xor8: XorLookupArtifact::new_from_iter(&mut iter),
            xor7: XorLookupArtifact::new_from_iter(&mut iter),
            xor4: XorLookupArtifact::new_from_iter(&mut iter),
        }
    }

    pub fn verify_succinct_mle_claims(
        &self,
        lookup_elements: &BlakeXorElements,
    ) -> Result<(), InvalidClaimError> {
        let Self {
            xor12,
            xor9,
            xor8,
            xor7,
            xor4,
        } = self;

        xor12.verify_succinct_mle_claims(lookup_elements.get(12))?;
        xor9.verify_succinct_mle_claims(lookup_elements.get(9))?;
        xor8.verify_succinct_mle_claims(lookup_elements.get(8))?;
        xor7.verify_succinct_mle_claims(lookup_elements.get(7))?;
        xor4.verify_succinct_mle_claims(lookup_elements.get(4))?;

        Ok(())
    }

    pub fn accumulate_mle_eval_iop_claims(&self, acc: &mut MleClaimAccumulator) {
        let Self {
            xor12,
            xor9,
            xor8,
            xor7,
            xor4,
        } = self;
        xor12.accumulate_mle_eval_iop_claims(acc);
        xor9.accumulate_mle_eval_iop_claims(acc);
        xor8.accumulate_mle_eval_iop_claims(acc);
        xor7.accumulate_mle_eval_iop_claims(acc);
        xor4.accumulate_mle_eval_iop_claims(acc);
    }
}

pub struct XorLookupArtifact<const ELEM_BITS: u32, const EXPAND_BITS: u32> {
    /// `2^(2*EXPAND_BITS)` many LogUp instances.
    artifacts: Vec<LogUpArtifactInstance>,
}

impl<const ELEM_BITS: u32, const EXPAND_BITS: u32> XorLookupArtifact<ELEM_BITS, EXPAND_BITS> {
    pub fn new_from_iter(mut iter: impl Iterator<Item = LookupArtifactInstance>) -> Self {
        Self {
            artifacts: (0..1 << (2 * EXPAND_BITS))
                .map(|_| match iter.next() {
                    // TODO: check input MLEs have expected number of variables.
                    Some(LookupArtifactInstance::LogUp(artifact)) => artifact,
                    _ => panic!(),
                })
                .collect(),
        }
    }

    fn verify_succinct_mle_claims(
        &self,
        lookup_elements: &XorElements,
    ) -> Result<(), InvalidClaimError> {
        for (i, artifact) in self.artifacts.iter().enumerate() {
            let eval_point = &artifact.eval_point;
            let denoms_claim = artifact.input_denominators_claim;
            let denoms_eval = eval_logup_denominators_mle::<ELEM_BITS, EXPAND_BITS>(
                i,
                lookup_elements,
                eval_point,
            )
            .unwrap();

            if denoms_claim != denoms_eval {
                return Err(InvalidClaimError);
            }
        }

        Ok(())
    }

    fn accumulate_mle_eval_iop_claims(&self, acc: &mut MleClaimAccumulator) {
        let Self { artifacts } = self;
        for artifact in artifacts {
            acc.accumulate(artifact.input_n_variables, artifact.input_numerators_claim);
        }
    }
}

pub fn generate_lookup_instances<const ELEM_BITS: u32, const EXPAND_BITS: u32>(
    lookup_data: XorTableLookupData<ELEM_BITS, EXPAND_BITS>,
    lookup_elements: &XorElements,
    collection_for_univariate_iop: &mut MleCollection<SimdBackend>,
) -> Vec<Layer<SimdBackend>> {
    let _span = span!(Level::INFO, "Xor interaction trace").entered();
    let mut xor_lookup_layers = Vec::new();

    // There are 2^(2*EXPAND_BITS) columns, for each combination of ah, bh.
    for (column_index, mults) in lookup_data.xor_accum.mults.iter().enumerate() {
        let numerators = Mle::<SimdBackend, BaseField>::new(mults.clone());
        collection_for_univariate_iop.push(numerators.clone());
        let denominators =
            gen_logup_denominators_mle::<ELEM_BITS, EXPAND_BITS>(column_index, lookup_elements);
        xor_lookup_layers.push(Layer::LogUpMultiplicities {
            numerators,
            denominators,
        });
    }

    xor_lookup_layers
}

/// Returns an MLE representing the LogUp denominator terms for the xor table.
fn gen_logup_denominators_mle<const ELEM_BITS: u32, const EXPAND_BITS: u32>(
    column_index: usize,
    lookup_elements: &XorElements,
) -> Mle<SimdBackend, SecureField> {
    let offsets_vec = u32x16::from_array(array::from_fn(|i| i as u32));
    let column_bits = column_bits::<ELEM_BITS, EXPAND_BITS>();
    let column_size = 1 << column_bits;
    let mut denominators = Mle::<SimdBackend, SecureField>::new(SecureColumn::zeros(column_size));

    // Extract ah, bh from column index.
    let ah = column_index as u32 >> EXPAND_BITS;
    let bh = column_index as u32 & ((1 << EXPAND_BITS) - 1);

    // Each column has 2^(2*LIMB_BITS) rows, packed in N_LANES.
    for vec_row in 0..1 << (column_bits - LOG_N_LANES) {
        let limb_bits = limb_bits::<ELEM_BITS, EXPAND_BITS>();

        // vec_row is LIMB_BITS of al and LIMB_BITS - LOG_N_LANES of bl.
        // Extract al, blh from vec_row.
        let al = vec_row >> (limb_bits - LOG_N_LANES);
        let blh = vec_row & ((1 << (limb_bits - LOG_N_LANES)) - 1);

        // Construct the 3 vectors a, b, c.
        let a = u32x16::splat((ah << limb_bits) | al);
        // bll is just the consecutive numbers 0..N_LANES-1.
        let b = u32x16::splat((bh << limb_bits) | (blh << LOG_N_LANES)) | offsets_vec;
        let c = a ^ b;

        let denominator = lookup_elements
            .combine(&[a, b, c].map(|v| unsafe { PackedBaseField::from_simd_unchecked(v) }));
        denominators.data[vec_row as usize] = denominator;
    }

    denominators
}

/// Evaluates the succinct MLE representing the LogUp denominator terms for the xor table.
///
/// Evaluates the MLE returned by [`gen_logup_denominators_mle`].
fn eval_logup_denominators_mle<const ELEM_BITS: u32, const EXPAND_BITS: u32>(
    column_index: usize,
    lookup_elements: &XorElements,
    eval_point: &[SecureField],
) -> Result<SecureField, InvalidEvalPoint> {
    assert!(column_index < 1 << (2 * EXPAND_BITS));
    let limb_bits = limb_bits::<ELEM_BITS, EXPAND_BITS>() as usize;
    if eval_point.len() != limb_bits * 2 {
        return Err(InvalidEvalPoint);
    }

    let (al_assignment, bl_assignment) = eval_point.split_at(limb_bits);
    let cl_assignment = &zip(al_assignment, bl_assignment)
        // Note `a ^ b = a + b - 2 * a * b` for all `a, b` in `{0, 1}`.
        .map(|(&li, &ri)| li + ri - (li * ri).double())
        .collect_vec();

    let al = pack_little_endian_bits(al_assignment);
    let bl = pack_little_endian_bits(bl_assignment);
    let cl = pack_little_endian_bits(cl_assignment);

    // Extract ah, bh from column index.
    let ah = column_index >> EXPAND_BITS;
    let bh = column_index & ((1 << EXPAND_BITS) - 1);
    let ch = ah ^ bh;

    let a = al + BaseField::from(ah << limb_bits);
    let b = bl + BaseField::from(bh << limb_bits);
    let c = cl + BaseField::from(ch << limb_bits);

    Ok(lookup_elements.combine(&[a, b, c]))
}

fn pack_little_endian_bits(bits: &[SecureField]) -> SecureField {
    bits.iter()
        .fold(SecureField::zero(), |acc, &bit| acc.double() + bit)
}

/// Eval point is invalid.
#[derive(Debug)]
struct InvalidEvalPoint;

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::core::channel::Channel;
    use crate::core::test_utils::test_channel;
    use crate::examples::blake::xor_table::XorElements;
    use crate::examples::blake_gkr::xor_table::{
        eval_logup_denominators_mle, gen_logup_denominators_mle,
    };

    #[test]
    fn eval_logup_denominators_mle_works() {
        const ELEM_BITS: u32 = 8;
        const EXPAND_BITS: u32 = 2;
        let column_index = 0b1011;
        assert!((0..1 << (2 * EXPAND_BITS)).contains(&column_index));
        let channel = &mut test_channel();
        let lookup_elements = XorElements::draw(channel);
        let denominators_mle =
            gen_logup_denominators_mle::<ELEM_BITS, EXPAND_BITS>(column_index, &lookup_elements);
        let eval_point = (0..denominators_mle.n_variables())
            .map(|_| channel.draw_felt())
            .collect_vec();

        let eval = eval_logup_denominators_mle::<ELEM_BITS, EXPAND_BITS>(
            column_index,
            &lookup_elements,
            &eval_point,
        )
        .unwrap();

        assert_eq!(eval, denominators_mle.eval_at_point(&eval_point));
    }
}
