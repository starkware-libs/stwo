use num_traits::Zero;

use super::structs::WideFibComponent;
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentTrace, Mask};
use crate::core::backend::{CPUBackend, Column};
use crate::core::circle::CirclePoint;
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::CanonicCoset;
use crate::core::utils::bit_reverse;
use crate::core::ColumnVec;

impl Component<CPUBackend> for WideFibComponent {
    fn n_constraints(&self) -> usize {
        255
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }
    fn trace_log_degree_bounds(&self) -> Vec<u32> {
        vec![self.log_size; 256]
    }
    #[allow(unused_parens)]
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, CPUBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CPUBackend>,
    ) {
        let max_constraint_degree = Component::<CPUBackend>::max_constraint_log_degree_bound(self);
        let trace_eval_domain = CanonicCoset::new(max_constraint_degree).circle_domain();
        let mut trace_evals = vec![];
        for poly_index in 0..256 {
            let poly = &trace.columns[poly_index];
            trace_evals.push(poly.evaluate(trace_eval_domain));
        }
        let zero_domain = CanonicCoset::new(self.log_size).coset;
        let eval_domain = CanonicCoset::new(max_constraint_degree).circle_domain();
        let mut denoms = vec![];
        for point in eval_domain.iter() {
            denoms.push(coset_vanishing(zero_domain, point));
        }
        bit_reverse(&mut denoms);
        let mut denom_inverses = vec![BaseField::zero(); 1 << (max_constraint_degree)];
        BaseField::batch_inverse(&denoms, &mut denom_inverses);
        let mut numerators = vec![SecureField::zero(); 1 << (max_constraint_degree)];
        let [mut accum] = evaluation_accumulator.columns([(
            max_constraint_degree,
            Component::<CPUBackend>::n_constraints(self),
        )]);

        #[allow(clippy::needless_range_loop)]
        for i in 0..eval_domain.size() {
            numerators[i] += accum.random_coeff_powers[254]
                * (trace_evals[0].values.at(i) - BaseField::from_u32_unchecked(1));
            numerators[i] += accum.random_coeff_powers[253]
                * (trace_evals[2].values.at(i)
                    - ((trace_evals[0].values.at(i) * trace_evals[0].values.at(i))
                        + (trace_evals[1].values.at(i) * trace_evals[1].values.at(i))));
            numerators[i] += accum.random_coeff_powers[252]
                * (trace_evals[3].values.at(i)
                    - ((trace_evals[1].values.at(i) * trace_evals[1].values.at(i))
                        + (trace_evals[2].values.at(i) * trace_evals[2].values.at(i))));
            numerators[i] += accum.random_coeff_powers[251]
                * (trace_evals[4].values.at(i)
                    - ((trace_evals[2].values.at(i) * trace_evals[2].values.at(i))
                        + (trace_evals[3].values.at(i) * trace_evals[3].values.at(i))));
            numerators[i] += accum.random_coeff_powers[250]
                * (trace_evals[5].values.at(i)
                    - ((trace_evals[3].values.at(i) * trace_evals[3].values.at(i))
                        + (trace_evals[4].values.at(i) * trace_evals[4].values.at(i))));
            numerators[i] += accum.random_coeff_powers[249]
                * (trace_evals[6].values.at(i)
                    - ((trace_evals[4].values.at(i) * trace_evals[4].values.at(i))
                        + (trace_evals[5].values.at(i) * trace_evals[5].values.at(i))));
            numerators[i] += accum.random_coeff_powers[248]
                * (trace_evals[7].values.at(i)
                    - ((trace_evals[5].values.at(i) * trace_evals[5].values.at(i))
                        + (trace_evals[6].values.at(i) * trace_evals[6].values.at(i))));
            numerators[i] += accum.random_coeff_powers[247]
                * (trace_evals[8].values.at(i)
                    - ((trace_evals[6].values.at(i) * trace_evals[6].values.at(i))
                        + (trace_evals[7].values.at(i) * trace_evals[7].values.at(i))));
            numerators[i] += accum.random_coeff_powers[246]
                * (trace_evals[9].values.at(i)
                    - ((trace_evals[7].values.at(i) * trace_evals[7].values.at(i))
                        + (trace_evals[8].values.at(i) * trace_evals[8].values.at(i))));
            numerators[i] += accum.random_coeff_powers[245]
                * (trace_evals[10].values.at(i)
                    - ((trace_evals[8].values.at(i) * trace_evals[8].values.at(i))
                        + (trace_evals[9].values.at(i) * trace_evals[9].values.at(i))));
            numerators[i] += accum.random_coeff_powers[244]
                * (trace_evals[11].values.at(i)
                    - ((trace_evals[9].values.at(i) * trace_evals[9].values.at(i))
                        + (trace_evals[10].values.at(i) * trace_evals[10].values.at(i))));
            numerators[i] += accum.random_coeff_powers[243]
                * (trace_evals[12].values.at(i)
                    - ((trace_evals[10].values.at(i) * trace_evals[10].values.at(i))
                        + (trace_evals[11].values.at(i) * trace_evals[11].values.at(i))));
            numerators[i] += accum.random_coeff_powers[242]
                * (trace_evals[13].values.at(i)
                    - ((trace_evals[11].values.at(i) * trace_evals[11].values.at(i))
                        + (trace_evals[12].values.at(i) * trace_evals[12].values.at(i))));
            numerators[i] += accum.random_coeff_powers[241]
                * (trace_evals[14].values.at(i)
                    - ((trace_evals[12].values.at(i) * trace_evals[12].values.at(i))
                        + (trace_evals[13].values.at(i) * trace_evals[13].values.at(i))));
            numerators[i] += accum.random_coeff_powers[240]
                * (trace_evals[15].values.at(i)
                    - ((trace_evals[13].values.at(i) * trace_evals[13].values.at(i))
                        + (trace_evals[14].values.at(i) * trace_evals[14].values.at(i))));
            numerators[i] += accum.random_coeff_powers[239]
                * (trace_evals[16].values.at(i)
                    - ((trace_evals[14].values.at(i) * trace_evals[14].values.at(i))
                        + (trace_evals[15].values.at(i) * trace_evals[15].values.at(i))));
            numerators[i] += accum.random_coeff_powers[238]
                * (trace_evals[17].values.at(i)
                    - ((trace_evals[15].values.at(i) * trace_evals[15].values.at(i))
                        + (trace_evals[16].values.at(i) * trace_evals[16].values.at(i))));
            numerators[i] += accum.random_coeff_powers[237]
                * (trace_evals[18].values.at(i)
                    - ((trace_evals[16].values.at(i) * trace_evals[16].values.at(i))
                        + (trace_evals[17].values.at(i) * trace_evals[17].values.at(i))));
            numerators[i] += accum.random_coeff_powers[236]
                * (trace_evals[19].values.at(i)
                    - ((trace_evals[17].values.at(i) * trace_evals[17].values.at(i))
                        + (trace_evals[18].values.at(i) * trace_evals[18].values.at(i))));
            numerators[i] += accum.random_coeff_powers[235]
                * (trace_evals[20].values.at(i)
                    - ((trace_evals[18].values.at(i) * trace_evals[18].values.at(i))
                        + (trace_evals[19].values.at(i) * trace_evals[19].values.at(i))));
            numerators[i] += accum.random_coeff_powers[234]
                * (trace_evals[21].values.at(i)
                    - ((trace_evals[19].values.at(i) * trace_evals[19].values.at(i))
                        + (trace_evals[20].values.at(i) * trace_evals[20].values.at(i))));
            numerators[i] += accum.random_coeff_powers[233]
                * (trace_evals[22].values.at(i)
                    - ((trace_evals[20].values.at(i) * trace_evals[20].values.at(i))
                        + (trace_evals[21].values.at(i) * trace_evals[21].values.at(i))));
            numerators[i] += accum.random_coeff_powers[232]
                * (trace_evals[23].values.at(i)
                    - ((trace_evals[21].values.at(i) * trace_evals[21].values.at(i))
                        + (trace_evals[22].values.at(i) * trace_evals[22].values.at(i))));
            numerators[i] += accum.random_coeff_powers[231]
                * (trace_evals[24].values.at(i)
                    - ((trace_evals[22].values.at(i) * trace_evals[22].values.at(i))
                        + (trace_evals[23].values.at(i) * trace_evals[23].values.at(i))));
            numerators[i] += accum.random_coeff_powers[230]
                * (trace_evals[25].values.at(i)
                    - ((trace_evals[23].values.at(i) * trace_evals[23].values.at(i))
                        + (trace_evals[24].values.at(i) * trace_evals[24].values.at(i))));
            numerators[i] += accum.random_coeff_powers[229]
                * (trace_evals[26].values.at(i)
                    - ((trace_evals[24].values.at(i) * trace_evals[24].values.at(i))
                        + (trace_evals[25].values.at(i) * trace_evals[25].values.at(i))));
            numerators[i] += accum.random_coeff_powers[228]
                * (trace_evals[27].values.at(i)
                    - ((trace_evals[25].values.at(i) * trace_evals[25].values.at(i))
                        + (trace_evals[26].values.at(i) * trace_evals[26].values.at(i))));
            numerators[i] += accum.random_coeff_powers[227]
                * (trace_evals[28].values.at(i)
                    - ((trace_evals[26].values.at(i) * trace_evals[26].values.at(i))
                        + (trace_evals[27].values.at(i) * trace_evals[27].values.at(i))));
            numerators[i] += accum.random_coeff_powers[226]
                * (trace_evals[29].values.at(i)
                    - ((trace_evals[27].values.at(i) * trace_evals[27].values.at(i))
                        + (trace_evals[28].values.at(i) * trace_evals[28].values.at(i))));
            numerators[i] += accum.random_coeff_powers[225]
                * (trace_evals[30].values.at(i)
                    - ((trace_evals[28].values.at(i) * trace_evals[28].values.at(i))
                        + (trace_evals[29].values.at(i) * trace_evals[29].values.at(i))));
            numerators[i] += accum.random_coeff_powers[224]
                * (trace_evals[31].values.at(i)
                    - ((trace_evals[29].values.at(i) * trace_evals[29].values.at(i))
                        + (trace_evals[30].values.at(i) * trace_evals[30].values.at(i))));
            numerators[i] += accum.random_coeff_powers[223]
                * (trace_evals[32].values.at(i)
                    - ((trace_evals[30].values.at(i) * trace_evals[30].values.at(i))
                        + (trace_evals[31].values.at(i) * trace_evals[31].values.at(i))));
            numerators[i] += accum.random_coeff_powers[222]
                * (trace_evals[33].values.at(i)
                    - ((trace_evals[31].values.at(i) * trace_evals[31].values.at(i))
                        + (trace_evals[32].values.at(i) * trace_evals[32].values.at(i))));
            numerators[i] += accum.random_coeff_powers[221]
                * (trace_evals[34].values.at(i)
                    - ((trace_evals[32].values.at(i) * trace_evals[32].values.at(i))
                        + (trace_evals[33].values.at(i) * trace_evals[33].values.at(i))));
            numerators[i] += accum.random_coeff_powers[220]
                * (trace_evals[35].values.at(i)
                    - ((trace_evals[33].values.at(i) * trace_evals[33].values.at(i))
                        + (trace_evals[34].values.at(i) * trace_evals[34].values.at(i))));
            numerators[i] += accum.random_coeff_powers[219]
                * (trace_evals[36].values.at(i)
                    - ((trace_evals[34].values.at(i) * trace_evals[34].values.at(i))
                        + (trace_evals[35].values.at(i) * trace_evals[35].values.at(i))));
            numerators[i] += accum.random_coeff_powers[218]
                * (trace_evals[37].values.at(i)
                    - ((trace_evals[35].values.at(i) * trace_evals[35].values.at(i))
                        + (trace_evals[36].values.at(i) * trace_evals[36].values.at(i))));
            numerators[i] += accum.random_coeff_powers[217]
                * (trace_evals[38].values.at(i)
                    - ((trace_evals[36].values.at(i) * trace_evals[36].values.at(i))
                        + (trace_evals[37].values.at(i) * trace_evals[37].values.at(i))));
            numerators[i] += accum.random_coeff_powers[216]
                * (trace_evals[39].values.at(i)
                    - ((trace_evals[37].values.at(i) * trace_evals[37].values.at(i))
                        + (trace_evals[38].values.at(i) * trace_evals[38].values.at(i))));
            numerators[i] += accum.random_coeff_powers[215]
                * (trace_evals[40].values.at(i)
                    - ((trace_evals[38].values.at(i) * trace_evals[38].values.at(i))
                        + (trace_evals[39].values.at(i) * trace_evals[39].values.at(i))));
            numerators[i] += accum.random_coeff_powers[214]
                * (trace_evals[41].values.at(i)
                    - ((trace_evals[39].values.at(i) * trace_evals[39].values.at(i))
                        + (trace_evals[40].values.at(i) * trace_evals[40].values.at(i))));
            numerators[i] += accum.random_coeff_powers[213]
                * (trace_evals[42].values.at(i)
                    - ((trace_evals[40].values.at(i) * trace_evals[40].values.at(i))
                        + (trace_evals[41].values.at(i) * trace_evals[41].values.at(i))));
            numerators[i] += accum.random_coeff_powers[212]
                * (trace_evals[43].values.at(i)
                    - ((trace_evals[41].values.at(i) * trace_evals[41].values.at(i))
                        + (trace_evals[42].values.at(i) * trace_evals[42].values.at(i))));
            numerators[i] += accum.random_coeff_powers[211]
                * (trace_evals[44].values.at(i)
                    - ((trace_evals[42].values.at(i) * trace_evals[42].values.at(i))
                        + (trace_evals[43].values.at(i) * trace_evals[43].values.at(i))));
            numerators[i] += accum.random_coeff_powers[210]
                * (trace_evals[45].values.at(i)
                    - ((trace_evals[43].values.at(i) * trace_evals[43].values.at(i))
                        + (trace_evals[44].values.at(i) * trace_evals[44].values.at(i))));
            numerators[i] += accum.random_coeff_powers[209]
                * (trace_evals[46].values.at(i)
                    - ((trace_evals[44].values.at(i) * trace_evals[44].values.at(i))
                        + (trace_evals[45].values.at(i) * trace_evals[45].values.at(i))));
            numerators[i] += accum.random_coeff_powers[208]
                * (trace_evals[47].values.at(i)
                    - ((trace_evals[45].values.at(i) * trace_evals[45].values.at(i))
                        + (trace_evals[46].values.at(i) * trace_evals[46].values.at(i))));
            numerators[i] += accum.random_coeff_powers[207]
                * (trace_evals[48].values.at(i)
                    - ((trace_evals[46].values.at(i) * trace_evals[46].values.at(i))
                        + (trace_evals[47].values.at(i) * trace_evals[47].values.at(i))));
            numerators[i] += accum.random_coeff_powers[206]
                * (trace_evals[49].values.at(i)
                    - ((trace_evals[47].values.at(i) * trace_evals[47].values.at(i))
                        + (trace_evals[48].values.at(i) * trace_evals[48].values.at(i))));
            numerators[i] += accum.random_coeff_powers[205]
                * (trace_evals[50].values.at(i)
                    - ((trace_evals[48].values.at(i) * trace_evals[48].values.at(i))
                        + (trace_evals[49].values.at(i) * trace_evals[49].values.at(i))));
            numerators[i] += accum.random_coeff_powers[204]
                * (trace_evals[51].values.at(i)
                    - ((trace_evals[49].values.at(i) * trace_evals[49].values.at(i))
                        + (trace_evals[50].values.at(i) * trace_evals[50].values.at(i))));
            numerators[i] += accum.random_coeff_powers[203]
                * (trace_evals[52].values.at(i)
                    - ((trace_evals[50].values.at(i) * trace_evals[50].values.at(i))
                        + (trace_evals[51].values.at(i) * trace_evals[51].values.at(i))));
            numerators[i] += accum.random_coeff_powers[202]
                * (trace_evals[53].values.at(i)
                    - ((trace_evals[51].values.at(i) * trace_evals[51].values.at(i))
                        + (trace_evals[52].values.at(i) * trace_evals[52].values.at(i))));
            numerators[i] += accum.random_coeff_powers[201]
                * (trace_evals[54].values.at(i)
                    - ((trace_evals[52].values.at(i) * trace_evals[52].values.at(i))
                        + (trace_evals[53].values.at(i) * trace_evals[53].values.at(i))));
            numerators[i] += accum.random_coeff_powers[200]
                * (trace_evals[55].values.at(i)
                    - ((trace_evals[53].values.at(i) * trace_evals[53].values.at(i))
                        + (trace_evals[54].values.at(i) * trace_evals[54].values.at(i))));
            numerators[i] += accum.random_coeff_powers[199]
                * (trace_evals[56].values.at(i)
                    - ((trace_evals[54].values.at(i) * trace_evals[54].values.at(i))
                        + (trace_evals[55].values.at(i) * trace_evals[55].values.at(i))));
            numerators[i] += accum.random_coeff_powers[198]
                * (trace_evals[57].values.at(i)
                    - ((trace_evals[55].values.at(i) * trace_evals[55].values.at(i))
                        + (trace_evals[56].values.at(i) * trace_evals[56].values.at(i))));
            numerators[i] += accum.random_coeff_powers[197]
                * (trace_evals[58].values.at(i)
                    - ((trace_evals[56].values.at(i) * trace_evals[56].values.at(i))
                        + (trace_evals[57].values.at(i) * trace_evals[57].values.at(i))));
            numerators[i] += accum.random_coeff_powers[196]
                * (trace_evals[59].values.at(i)
                    - ((trace_evals[57].values.at(i) * trace_evals[57].values.at(i))
                        + (trace_evals[58].values.at(i) * trace_evals[58].values.at(i))));
            numerators[i] += accum.random_coeff_powers[195]
                * (trace_evals[60].values.at(i)
                    - ((trace_evals[58].values.at(i) * trace_evals[58].values.at(i))
                        + (trace_evals[59].values.at(i) * trace_evals[59].values.at(i))));
            numerators[i] += accum.random_coeff_powers[194]
                * (trace_evals[61].values.at(i)
                    - ((trace_evals[59].values.at(i) * trace_evals[59].values.at(i))
                        + (trace_evals[60].values.at(i) * trace_evals[60].values.at(i))));
            numerators[i] += accum.random_coeff_powers[193]
                * (trace_evals[62].values.at(i)
                    - ((trace_evals[60].values.at(i) * trace_evals[60].values.at(i))
                        + (trace_evals[61].values.at(i) * trace_evals[61].values.at(i))));
            numerators[i] += accum.random_coeff_powers[192]
                * (trace_evals[63].values.at(i)
                    - ((trace_evals[61].values.at(i) * trace_evals[61].values.at(i))
                        + (trace_evals[62].values.at(i) * trace_evals[62].values.at(i))));
            numerators[i] += accum.random_coeff_powers[191]
                * (trace_evals[64].values.at(i)
                    - ((trace_evals[62].values.at(i) * trace_evals[62].values.at(i))
                        + (trace_evals[63].values.at(i) * trace_evals[63].values.at(i))));
            numerators[i] += accum.random_coeff_powers[190]
                * (trace_evals[65].values.at(i)
                    - ((trace_evals[63].values.at(i) * trace_evals[63].values.at(i))
                        + (trace_evals[64].values.at(i) * trace_evals[64].values.at(i))));
            numerators[i] += accum.random_coeff_powers[189]
                * (trace_evals[66].values.at(i)
                    - ((trace_evals[64].values.at(i) * trace_evals[64].values.at(i))
                        + (trace_evals[65].values.at(i) * trace_evals[65].values.at(i))));
            numerators[i] += accum.random_coeff_powers[188]
                * (trace_evals[67].values.at(i)
                    - ((trace_evals[65].values.at(i) * trace_evals[65].values.at(i))
                        + (trace_evals[66].values.at(i) * trace_evals[66].values.at(i))));
            numerators[i] += accum.random_coeff_powers[187]
                * (trace_evals[68].values.at(i)
                    - ((trace_evals[66].values.at(i) * trace_evals[66].values.at(i))
                        + (trace_evals[67].values.at(i) * trace_evals[67].values.at(i))));
            numerators[i] += accum.random_coeff_powers[186]
                * (trace_evals[69].values.at(i)
                    - ((trace_evals[67].values.at(i) * trace_evals[67].values.at(i))
                        + (trace_evals[68].values.at(i) * trace_evals[68].values.at(i))));
            numerators[i] += accum.random_coeff_powers[185]
                * (trace_evals[70].values.at(i)
                    - ((trace_evals[68].values.at(i) * trace_evals[68].values.at(i))
                        + (trace_evals[69].values.at(i) * trace_evals[69].values.at(i))));
            numerators[i] += accum.random_coeff_powers[184]
                * (trace_evals[71].values.at(i)
                    - ((trace_evals[69].values.at(i) * trace_evals[69].values.at(i))
                        + (trace_evals[70].values.at(i) * trace_evals[70].values.at(i))));
            numerators[i] += accum.random_coeff_powers[183]
                * (trace_evals[72].values.at(i)
                    - ((trace_evals[70].values.at(i) * trace_evals[70].values.at(i))
                        + (trace_evals[71].values.at(i) * trace_evals[71].values.at(i))));
            numerators[i] += accum.random_coeff_powers[182]
                * (trace_evals[73].values.at(i)
                    - ((trace_evals[71].values.at(i) * trace_evals[71].values.at(i))
                        + (trace_evals[72].values.at(i) * trace_evals[72].values.at(i))));
            numerators[i] += accum.random_coeff_powers[181]
                * (trace_evals[74].values.at(i)
                    - ((trace_evals[72].values.at(i) * trace_evals[72].values.at(i))
                        + (trace_evals[73].values.at(i) * trace_evals[73].values.at(i))));
            numerators[i] += accum.random_coeff_powers[180]
                * (trace_evals[75].values.at(i)
                    - ((trace_evals[73].values.at(i) * trace_evals[73].values.at(i))
                        + (trace_evals[74].values.at(i) * trace_evals[74].values.at(i))));
            numerators[i] += accum.random_coeff_powers[179]
                * (trace_evals[76].values.at(i)
                    - ((trace_evals[74].values.at(i) * trace_evals[74].values.at(i))
                        + (trace_evals[75].values.at(i) * trace_evals[75].values.at(i))));
            numerators[i] += accum.random_coeff_powers[178]
                * (trace_evals[77].values.at(i)
                    - ((trace_evals[75].values.at(i) * trace_evals[75].values.at(i))
                        + (trace_evals[76].values.at(i) * trace_evals[76].values.at(i))));
            numerators[i] += accum.random_coeff_powers[177]
                * (trace_evals[78].values.at(i)
                    - ((trace_evals[76].values.at(i) * trace_evals[76].values.at(i))
                        + (trace_evals[77].values.at(i) * trace_evals[77].values.at(i))));
            numerators[i] += accum.random_coeff_powers[176]
                * (trace_evals[79].values.at(i)
                    - ((trace_evals[77].values.at(i) * trace_evals[77].values.at(i))
                        + (trace_evals[78].values.at(i) * trace_evals[78].values.at(i))));
            numerators[i] += accum.random_coeff_powers[175]
                * (trace_evals[80].values.at(i)
                    - ((trace_evals[78].values.at(i) * trace_evals[78].values.at(i))
                        + (trace_evals[79].values.at(i) * trace_evals[79].values.at(i))));
            numerators[i] += accum.random_coeff_powers[174]
                * (trace_evals[81].values.at(i)
                    - ((trace_evals[79].values.at(i) * trace_evals[79].values.at(i))
                        + (trace_evals[80].values.at(i) * trace_evals[80].values.at(i))));
            numerators[i] += accum.random_coeff_powers[173]
                * (trace_evals[82].values.at(i)
                    - ((trace_evals[80].values.at(i) * trace_evals[80].values.at(i))
                        + (trace_evals[81].values.at(i) * trace_evals[81].values.at(i))));
            numerators[i] += accum.random_coeff_powers[172]
                * (trace_evals[83].values.at(i)
                    - ((trace_evals[81].values.at(i) * trace_evals[81].values.at(i))
                        + (trace_evals[82].values.at(i) * trace_evals[82].values.at(i))));
            numerators[i] += accum.random_coeff_powers[171]
                * (trace_evals[84].values.at(i)
                    - ((trace_evals[82].values.at(i) * trace_evals[82].values.at(i))
                        + (trace_evals[83].values.at(i) * trace_evals[83].values.at(i))));
            numerators[i] += accum.random_coeff_powers[170]
                * (trace_evals[85].values.at(i)
                    - ((trace_evals[83].values.at(i) * trace_evals[83].values.at(i))
                        + (trace_evals[84].values.at(i) * trace_evals[84].values.at(i))));
            numerators[i] += accum.random_coeff_powers[169]
                * (trace_evals[86].values.at(i)
                    - ((trace_evals[84].values.at(i) * trace_evals[84].values.at(i))
                        + (trace_evals[85].values.at(i) * trace_evals[85].values.at(i))));
            numerators[i] += accum.random_coeff_powers[168]
                * (trace_evals[87].values.at(i)
                    - ((trace_evals[85].values.at(i) * trace_evals[85].values.at(i))
                        + (trace_evals[86].values.at(i) * trace_evals[86].values.at(i))));
            numerators[i] += accum.random_coeff_powers[167]
                * (trace_evals[88].values.at(i)
                    - ((trace_evals[86].values.at(i) * trace_evals[86].values.at(i))
                        + (trace_evals[87].values.at(i) * trace_evals[87].values.at(i))));
            numerators[i] += accum.random_coeff_powers[166]
                * (trace_evals[89].values.at(i)
                    - ((trace_evals[87].values.at(i) * trace_evals[87].values.at(i))
                        + (trace_evals[88].values.at(i) * trace_evals[88].values.at(i))));
            numerators[i] += accum.random_coeff_powers[165]
                * (trace_evals[90].values.at(i)
                    - ((trace_evals[88].values.at(i) * trace_evals[88].values.at(i))
                        + (trace_evals[89].values.at(i) * trace_evals[89].values.at(i))));
            numerators[i] += accum.random_coeff_powers[164]
                * (trace_evals[91].values.at(i)
                    - ((trace_evals[89].values.at(i) * trace_evals[89].values.at(i))
                        + (trace_evals[90].values.at(i) * trace_evals[90].values.at(i))));
            numerators[i] += accum.random_coeff_powers[163]
                * (trace_evals[92].values.at(i)
                    - ((trace_evals[90].values.at(i) * trace_evals[90].values.at(i))
                        + (trace_evals[91].values.at(i) * trace_evals[91].values.at(i))));
            numerators[i] += accum.random_coeff_powers[162]
                * (trace_evals[93].values.at(i)
                    - ((trace_evals[91].values.at(i) * trace_evals[91].values.at(i))
                        + (trace_evals[92].values.at(i) * trace_evals[92].values.at(i))));
            numerators[i] += accum.random_coeff_powers[161]
                * (trace_evals[94].values.at(i)
                    - ((trace_evals[92].values.at(i) * trace_evals[92].values.at(i))
                        + (trace_evals[93].values.at(i) * trace_evals[93].values.at(i))));
            numerators[i] += accum.random_coeff_powers[160]
                * (trace_evals[95].values.at(i)
                    - ((trace_evals[93].values.at(i) * trace_evals[93].values.at(i))
                        + (trace_evals[94].values.at(i) * trace_evals[94].values.at(i))));
            numerators[i] += accum.random_coeff_powers[159]
                * (trace_evals[96].values.at(i)
                    - ((trace_evals[94].values.at(i) * trace_evals[94].values.at(i))
                        + (trace_evals[95].values.at(i) * trace_evals[95].values.at(i))));
            numerators[i] += accum.random_coeff_powers[158]
                * (trace_evals[97].values.at(i)
                    - ((trace_evals[95].values.at(i) * trace_evals[95].values.at(i))
                        + (trace_evals[96].values.at(i) * trace_evals[96].values.at(i))));
            numerators[i] += accum.random_coeff_powers[157]
                * (trace_evals[98].values.at(i)
                    - ((trace_evals[96].values.at(i) * trace_evals[96].values.at(i))
                        + (trace_evals[97].values.at(i) * trace_evals[97].values.at(i))));
            numerators[i] += accum.random_coeff_powers[156]
                * (trace_evals[99].values.at(i)
                    - ((trace_evals[97].values.at(i) * trace_evals[97].values.at(i))
                        + (trace_evals[98].values.at(i) * trace_evals[98].values.at(i))));
            numerators[i] += accum.random_coeff_powers[155]
                * (trace_evals[100].values.at(i)
                    - ((trace_evals[98].values.at(i) * trace_evals[98].values.at(i))
                        + (trace_evals[99].values.at(i) * trace_evals[99].values.at(i))));
            numerators[i] += accum.random_coeff_powers[154]
                * (trace_evals[101].values.at(i)
                    - ((trace_evals[99].values.at(i) * trace_evals[99].values.at(i))
                        + (trace_evals[100].values.at(i) * trace_evals[100].values.at(i))));
            numerators[i] += accum.random_coeff_powers[153]
                * (trace_evals[102].values.at(i)
                    - ((trace_evals[100].values.at(i) * trace_evals[100].values.at(i))
                        + (trace_evals[101].values.at(i) * trace_evals[101].values.at(i))));
            numerators[i] += accum.random_coeff_powers[152]
                * (trace_evals[103].values.at(i)
                    - ((trace_evals[101].values.at(i) * trace_evals[101].values.at(i))
                        + (trace_evals[102].values.at(i) * trace_evals[102].values.at(i))));
            numerators[i] += accum.random_coeff_powers[151]
                * (trace_evals[104].values.at(i)
                    - ((trace_evals[102].values.at(i) * trace_evals[102].values.at(i))
                        + (trace_evals[103].values.at(i) * trace_evals[103].values.at(i))));
            numerators[i] += accum.random_coeff_powers[150]
                * (trace_evals[105].values.at(i)
                    - ((trace_evals[103].values.at(i) * trace_evals[103].values.at(i))
                        + (trace_evals[104].values.at(i) * trace_evals[104].values.at(i))));
            numerators[i] += accum.random_coeff_powers[149]
                * (trace_evals[106].values.at(i)
                    - ((trace_evals[104].values.at(i) * trace_evals[104].values.at(i))
                        + (trace_evals[105].values.at(i) * trace_evals[105].values.at(i))));
            numerators[i] += accum.random_coeff_powers[148]
                * (trace_evals[107].values.at(i)
                    - ((trace_evals[105].values.at(i) * trace_evals[105].values.at(i))
                        + (trace_evals[106].values.at(i) * trace_evals[106].values.at(i))));
            numerators[i] += accum.random_coeff_powers[147]
                * (trace_evals[108].values.at(i)
                    - ((trace_evals[106].values.at(i) * trace_evals[106].values.at(i))
                        + (trace_evals[107].values.at(i) * trace_evals[107].values.at(i))));
            numerators[i] += accum.random_coeff_powers[146]
                * (trace_evals[109].values.at(i)
                    - ((trace_evals[107].values.at(i) * trace_evals[107].values.at(i))
                        + (trace_evals[108].values.at(i) * trace_evals[108].values.at(i))));
            numerators[i] += accum.random_coeff_powers[145]
                * (trace_evals[110].values.at(i)
                    - ((trace_evals[108].values.at(i) * trace_evals[108].values.at(i))
                        + (trace_evals[109].values.at(i) * trace_evals[109].values.at(i))));
            numerators[i] += accum.random_coeff_powers[144]
                * (trace_evals[111].values.at(i)
                    - ((trace_evals[109].values.at(i) * trace_evals[109].values.at(i))
                        + (trace_evals[110].values.at(i) * trace_evals[110].values.at(i))));
            numerators[i] += accum.random_coeff_powers[143]
                * (trace_evals[112].values.at(i)
                    - ((trace_evals[110].values.at(i) * trace_evals[110].values.at(i))
                        + (trace_evals[111].values.at(i) * trace_evals[111].values.at(i))));
            numerators[i] += accum.random_coeff_powers[142]
                * (trace_evals[113].values.at(i)
                    - ((trace_evals[111].values.at(i) * trace_evals[111].values.at(i))
                        + (trace_evals[112].values.at(i) * trace_evals[112].values.at(i))));
            numerators[i] += accum.random_coeff_powers[141]
                * (trace_evals[114].values.at(i)
                    - ((trace_evals[112].values.at(i) * trace_evals[112].values.at(i))
                        + (trace_evals[113].values.at(i) * trace_evals[113].values.at(i))));
            numerators[i] += accum.random_coeff_powers[140]
                * (trace_evals[115].values.at(i)
                    - ((trace_evals[113].values.at(i) * trace_evals[113].values.at(i))
                        + (trace_evals[114].values.at(i) * trace_evals[114].values.at(i))));
            numerators[i] += accum.random_coeff_powers[139]
                * (trace_evals[116].values.at(i)
                    - ((trace_evals[114].values.at(i) * trace_evals[114].values.at(i))
                        + (trace_evals[115].values.at(i) * trace_evals[115].values.at(i))));
            numerators[i] += accum.random_coeff_powers[138]
                * (trace_evals[117].values.at(i)
                    - ((trace_evals[115].values.at(i) * trace_evals[115].values.at(i))
                        + (trace_evals[116].values.at(i) * trace_evals[116].values.at(i))));
            numerators[i] += accum.random_coeff_powers[137]
                * (trace_evals[118].values.at(i)
                    - ((trace_evals[116].values.at(i) * trace_evals[116].values.at(i))
                        + (trace_evals[117].values.at(i) * trace_evals[117].values.at(i))));
            numerators[i] += accum.random_coeff_powers[136]
                * (trace_evals[119].values.at(i)
                    - ((trace_evals[117].values.at(i) * trace_evals[117].values.at(i))
                        + (trace_evals[118].values.at(i) * trace_evals[118].values.at(i))));
            numerators[i] += accum.random_coeff_powers[135]
                * (trace_evals[120].values.at(i)
                    - ((trace_evals[118].values.at(i) * trace_evals[118].values.at(i))
                        + (trace_evals[119].values.at(i) * trace_evals[119].values.at(i))));
            numerators[i] += accum.random_coeff_powers[134]
                * (trace_evals[121].values.at(i)
                    - ((trace_evals[119].values.at(i) * trace_evals[119].values.at(i))
                        + (trace_evals[120].values.at(i) * trace_evals[120].values.at(i))));
            numerators[i] += accum.random_coeff_powers[133]
                * (trace_evals[122].values.at(i)
                    - ((trace_evals[120].values.at(i) * trace_evals[120].values.at(i))
                        + (trace_evals[121].values.at(i) * trace_evals[121].values.at(i))));
            numerators[i] += accum.random_coeff_powers[132]
                * (trace_evals[123].values.at(i)
                    - ((trace_evals[121].values.at(i) * trace_evals[121].values.at(i))
                        + (trace_evals[122].values.at(i) * trace_evals[122].values.at(i))));
            numerators[i] += accum.random_coeff_powers[131]
                * (trace_evals[124].values.at(i)
                    - ((trace_evals[122].values.at(i) * trace_evals[122].values.at(i))
                        + (trace_evals[123].values.at(i) * trace_evals[123].values.at(i))));
            numerators[i] += accum.random_coeff_powers[130]
                * (trace_evals[125].values.at(i)
                    - ((trace_evals[123].values.at(i) * trace_evals[123].values.at(i))
                        + (trace_evals[124].values.at(i) * trace_evals[124].values.at(i))));
            numerators[i] += accum.random_coeff_powers[129]
                * (trace_evals[126].values.at(i)
                    - ((trace_evals[124].values.at(i) * trace_evals[124].values.at(i))
                        + (trace_evals[125].values.at(i) * trace_evals[125].values.at(i))));
            numerators[i] += accum.random_coeff_powers[128]
                * (trace_evals[127].values.at(i)
                    - ((trace_evals[125].values.at(i) * trace_evals[125].values.at(i))
                        + (trace_evals[126].values.at(i) * trace_evals[126].values.at(i))));
            numerators[i] += accum.random_coeff_powers[127]
                * (trace_evals[128].values.at(i)
                    - ((trace_evals[126].values.at(i) * trace_evals[126].values.at(i))
                        + (trace_evals[127].values.at(i) * trace_evals[127].values.at(i))));
            numerators[i] += accum.random_coeff_powers[126]
                * (trace_evals[129].values.at(i)
                    - ((trace_evals[127].values.at(i) * trace_evals[127].values.at(i))
                        + (trace_evals[128].values.at(i) * trace_evals[128].values.at(i))));
            numerators[i] += accum.random_coeff_powers[125]
                * (trace_evals[130].values.at(i)
                    - ((trace_evals[128].values.at(i) * trace_evals[128].values.at(i))
                        + (trace_evals[129].values.at(i) * trace_evals[129].values.at(i))));
            numerators[i] += accum.random_coeff_powers[124]
                * (trace_evals[131].values.at(i)
                    - ((trace_evals[129].values.at(i) * trace_evals[129].values.at(i))
                        + (trace_evals[130].values.at(i) * trace_evals[130].values.at(i))));
            numerators[i] += accum.random_coeff_powers[123]
                * (trace_evals[132].values.at(i)
                    - ((trace_evals[130].values.at(i) * trace_evals[130].values.at(i))
                        + (trace_evals[131].values.at(i) * trace_evals[131].values.at(i))));
            numerators[i] += accum.random_coeff_powers[122]
                * (trace_evals[133].values.at(i)
                    - ((trace_evals[131].values.at(i) * trace_evals[131].values.at(i))
                        + (trace_evals[132].values.at(i) * trace_evals[132].values.at(i))));
            numerators[i] += accum.random_coeff_powers[121]
                * (trace_evals[134].values.at(i)
                    - ((trace_evals[132].values.at(i) * trace_evals[132].values.at(i))
                        + (trace_evals[133].values.at(i) * trace_evals[133].values.at(i))));
            numerators[i] += accum.random_coeff_powers[120]
                * (trace_evals[135].values.at(i)
                    - ((trace_evals[133].values.at(i) * trace_evals[133].values.at(i))
                        + (trace_evals[134].values.at(i) * trace_evals[134].values.at(i))));
            numerators[i] += accum.random_coeff_powers[119]
                * (trace_evals[136].values.at(i)
                    - ((trace_evals[134].values.at(i) * trace_evals[134].values.at(i))
                        + (trace_evals[135].values.at(i) * trace_evals[135].values.at(i))));
            numerators[i] += accum.random_coeff_powers[118]
                * (trace_evals[137].values.at(i)
                    - ((trace_evals[135].values.at(i) * trace_evals[135].values.at(i))
                        + (trace_evals[136].values.at(i) * trace_evals[136].values.at(i))));
            numerators[i] += accum.random_coeff_powers[117]
                * (trace_evals[138].values.at(i)
                    - ((trace_evals[136].values.at(i) * trace_evals[136].values.at(i))
                        + (trace_evals[137].values.at(i) * trace_evals[137].values.at(i))));
            numerators[i] += accum.random_coeff_powers[116]
                * (trace_evals[139].values.at(i)
                    - ((trace_evals[137].values.at(i) * trace_evals[137].values.at(i))
                        + (trace_evals[138].values.at(i) * trace_evals[138].values.at(i))));
            numerators[i] += accum.random_coeff_powers[115]
                * (trace_evals[140].values.at(i)
                    - ((trace_evals[138].values.at(i) * trace_evals[138].values.at(i))
                        + (trace_evals[139].values.at(i) * trace_evals[139].values.at(i))));
            numerators[i] += accum.random_coeff_powers[114]
                * (trace_evals[141].values.at(i)
                    - ((trace_evals[139].values.at(i) * trace_evals[139].values.at(i))
                        + (trace_evals[140].values.at(i) * trace_evals[140].values.at(i))));
            numerators[i] += accum.random_coeff_powers[113]
                * (trace_evals[142].values.at(i)
                    - ((trace_evals[140].values.at(i) * trace_evals[140].values.at(i))
                        + (trace_evals[141].values.at(i) * trace_evals[141].values.at(i))));
            numerators[i] += accum.random_coeff_powers[112]
                * (trace_evals[143].values.at(i)
                    - ((trace_evals[141].values.at(i) * trace_evals[141].values.at(i))
                        + (trace_evals[142].values.at(i) * trace_evals[142].values.at(i))));
            numerators[i] += accum.random_coeff_powers[111]
                * (trace_evals[144].values.at(i)
                    - ((trace_evals[142].values.at(i) * trace_evals[142].values.at(i))
                        + (trace_evals[143].values.at(i) * trace_evals[143].values.at(i))));
            numerators[i] += accum.random_coeff_powers[110]
                * (trace_evals[145].values.at(i)
                    - ((trace_evals[143].values.at(i) * trace_evals[143].values.at(i))
                        + (trace_evals[144].values.at(i) * trace_evals[144].values.at(i))));
            numerators[i] += accum.random_coeff_powers[109]
                * (trace_evals[146].values.at(i)
                    - ((trace_evals[144].values.at(i) * trace_evals[144].values.at(i))
                        + (trace_evals[145].values.at(i) * trace_evals[145].values.at(i))));
            numerators[i] += accum.random_coeff_powers[108]
                * (trace_evals[147].values.at(i)
                    - ((trace_evals[145].values.at(i) * trace_evals[145].values.at(i))
                        + (trace_evals[146].values.at(i) * trace_evals[146].values.at(i))));
            numerators[i] += accum.random_coeff_powers[107]
                * (trace_evals[148].values.at(i)
                    - ((trace_evals[146].values.at(i) * trace_evals[146].values.at(i))
                        + (trace_evals[147].values.at(i) * trace_evals[147].values.at(i))));
            numerators[i] += accum.random_coeff_powers[106]
                * (trace_evals[149].values.at(i)
                    - ((trace_evals[147].values.at(i) * trace_evals[147].values.at(i))
                        + (trace_evals[148].values.at(i) * trace_evals[148].values.at(i))));
            numerators[i] += accum.random_coeff_powers[105]
                * (trace_evals[150].values.at(i)
                    - ((trace_evals[148].values.at(i) * trace_evals[148].values.at(i))
                        + (trace_evals[149].values.at(i) * trace_evals[149].values.at(i))));
            numerators[i] += accum.random_coeff_powers[104]
                * (trace_evals[151].values.at(i)
                    - ((trace_evals[149].values.at(i) * trace_evals[149].values.at(i))
                        + (trace_evals[150].values.at(i) * trace_evals[150].values.at(i))));
            numerators[i] += accum.random_coeff_powers[103]
                * (trace_evals[152].values.at(i)
                    - ((trace_evals[150].values.at(i) * trace_evals[150].values.at(i))
                        + (trace_evals[151].values.at(i) * trace_evals[151].values.at(i))));
            numerators[i] += accum.random_coeff_powers[102]
                * (trace_evals[153].values.at(i)
                    - ((trace_evals[151].values.at(i) * trace_evals[151].values.at(i))
                        + (trace_evals[152].values.at(i) * trace_evals[152].values.at(i))));
            numerators[i] += accum.random_coeff_powers[101]
                * (trace_evals[154].values.at(i)
                    - ((trace_evals[152].values.at(i) * trace_evals[152].values.at(i))
                        + (trace_evals[153].values.at(i) * trace_evals[153].values.at(i))));
            numerators[i] += accum.random_coeff_powers[100]
                * (trace_evals[155].values.at(i)
                    - ((trace_evals[153].values.at(i) * trace_evals[153].values.at(i))
                        + (trace_evals[154].values.at(i) * trace_evals[154].values.at(i))));
            numerators[i] += accum.random_coeff_powers[99]
                * (trace_evals[156].values.at(i)
                    - ((trace_evals[154].values.at(i) * trace_evals[154].values.at(i))
                        + (trace_evals[155].values.at(i) * trace_evals[155].values.at(i))));
            numerators[i] += accum.random_coeff_powers[98]
                * (trace_evals[157].values.at(i)
                    - ((trace_evals[155].values.at(i) * trace_evals[155].values.at(i))
                        + (trace_evals[156].values.at(i) * trace_evals[156].values.at(i))));
            numerators[i] += accum.random_coeff_powers[97]
                * (trace_evals[158].values.at(i)
                    - ((trace_evals[156].values.at(i) * trace_evals[156].values.at(i))
                        + (trace_evals[157].values.at(i) * trace_evals[157].values.at(i))));
            numerators[i] += accum.random_coeff_powers[96]
                * (trace_evals[159].values.at(i)
                    - ((trace_evals[157].values.at(i) * trace_evals[157].values.at(i))
                        + (trace_evals[158].values.at(i) * trace_evals[158].values.at(i))));
            numerators[i] += accum.random_coeff_powers[95]
                * (trace_evals[160].values.at(i)
                    - ((trace_evals[158].values.at(i) * trace_evals[158].values.at(i))
                        + (trace_evals[159].values.at(i) * trace_evals[159].values.at(i))));
            numerators[i] += accum.random_coeff_powers[94]
                * (trace_evals[161].values.at(i)
                    - ((trace_evals[159].values.at(i) * trace_evals[159].values.at(i))
                        + (trace_evals[160].values.at(i) * trace_evals[160].values.at(i))));
            numerators[i] += accum.random_coeff_powers[93]
                * (trace_evals[162].values.at(i)
                    - ((trace_evals[160].values.at(i) * trace_evals[160].values.at(i))
                        + (trace_evals[161].values.at(i) * trace_evals[161].values.at(i))));
            numerators[i] += accum.random_coeff_powers[92]
                * (trace_evals[163].values.at(i)
                    - ((trace_evals[161].values.at(i) * trace_evals[161].values.at(i))
                        + (trace_evals[162].values.at(i) * trace_evals[162].values.at(i))));
            numerators[i] += accum.random_coeff_powers[91]
                * (trace_evals[164].values.at(i)
                    - ((trace_evals[162].values.at(i) * trace_evals[162].values.at(i))
                        + (trace_evals[163].values.at(i) * trace_evals[163].values.at(i))));
            numerators[i] += accum.random_coeff_powers[90]
                * (trace_evals[165].values.at(i)
                    - ((trace_evals[163].values.at(i) * trace_evals[163].values.at(i))
                        + (trace_evals[164].values.at(i) * trace_evals[164].values.at(i))));
            numerators[i] += accum.random_coeff_powers[89]
                * (trace_evals[166].values.at(i)
                    - ((trace_evals[164].values.at(i) * trace_evals[164].values.at(i))
                        + (trace_evals[165].values.at(i) * trace_evals[165].values.at(i))));
            numerators[i] += accum.random_coeff_powers[88]
                * (trace_evals[167].values.at(i)
                    - ((trace_evals[165].values.at(i) * trace_evals[165].values.at(i))
                        + (trace_evals[166].values.at(i) * trace_evals[166].values.at(i))));
            numerators[i] += accum.random_coeff_powers[87]
                * (trace_evals[168].values.at(i)
                    - ((trace_evals[166].values.at(i) * trace_evals[166].values.at(i))
                        + (trace_evals[167].values.at(i) * trace_evals[167].values.at(i))));
            numerators[i] += accum.random_coeff_powers[86]
                * (trace_evals[169].values.at(i)
                    - ((trace_evals[167].values.at(i) * trace_evals[167].values.at(i))
                        + (trace_evals[168].values.at(i) * trace_evals[168].values.at(i))));
            numerators[i] += accum.random_coeff_powers[85]
                * (trace_evals[170].values.at(i)
                    - ((trace_evals[168].values.at(i) * trace_evals[168].values.at(i))
                        + (trace_evals[169].values.at(i) * trace_evals[169].values.at(i))));
            numerators[i] += accum.random_coeff_powers[84]
                * (trace_evals[171].values.at(i)
                    - ((trace_evals[169].values.at(i) * trace_evals[169].values.at(i))
                        + (trace_evals[170].values.at(i) * trace_evals[170].values.at(i))));
            numerators[i] += accum.random_coeff_powers[83]
                * (trace_evals[172].values.at(i)
                    - ((trace_evals[170].values.at(i) * trace_evals[170].values.at(i))
                        + (trace_evals[171].values.at(i) * trace_evals[171].values.at(i))));
            numerators[i] += accum.random_coeff_powers[82]
                * (trace_evals[173].values.at(i)
                    - ((trace_evals[171].values.at(i) * trace_evals[171].values.at(i))
                        + (trace_evals[172].values.at(i) * trace_evals[172].values.at(i))));
            numerators[i] += accum.random_coeff_powers[81]
                * (trace_evals[174].values.at(i)
                    - ((trace_evals[172].values.at(i) * trace_evals[172].values.at(i))
                        + (trace_evals[173].values.at(i) * trace_evals[173].values.at(i))));
            numerators[i] += accum.random_coeff_powers[80]
                * (trace_evals[175].values.at(i)
                    - ((trace_evals[173].values.at(i) * trace_evals[173].values.at(i))
                        + (trace_evals[174].values.at(i) * trace_evals[174].values.at(i))));
            numerators[i] += accum.random_coeff_powers[79]
                * (trace_evals[176].values.at(i)
                    - ((trace_evals[174].values.at(i) * trace_evals[174].values.at(i))
                        + (trace_evals[175].values.at(i) * trace_evals[175].values.at(i))));
            numerators[i] += accum.random_coeff_powers[78]
                * (trace_evals[177].values.at(i)
                    - ((trace_evals[175].values.at(i) * trace_evals[175].values.at(i))
                        + (trace_evals[176].values.at(i) * trace_evals[176].values.at(i))));
            numerators[i] += accum.random_coeff_powers[77]
                * (trace_evals[178].values.at(i)
                    - ((trace_evals[176].values.at(i) * trace_evals[176].values.at(i))
                        + (trace_evals[177].values.at(i) * trace_evals[177].values.at(i))));
            numerators[i] += accum.random_coeff_powers[76]
                * (trace_evals[179].values.at(i)
                    - ((trace_evals[177].values.at(i) * trace_evals[177].values.at(i))
                        + (trace_evals[178].values.at(i) * trace_evals[178].values.at(i))));
            numerators[i] += accum.random_coeff_powers[75]
                * (trace_evals[180].values.at(i)
                    - ((trace_evals[178].values.at(i) * trace_evals[178].values.at(i))
                        + (trace_evals[179].values.at(i) * trace_evals[179].values.at(i))));
            numerators[i] += accum.random_coeff_powers[74]
                * (trace_evals[181].values.at(i)
                    - ((trace_evals[179].values.at(i) * trace_evals[179].values.at(i))
                        + (trace_evals[180].values.at(i) * trace_evals[180].values.at(i))));
            numerators[i] += accum.random_coeff_powers[73]
                * (trace_evals[182].values.at(i)
                    - ((trace_evals[180].values.at(i) * trace_evals[180].values.at(i))
                        + (trace_evals[181].values.at(i) * trace_evals[181].values.at(i))));
            numerators[i] += accum.random_coeff_powers[72]
                * (trace_evals[183].values.at(i)
                    - ((trace_evals[181].values.at(i) * trace_evals[181].values.at(i))
                        + (trace_evals[182].values.at(i) * trace_evals[182].values.at(i))));
            numerators[i] += accum.random_coeff_powers[71]
                * (trace_evals[184].values.at(i)
                    - ((trace_evals[182].values.at(i) * trace_evals[182].values.at(i))
                        + (trace_evals[183].values.at(i) * trace_evals[183].values.at(i))));
            numerators[i] += accum.random_coeff_powers[70]
                * (trace_evals[185].values.at(i)
                    - ((trace_evals[183].values.at(i) * trace_evals[183].values.at(i))
                        + (trace_evals[184].values.at(i) * trace_evals[184].values.at(i))));
            numerators[i] += accum.random_coeff_powers[69]
                * (trace_evals[186].values.at(i)
                    - ((trace_evals[184].values.at(i) * trace_evals[184].values.at(i))
                        + (trace_evals[185].values.at(i) * trace_evals[185].values.at(i))));
            numerators[i] += accum.random_coeff_powers[68]
                * (trace_evals[187].values.at(i)
                    - ((trace_evals[185].values.at(i) * trace_evals[185].values.at(i))
                        + (trace_evals[186].values.at(i) * trace_evals[186].values.at(i))));
            numerators[i] += accum.random_coeff_powers[67]
                * (trace_evals[188].values.at(i)
                    - ((trace_evals[186].values.at(i) * trace_evals[186].values.at(i))
                        + (trace_evals[187].values.at(i) * trace_evals[187].values.at(i))));
            numerators[i] += accum.random_coeff_powers[66]
                * (trace_evals[189].values.at(i)
                    - ((trace_evals[187].values.at(i) * trace_evals[187].values.at(i))
                        + (trace_evals[188].values.at(i) * trace_evals[188].values.at(i))));
            numerators[i] += accum.random_coeff_powers[65]
                * (trace_evals[190].values.at(i)
                    - ((trace_evals[188].values.at(i) * trace_evals[188].values.at(i))
                        + (trace_evals[189].values.at(i) * trace_evals[189].values.at(i))));
            numerators[i] += accum.random_coeff_powers[64]
                * (trace_evals[191].values.at(i)
                    - ((trace_evals[189].values.at(i) * trace_evals[189].values.at(i))
                        + (trace_evals[190].values.at(i) * trace_evals[190].values.at(i))));
            numerators[i] += accum.random_coeff_powers[63]
                * (trace_evals[192].values.at(i)
                    - ((trace_evals[190].values.at(i) * trace_evals[190].values.at(i))
                        + (trace_evals[191].values.at(i) * trace_evals[191].values.at(i))));
            numerators[i] += accum.random_coeff_powers[62]
                * (trace_evals[193].values.at(i)
                    - ((trace_evals[191].values.at(i) * trace_evals[191].values.at(i))
                        + (trace_evals[192].values.at(i) * trace_evals[192].values.at(i))));
            numerators[i] += accum.random_coeff_powers[61]
                * (trace_evals[194].values.at(i)
                    - ((trace_evals[192].values.at(i) * trace_evals[192].values.at(i))
                        + (trace_evals[193].values.at(i) * trace_evals[193].values.at(i))));
            numerators[i] += accum.random_coeff_powers[60]
                * (trace_evals[195].values.at(i)
                    - ((trace_evals[193].values.at(i) * trace_evals[193].values.at(i))
                        + (trace_evals[194].values.at(i) * trace_evals[194].values.at(i))));
            numerators[i] += accum.random_coeff_powers[59]
                * (trace_evals[196].values.at(i)
                    - ((trace_evals[194].values.at(i) * trace_evals[194].values.at(i))
                        + (trace_evals[195].values.at(i) * trace_evals[195].values.at(i))));
            numerators[i] += accum.random_coeff_powers[58]
                * (trace_evals[197].values.at(i)
                    - ((trace_evals[195].values.at(i) * trace_evals[195].values.at(i))
                        + (trace_evals[196].values.at(i) * trace_evals[196].values.at(i))));
            numerators[i] += accum.random_coeff_powers[57]
                * (trace_evals[198].values.at(i)
                    - ((trace_evals[196].values.at(i) * trace_evals[196].values.at(i))
                        + (trace_evals[197].values.at(i) * trace_evals[197].values.at(i))));
            numerators[i] += accum.random_coeff_powers[56]
                * (trace_evals[199].values.at(i)
                    - ((trace_evals[197].values.at(i) * trace_evals[197].values.at(i))
                        + (trace_evals[198].values.at(i) * trace_evals[198].values.at(i))));
            numerators[i] += accum.random_coeff_powers[55]
                * (trace_evals[200].values.at(i)
                    - ((trace_evals[198].values.at(i) * trace_evals[198].values.at(i))
                        + (trace_evals[199].values.at(i) * trace_evals[199].values.at(i))));
            numerators[i] += accum.random_coeff_powers[54]
                * (trace_evals[201].values.at(i)
                    - ((trace_evals[199].values.at(i) * trace_evals[199].values.at(i))
                        + (trace_evals[200].values.at(i) * trace_evals[200].values.at(i))));
            numerators[i] += accum.random_coeff_powers[53]
                * (trace_evals[202].values.at(i)
                    - ((trace_evals[200].values.at(i) * trace_evals[200].values.at(i))
                        + (trace_evals[201].values.at(i) * trace_evals[201].values.at(i))));
            numerators[i] += accum.random_coeff_powers[52]
                * (trace_evals[203].values.at(i)
                    - ((trace_evals[201].values.at(i) * trace_evals[201].values.at(i))
                        + (trace_evals[202].values.at(i) * trace_evals[202].values.at(i))));
            numerators[i] += accum.random_coeff_powers[51]
                * (trace_evals[204].values.at(i)
                    - ((trace_evals[202].values.at(i) * trace_evals[202].values.at(i))
                        + (trace_evals[203].values.at(i) * trace_evals[203].values.at(i))));
            numerators[i] += accum.random_coeff_powers[50]
                * (trace_evals[205].values.at(i)
                    - ((trace_evals[203].values.at(i) * trace_evals[203].values.at(i))
                        + (trace_evals[204].values.at(i) * trace_evals[204].values.at(i))));
            numerators[i] += accum.random_coeff_powers[49]
                * (trace_evals[206].values.at(i)
                    - ((trace_evals[204].values.at(i) * trace_evals[204].values.at(i))
                        + (trace_evals[205].values.at(i) * trace_evals[205].values.at(i))));
            numerators[i] += accum.random_coeff_powers[48]
                * (trace_evals[207].values.at(i)
                    - ((trace_evals[205].values.at(i) * trace_evals[205].values.at(i))
                        + (trace_evals[206].values.at(i) * trace_evals[206].values.at(i))));
            numerators[i] += accum.random_coeff_powers[47]
                * (trace_evals[208].values.at(i)
                    - ((trace_evals[206].values.at(i) * trace_evals[206].values.at(i))
                        + (trace_evals[207].values.at(i) * trace_evals[207].values.at(i))));
            numerators[i] += accum.random_coeff_powers[46]
                * (trace_evals[209].values.at(i)
                    - ((trace_evals[207].values.at(i) * trace_evals[207].values.at(i))
                        + (trace_evals[208].values.at(i) * trace_evals[208].values.at(i))));
            numerators[i] += accum.random_coeff_powers[45]
                * (trace_evals[210].values.at(i)
                    - ((trace_evals[208].values.at(i) * trace_evals[208].values.at(i))
                        + (trace_evals[209].values.at(i) * trace_evals[209].values.at(i))));
            numerators[i] += accum.random_coeff_powers[44]
                * (trace_evals[211].values.at(i)
                    - ((trace_evals[209].values.at(i) * trace_evals[209].values.at(i))
                        + (trace_evals[210].values.at(i) * trace_evals[210].values.at(i))));
            numerators[i] += accum.random_coeff_powers[43]
                * (trace_evals[212].values.at(i)
                    - ((trace_evals[210].values.at(i) * trace_evals[210].values.at(i))
                        + (trace_evals[211].values.at(i) * trace_evals[211].values.at(i))));
            numerators[i] += accum.random_coeff_powers[42]
                * (trace_evals[213].values.at(i)
                    - ((trace_evals[211].values.at(i) * trace_evals[211].values.at(i))
                        + (trace_evals[212].values.at(i) * trace_evals[212].values.at(i))));
            numerators[i] += accum.random_coeff_powers[41]
                * (trace_evals[214].values.at(i)
                    - ((trace_evals[212].values.at(i) * trace_evals[212].values.at(i))
                        + (trace_evals[213].values.at(i) * trace_evals[213].values.at(i))));
            numerators[i] += accum.random_coeff_powers[40]
                * (trace_evals[215].values.at(i)
                    - ((trace_evals[213].values.at(i) * trace_evals[213].values.at(i))
                        + (trace_evals[214].values.at(i) * trace_evals[214].values.at(i))));
            numerators[i] += accum.random_coeff_powers[39]
                * (trace_evals[216].values.at(i)
                    - ((trace_evals[214].values.at(i) * trace_evals[214].values.at(i))
                        + (trace_evals[215].values.at(i) * trace_evals[215].values.at(i))));
            numerators[i] += accum.random_coeff_powers[38]
                * (trace_evals[217].values.at(i)
                    - ((trace_evals[215].values.at(i) * trace_evals[215].values.at(i))
                        + (trace_evals[216].values.at(i) * trace_evals[216].values.at(i))));
            numerators[i] += accum.random_coeff_powers[37]
                * (trace_evals[218].values.at(i)
                    - ((trace_evals[216].values.at(i) * trace_evals[216].values.at(i))
                        + (trace_evals[217].values.at(i) * trace_evals[217].values.at(i))));
            numerators[i] += accum.random_coeff_powers[36]
                * (trace_evals[219].values.at(i)
                    - ((trace_evals[217].values.at(i) * trace_evals[217].values.at(i))
                        + (trace_evals[218].values.at(i) * trace_evals[218].values.at(i))));
            numerators[i] += accum.random_coeff_powers[35]
                * (trace_evals[220].values.at(i)
                    - ((trace_evals[218].values.at(i) * trace_evals[218].values.at(i))
                        + (trace_evals[219].values.at(i) * trace_evals[219].values.at(i))));
            numerators[i] += accum.random_coeff_powers[34]
                * (trace_evals[221].values.at(i)
                    - ((trace_evals[219].values.at(i) * trace_evals[219].values.at(i))
                        + (trace_evals[220].values.at(i) * trace_evals[220].values.at(i))));
            numerators[i] += accum.random_coeff_powers[33]
                * (trace_evals[222].values.at(i)
                    - ((trace_evals[220].values.at(i) * trace_evals[220].values.at(i))
                        + (trace_evals[221].values.at(i) * trace_evals[221].values.at(i))));
            numerators[i] += accum.random_coeff_powers[32]
                * (trace_evals[223].values.at(i)
                    - ((trace_evals[221].values.at(i) * trace_evals[221].values.at(i))
                        + (trace_evals[222].values.at(i) * trace_evals[222].values.at(i))));
            numerators[i] += accum.random_coeff_powers[31]
                * (trace_evals[224].values.at(i)
                    - ((trace_evals[222].values.at(i) * trace_evals[222].values.at(i))
                        + (trace_evals[223].values.at(i) * trace_evals[223].values.at(i))));
            numerators[i] += accum.random_coeff_powers[30]
                * (trace_evals[225].values.at(i)
                    - ((trace_evals[223].values.at(i) * trace_evals[223].values.at(i))
                        + (trace_evals[224].values.at(i) * trace_evals[224].values.at(i))));
            numerators[i] += accum.random_coeff_powers[29]
                * (trace_evals[226].values.at(i)
                    - ((trace_evals[224].values.at(i) * trace_evals[224].values.at(i))
                        + (trace_evals[225].values.at(i) * trace_evals[225].values.at(i))));
            numerators[i] += accum.random_coeff_powers[28]
                * (trace_evals[227].values.at(i)
                    - ((trace_evals[225].values.at(i) * trace_evals[225].values.at(i))
                        + (trace_evals[226].values.at(i) * trace_evals[226].values.at(i))));
            numerators[i] += accum.random_coeff_powers[27]
                * (trace_evals[228].values.at(i)
                    - ((trace_evals[226].values.at(i) * trace_evals[226].values.at(i))
                        + (trace_evals[227].values.at(i) * trace_evals[227].values.at(i))));
            numerators[i] += accum.random_coeff_powers[26]
                * (trace_evals[229].values.at(i)
                    - ((trace_evals[227].values.at(i) * trace_evals[227].values.at(i))
                        + (trace_evals[228].values.at(i) * trace_evals[228].values.at(i))));
            numerators[i] += accum.random_coeff_powers[25]
                * (trace_evals[230].values.at(i)
                    - ((trace_evals[228].values.at(i) * trace_evals[228].values.at(i))
                        + (trace_evals[229].values.at(i) * trace_evals[229].values.at(i))));
            numerators[i] += accum.random_coeff_powers[24]
                * (trace_evals[231].values.at(i)
                    - ((trace_evals[229].values.at(i) * trace_evals[229].values.at(i))
                        + (trace_evals[230].values.at(i) * trace_evals[230].values.at(i))));
            numerators[i] += accum.random_coeff_powers[23]
                * (trace_evals[232].values.at(i)
                    - ((trace_evals[230].values.at(i) * trace_evals[230].values.at(i))
                        + (trace_evals[231].values.at(i) * trace_evals[231].values.at(i))));
            numerators[i] += accum.random_coeff_powers[22]
                * (trace_evals[233].values.at(i)
                    - ((trace_evals[231].values.at(i) * trace_evals[231].values.at(i))
                        + (trace_evals[232].values.at(i) * trace_evals[232].values.at(i))));
            numerators[i] += accum.random_coeff_powers[21]
                * (trace_evals[234].values.at(i)
                    - ((trace_evals[232].values.at(i) * trace_evals[232].values.at(i))
                        + (trace_evals[233].values.at(i) * trace_evals[233].values.at(i))));
            numerators[i] += accum.random_coeff_powers[20]
                * (trace_evals[235].values.at(i)
                    - ((trace_evals[233].values.at(i) * trace_evals[233].values.at(i))
                        + (trace_evals[234].values.at(i) * trace_evals[234].values.at(i))));
            numerators[i] += accum.random_coeff_powers[19]
                * (trace_evals[236].values.at(i)
                    - ((trace_evals[234].values.at(i) * trace_evals[234].values.at(i))
                        + (trace_evals[235].values.at(i) * trace_evals[235].values.at(i))));
            numerators[i] += accum.random_coeff_powers[18]
                * (trace_evals[237].values.at(i)
                    - ((trace_evals[235].values.at(i) * trace_evals[235].values.at(i))
                        + (trace_evals[236].values.at(i) * trace_evals[236].values.at(i))));
            numerators[i] += accum.random_coeff_powers[17]
                * (trace_evals[238].values.at(i)
                    - ((trace_evals[236].values.at(i) * trace_evals[236].values.at(i))
                        + (trace_evals[237].values.at(i) * trace_evals[237].values.at(i))));
            numerators[i] += accum.random_coeff_powers[16]
                * (trace_evals[239].values.at(i)
                    - ((trace_evals[237].values.at(i) * trace_evals[237].values.at(i))
                        + (trace_evals[238].values.at(i) * trace_evals[238].values.at(i))));
            numerators[i] += accum.random_coeff_powers[15]
                * (trace_evals[240].values.at(i)
                    - ((trace_evals[238].values.at(i) * trace_evals[238].values.at(i))
                        + (trace_evals[239].values.at(i) * trace_evals[239].values.at(i))));
            numerators[i] += accum.random_coeff_powers[14]
                * (trace_evals[241].values.at(i)
                    - ((trace_evals[239].values.at(i) * trace_evals[239].values.at(i))
                        + (trace_evals[240].values.at(i) * trace_evals[240].values.at(i))));
            numerators[i] += accum.random_coeff_powers[13]
                * (trace_evals[242].values.at(i)
                    - ((trace_evals[240].values.at(i) * trace_evals[240].values.at(i))
                        + (trace_evals[241].values.at(i) * trace_evals[241].values.at(i))));
            numerators[i] += accum.random_coeff_powers[12]
                * (trace_evals[243].values.at(i)
                    - ((trace_evals[241].values.at(i) * trace_evals[241].values.at(i))
                        + (trace_evals[242].values.at(i) * trace_evals[242].values.at(i))));
            numerators[i] += accum.random_coeff_powers[11]
                * (trace_evals[244].values.at(i)
                    - ((trace_evals[242].values.at(i) * trace_evals[242].values.at(i))
                        + (trace_evals[243].values.at(i) * trace_evals[243].values.at(i))));
            numerators[i] += accum.random_coeff_powers[10]
                * (trace_evals[245].values.at(i)
                    - ((trace_evals[243].values.at(i) * trace_evals[243].values.at(i))
                        + (trace_evals[244].values.at(i) * trace_evals[244].values.at(i))));
            numerators[i] += accum.random_coeff_powers[9]
                * (trace_evals[246].values.at(i)
                    - ((trace_evals[244].values.at(i) * trace_evals[244].values.at(i))
                        + (trace_evals[245].values.at(i) * trace_evals[245].values.at(i))));
            numerators[i] += accum.random_coeff_powers[8]
                * (trace_evals[247].values.at(i)
                    - ((trace_evals[245].values.at(i) * trace_evals[245].values.at(i))
                        + (trace_evals[246].values.at(i) * trace_evals[246].values.at(i))));
            numerators[i] += accum.random_coeff_powers[7]
                * (trace_evals[248].values.at(i)
                    - ((trace_evals[246].values.at(i) * trace_evals[246].values.at(i))
                        + (trace_evals[247].values.at(i) * trace_evals[247].values.at(i))));
            numerators[i] += accum.random_coeff_powers[6]
                * (trace_evals[249].values.at(i)
                    - ((trace_evals[247].values.at(i) * trace_evals[247].values.at(i))
                        + (trace_evals[248].values.at(i) * trace_evals[248].values.at(i))));
            numerators[i] += accum.random_coeff_powers[5]
                * (trace_evals[250].values.at(i)
                    - ((trace_evals[248].values.at(i) * trace_evals[248].values.at(i))
                        + (trace_evals[249].values.at(i) * trace_evals[249].values.at(i))));
            numerators[i] += accum.random_coeff_powers[4]
                * (trace_evals[251].values.at(i)
                    - ((trace_evals[249].values.at(i) * trace_evals[249].values.at(i))
                        + (trace_evals[250].values.at(i) * trace_evals[250].values.at(i))));
            numerators[i] += accum.random_coeff_powers[3]
                * (trace_evals[252].values.at(i)
                    - ((trace_evals[250].values.at(i) * trace_evals[250].values.at(i))
                        + (trace_evals[251].values.at(i) * trace_evals[251].values.at(i))));
            numerators[i] += accum.random_coeff_powers[2]
                * (trace_evals[253].values.at(i)
                    - ((trace_evals[251].values.at(i) * trace_evals[251].values.at(i))
                        + (trace_evals[252].values.at(i) * trace_evals[252].values.at(i))));
            numerators[i] += accum.random_coeff_powers[1]
                * (trace_evals[254].values.at(i)
                    - ((trace_evals[252].values.at(i) * trace_evals[252].values.at(i))
                        + (trace_evals[253].values.at(i) * trace_evals[253].values.at(i))));
            numerators[i] += accum.random_coeff_powers[0]
                * (trace_evals[255].values.at(i)
                    - ((trace_evals[253].values.at(i) * trace_evals[253].values.at(i))
                        + (trace_evals[254].values.at(i) * trace_evals[254].values.at(i))));
        }
        for (i, (num, denom)) in numerators.iter().zip(denom_inverses.iter()).enumerate() {
            accum.accumulate(i, *num * *denom);
        }
    }
    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> ColumnVec<Vec<CirclePoint<SecureField>>> {
        let mask = Mask(vec![vec![0_usize]; 256]);
        mask.iter()
            .map(|col| col.iter().map(|_| point).collect())
            .collect()
    }
    #[allow(unused_parens)]
    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &ColumnVec<Vec<SecureField>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    ) {
        let constraint_zero_domain = CanonicCoset::new(self.log_size).coset;
        let denominator = coset_vanishing(constraint_zero_domain, point);
        let numerator = (mask[0][0] - BaseField::from_u32_unchecked(1));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[2][0] - ((mask[0][0] * mask[0][0]) + (mask[1][0] * mask[1][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[3][0] - ((mask[1][0] * mask[1][0]) + (mask[2][0] * mask[2][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[4][0] - ((mask[2][0] * mask[2][0]) + (mask[3][0] * mask[3][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[5][0] - ((mask[3][0] * mask[3][0]) + (mask[4][0] * mask[4][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[6][0] - ((mask[4][0] * mask[4][0]) + (mask[5][0] * mask[5][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[7][0] - ((mask[5][0] * mask[5][0]) + (mask[6][0] * mask[6][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[8][0] - ((mask[6][0] * mask[6][0]) + (mask[7][0] * mask[7][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[9][0] - ((mask[7][0] * mask[7][0]) + (mask[8][0] * mask[8][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[10][0] - ((mask[8][0] * mask[8][0]) + (mask[9][0] * mask[9][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[11][0] - ((mask[9][0] * mask[9][0]) + (mask[10][0] * mask[10][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[12][0] - ((mask[10][0] * mask[10][0]) + (mask[11][0] * mask[11][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[13][0] - ((mask[11][0] * mask[11][0]) + (mask[12][0] * mask[12][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[14][0] - ((mask[12][0] * mask[12][0]) + (mask[13][0] * mask[13][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[15][0] - ((mask[13][0] * mask[13][0]) + (mask[14][0] * mask[14][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[16][0] - ((mask[14][0] * mask[14][0]) + (mask[15][0] * mask[15][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[17][0] - ((mask[15][0] * mask[15][0]) + (mask[16][0] * mask[16][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[18][0] - ((mask[16][0] * mask[16][0]) + (mask[17][0] * mask[17][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[19][0] - ((mask[17][0] * mask[17][0]) + (mask[18][0] * mask[18][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[20][0] - ((mask[18][0] * mask[18][0]) + (mask[19][0] * mask[19][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[21][0] - ((mask[19][0] * mask[19][0]) + (mask[20][0] * mask[20][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[22][0] - ((mask[20][0] * mask[20][0]) + (mask[21][0] * mask[21][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[23][0] - ((mask[21][0] * mask[21][0]) + (mask[22][0] * mask[22][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[24][0] - ((mask[22][0] * mask[22][0]) + (mask[23][0] * mask[23][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[25][0] - ((mask[23][0] * mask[23][0]) + (mask[24][0] * mask[24][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[26][0] - ((mask[24][0] * mask[24][0]) + (mask[25][0] * mask[25][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[27][0] - ((mask[25][0] * mask[25][0]) + (mask[26][0] * mask[26][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[28][0] - ((mask[26][0] * mask[26][0]) + (mask[27][0] * mask[27][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[29][0] - ((mask[27][0] * mask[27][0]) + (mask[28][0] * mask[28][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[30][0] - ((mask[28][0] * mask[28][0]) + (mask[29][0] * mask[29][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[31][0] - ((mask[29][0] * mask[29][0]) + (mask[30][0] * mask[30][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[32][0] - ((mask[30][0] * mask[30][0]) + (mask[31][0] * mask[31][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[33][0] - ((mask[31][0] * mask[31][0]) + (mask[32][0] * mask[32][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[34][0] - ((mask[32][0] * mask[32][0]) + (mask[33][0] * mask[33][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[35][0] - ((mask[33][0] * mask[33][0]) + (mask[34][0] * mask[34][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[36][0] - ((mask[34][0] * mask[34][0]) + (mask[35][0] * mask[35][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[37][0] - ((mask[35][0] * mask[35][0]) + (mask[36][0] * mask[36][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[38][0] - ((mask[36][0] * mask[36][0]) + (mask[37][0] * mask[37][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[39][0] - ((mask[37][0] * mask[37][0]) + (mask[38][0] * mask[38][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[40][0] - ((mask[38][0] * mask[38][0]) + (mask[39][0] * mask[39][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[41][0] - ((mask[39][0] * mask[39][0]) + (mask[40][0] * mask[40][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[42][0] - ((mask[40][0] * mask[40][0]) + (mask[41][0] * mask[41][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[43][0] - ((mask[41][0] * mask[41][0]) + (mask[42][0] * mask[42][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[44][0] - ((mask[42][0] * mask[42][0]) + (mask[43][0] * mask[43][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[45][0] - ((mask[43][0] * mask[43][0]) + (mask[44][0] * mask[44][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[46][0] - ((mask[44][0] * mask[44][0]) + (mask[45][0] * mask[45][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[47][0] - ((mask[45][0] * mask[45][0]) + (mask[46][0] * mask[46][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[48][0] - ((mask[46][0] * mask[46][0]) + (mask[47][0] * mask[47][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[49][0] - ((mask[47][0] * mask[47][0]) + (mask[48][0] * mask[48][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[50][0] - ((mask[48][0] * mask[48][0]) + (mask[49][0] * mask[49][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[51][0] - ((mask[49][0] * mask[49][0]) + (mask[50][0] * mask[50][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[52][0] - ((mask[50][0] * mask[50][0]) + (mask[51][0] * mask[51][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[53][0] - ((mask[51][0] * mask[51][0]) + (mask[52][0] * mask[52][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[54][0] - ((mask[52][0] * mask[52][0]) + (mask[53][0] * mask[53][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[55][0] - ((mask[53][0] * mask[53][0]) + (mask[54][0] * mask[54][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[56][0] - ((mask[54][0] * mask[54][0]) + (mask[55][0] * mask[55][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[57][0] - ((mask[55][0] * mask[55][0]) + (mask[56][0] * mask[56][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[58][0] - ((mask[56][0] * mask[56][0]) + (mask[57][0] * mask[57][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[59][0] - ((mask[57][0] * mask[57][0]) + (mask[58][0] * mask[58][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[60][0] - ((mask[58][0] * mask[58][0]) + (mask[59][0] * mask[59][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[61][0] - ((mask[59][0] * mask[59][0]) + (mask[60][0] * mask[60][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[62][0] - ((mask[60][0] * mask[60][0]) + (mask[61][0] * mask[61][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[63][0] - ((mask[61][0] * mask[61][0]) + (mask[62][0] * mask[62][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[64][0] - ((mask[62][0] * mask[62][0]) + (mask[63][0] * mask[63][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[65][0] - ((mask[63][0] * mask[63][0]) + (mask[64][0] * mask[64][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[66][0] - ((mask[64][0] * mask[64][0]) + (mask[65][0] * mask[65][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[67][0] - ((mask[65][0] * mask[65][0]) + (mask[66][0] * mask[66][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[68][0] - ((mask[66][0] * mask[66][0]) + (mask[67][0] * mask[67][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[69][0] - ((mask[67][0] * mask[67][0]) + (mask[68][0] * mask[68][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[70][0] - ((mask[68][0] * mask[68][0]) + (mask[69][0] * mask[69][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[71][0] - ((mask[69][0] * mask[69][0]) + (mask[70][0] * mask[70][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[72][0] - ((mask[70][0] * mask[70][0]) + (mask[71][0] * mask[71][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[73][0] - ((mask[71][0] * mask[71][0]) + (mask[72][0] * mask[72][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[74][0] - ((mask[72][0] * mask[72][0]) + (mask[73][0] * mask[73][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[75][0] - ((mask[73][0] * mask[73][0]) + (mask[74][0] * mask[74][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[76][0] - ((mask[74][0] * mask[74][0]) + (mask[75][0] * mask[75][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[77][0] - ((mask[75][0] * mask[75][0]) + (mask[76][0] * mask[76][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[78][0] - ((mask[76][0] * mask[76][0]) + (mask[77][0] * mask[77][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[79][0] - ((mask[77][0] * mask[77][0]) + (mask[78][0] * mask[78][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[80][0] - ((mask[78][0] * mask[78][0]) + (mask[79][0] * mask[79][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[81][0] - ((mask[79][0] * mask[79][0]) + (mask[80][0] * mask[80][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[82][0] - ((mask[80][0] * mask[80][0]) + (mask[81][0] * mask[81][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[83][0] - ((mask[81][0] * mask[81][0]) + (mask[82][0] * mask[82][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[84][0] - ((mask[82][0] * mask[82][0]) + (mask[83][0] * mask[83][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[85][0] - ((mask[83][0] * mask[83][0]) + (mask[84][0] * mask[84][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[86][0] - ((mask[84][0] * mask[84][0]) + (mask[85][0] * mask[85][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[87][0] - ((mask[85][0] * mask[85][0]) + (mask[86][0] * mask[86][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[88][0] - ((mask[86][0] * mask[86][0]) + (mask[87][0] * mask[87][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[89][0] - ((mask[87][0] * mask[87][0]) + (mask[88][0] * mask[88][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[90][0] - ((mask[88][0] * mask[88][0]) + (mask[89][0] * mask[89][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[91][0] - ((mask[89][0] * mask[89][0]) + (mask[90][0] * mask[90][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[92][0] - ((mask[90][0] * mask[90][0]) + (mask[91][0] * mask[91][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[93][0] - ((mask[91][0] * mask[91][0]) + (mask[92][0] * mask[92][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[94][0] - ((mask[92][0] * mask[92][0]) + (mask[93][0] * mask[93][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[95][0] - ((mask[93][0] * mask[93][0]) + (mask[94][0] * mask[94][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[96][0] - ((mask[94][0] * mask[94][0]) + (mask[95][0] * mask[95][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[97][0] - ((mask[95][0] * mask[95][0]) + (mask[96][0] * mask[96][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[98][0] - ((mask[96][0] * mask[96][0]) + (mask[97][0] * mask[97][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator = (mask[99][0] - ((mask[97][0] * mask[97][0]) + (mask[98][0] * mask[98][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[100][0] - ((mask[98][0] * mask[98][0]) + (mask[99][0] * mask[99][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[101][0] - ((mask[99][0] * mask[99][0]) + (mask[100][0] * mask[100][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[102][0] - ((mask[100][0] * mask[100][0]) + (mask[101][0] * mask[101][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[103][0] - ((mask[101][0] * mask[101][0]) + (mask[102][0] * mask[102][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[104][0] - ((mask[102][0] * mask[102][0]) + (mask[103][0] * mask[103][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[105][0] - ((mask[103][0] * mask[103][0]) + (mask[104][0] * mask[104][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[106][0] - ((mask[104][0] * mask[104][0]) + (mask[105][0] * mask[105][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[107][0] - ((mask[105][0] * mask[105][0]) + (mask[106][0] * mask[106][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[108][0] - ((mask[106][0] * mask[106][0]) + (mask[107][0] * mask[107][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[109][0] - ((mask[107][0] * mask[107][0]) + (mask[108][0] * mask[108][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[110][0] - ((mask[108][0] * mask[108][0]) + (mask[109][0] * mask[109][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[111][0] - ((mask[109][0] * mask[109][0]) + (mask[110][0] * mask[110][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[112][0] - ((mask[110][0] * mask[110][0]) + (mask[111][0] * mask[111][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[113][0] - ((mask[111][0] * mask[111][0]) + (mask[112][0] * mask[112][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[114][0] - ((mask[112][0] * mask[112][0]) + (mask[113][0] * mask[113][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[115][0] - ((mask[113][0] * mask[113][0]) + (mask[114][0] * mask[114][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[116][0] - ((mask[114][0] * mask[114][0]) + (mask[115][0] * mask[115][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[117][0] - ((mask[115][0] * mask[115][0]) + (mask[116][0] * mask[116][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[118][0] - ((mask[116][0] * mask[116][0]) + (mask[117][0] * mask[117][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[119][0] - ((mask[117][0] * mask[117][0]) + (mask[118][0] * mask[118][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[120][0] - ((mask[118][0] * mask[118][0]) + (mask[119][0] * mask[119][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[121][0] - ((mask[119][0] * mask[119][0]) + (mask[120][0] * mask[120][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[122][0] - ((mask[120][0] * mask[120][0]) + (mask[121][0] * mask[121][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[123][0] - ((mask[121][0] * mask[121][0]) + (mask[122][0] * mask[122][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[124][0] - ((mask[122][0] * mask[122][0]) + (mask[123][0] * mask[123][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[125][0] - ((mask[123][0] * mask[123][0]) + (mask[124][0] * mask[124][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[126][0] - ((mask[124][0] * mask[124][0]) + (mask[125][0] * mask[125][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[127][0] - ((mask[125][0] * mask[125][0]) + (mask[126][0] * mask[126][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[128][0] - ((mask[126][0] * mask[126][0]) + (mask[127][0] * mask[127][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[129][0] - ((mask[127][0] * mask[127][0]) + (mask[128][0] * mask[128][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[130][0] - ((mask[128][0] * mask[128][0]) + (mask[129][0] * mask[129][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[131][0] - ((mask[129][0] * mask[129][0]) + (mask[130][0] * mask[130][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[132][0] - ((mask[130][0] * mask[130][0]) + (mask[131][0] * mask[131][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[133][0] - ((mask[131][0] * mask[131][0]) + (mask[132][0] * mask[132][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[134][0] - ((mask[132][0] * mask[132][0]) + (mask[133][0] * mask[133][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[135][0] - ((mask[133][0] * mask[133][0]) + (mask[134][0] * mask[134][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[136][0] - ((mask[134][0] * mask[134][0]) + (mask[135][0] * mask[135][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[137][0] - ((mask[135][0] * mask[135][0]) + (mask[136][0] * mask[136][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[138][0] - ((mask[136][0] * mask[136][0]) + (mask[137][0] * mask[137][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[139][0] - ((mask[137][0] * mask[137][0]) + (mask[138][0] * mask[138][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[140][0] - ((mask[138][0] * mask[138][0]) + (mask[139][0] * mask[139][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[141][0] - ((mask[139][0] * mask[139][0]) + (mask[140][0] * mask[140][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[142][0] - ((mask[140][0] * mask[140][0]) + (mask[141][0] * mask[141][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[143][0] - ((mask[141][0] * mask[141][0]) + (mask[142][0] * mask[142][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[144][0] - ((mask[142][0] * mask[142][0]) + (mask[143][0] * mask[143][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[145][0] - ((mask[143][0] * mask[143][0]) + (mask[144][0] * mask[144][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[146][0] - ((mask[144][0] * mask[144][0]) + (mask[145][0] * mask[145][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[147][0] - ((mask[145][0] * mask[145][0]) + (mask[146][0] * mask[146][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[148][0] - ((mask[146][0] * mask[146][0]) + (mask[147][0] * mask[147][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[149][0] - ((mask[147][0] * mask[147][0]) + (mask[148][0] * mask[148][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[150][0] - ((mask[148][0] * mask[148][0]) + (mask[149][0] * mask[149][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[151][0] - ((mask[149][0] * mask[149][0]) + (mask[150][0] * mask[150][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[152][0] - ((mask[150][0] * mask[150][0]) + (mask[151][0] * mask[151][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[153][0] - ((mask[151][0] * mask[151][0]) + (mask[152][0] * mask[152][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[154][0] - ((mask[152][0] * mask[152][0]) + (mask[153][0] * mask[153][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[155][0] - ((mask[153][0] * mask[153][0]) + (mask[154][0] * mask[154][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[156][0] - ((mask[154][0] * mask[154][0]) + (mask[155][0] * mask[155][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[157][0] - ((mask[155][0] * mask[155][0]) + (mask[156][0] * mask[156][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[158][0] - ((mask[156][0] * mask[156][0]) + (mask[157][0] * mask[157][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[159][0] - ((mask[157][0] * mask[157][0]) + (mask[158][0] * mask[158][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[160][0] - ((mask[158][0] * mask[158][0]) + (mask[159][0] * mask[159][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[161][0] - ((mask[159][0] * mask[159][0]) + (mask[160][0] * mask[160][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[162][0] - ((mask[160][0] * mask[160][0]) + (mask[161][0] * mask[161][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[163][0] - ((mask[161][0] * mask[161][0]) + (mask[162][0] * mask[162][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[164][0] - ((mask[162][0] * mask[162][0]) + (mask[163][0] * mask[163][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[165][0] - ((mask[163][0] * mask[163][0]) + (mask[164][0] * mask[164][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[166][0] - ((mask[164][0] * mask[164][0]) + (mask[165][0] * mask[165][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[167][0] - ((mask[165][0] * mask[165][0]) + (mask[166][0] * mask[166][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[168][0] - ((mask[166][0] * mask[166][0]) + (mask[167][0] * mask[167][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[169][0] - ((mask[167][0] * mask[167][0]) + (mask[168][0] * mask[168][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[170][0] - ((mask[168][0] * mask[168][0]) + (mask[169][0] * mask[169][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[171][0] - ((mask[169][0] * mask[169][0]) + (mask[170][0] * mask[170][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[172][0] - ((mask[170][0] * mask[170][0]) + (mask[171][0] * mask[171][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[173][0] - ((mask[171][0] * mask[171][0]) + (mask[172][0] * mask[172][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[174][0] - ((mask[172][0] * mask[172][0]) + (mask[173][0] * mask[173][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[175][0] - ((mask[173][0] * mask[173][0]) + (mask[174][0] * mask[174][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[176][0] - ((mask[174][0] * mask[174][0]) + (mask[175][0] * mask[175][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[177][0] - ((mask[175][0] * mask[175][0]) + (mask[176][0] * mask[176][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[178][0] - ((mask[176][0] * mask[176][0]) + (mask[177][0] * mask[177][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[179][0] - ((mask[177][0] * mask[177][0]) + (mask[178][0] * mask[178][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[180][0] - ((mask[178][0] * mask[178][0]) + (mask[179][0] * mask[179][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[181][0] - ((mask[179][0] * mask[179][0]) + (mask[180][0] * mask[180][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[182][0] - ((mask[180][0] * mask[180][0]) + (mask[181][0] * mask[181][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[183][0] - ((mask[181][0] * mask[181][0]) + (mask[182][0] * mask[182][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[184][0] - ((mask[182][0] * mask[182][0]) + (mask[183][0] * mask[183][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[185][0] - ((mask[183][0] * mask[183][0]) + (mask[184][0] * mask[184][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[186][0] - ((mask[184][0] * mask[184][0]) + (mask[185][0] * mask[185][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[187][0] - ((mask[185][0] * mask[185][0]) + (mask[186][0] * mask[186][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[188][0] - ((mask[186][0] * mask[186][0]) + (mask[187][0] * mask[187][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[189][0] - ((mask[187][0] * mask[187][0]) + (mask[188][0] * mask[188][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[190][0] - ((mask[188][0] * mask[188][0]) + (mask[189][0] * mask[189][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[191][0] - ((mask[189][0] * mask[189][0]) + (mask[190][0] * mask[190][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[192][0] - ((mask[190][0] * mask[190][0]) + (mask[191][0] * mask[191][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[193][0] - ((mask[191][0] * mask[191][0]) + (mask[192][0] * mask[192][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[194][0] - ((mask[192][0] * mask[192][0]) + (mask[193][0] * mask[193][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[195][0] - ((mask[193][0] * mask[193][0]) + (mask[194][0] * mask[194][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[196][0] - ((mask[194][0] * mask[194][0]) + (mask[195][0] * mask[195][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[197][0] - ((mask[195][0] * mask[195][0]) + (mask[196][0] * mask[196][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[198][0] - ((mask[196][0] * mask[196][0]) + (mask[197][0] * mask[197][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[199][0] - ((mask[197][0] * mask[197][0]) + (mask[198][0] * mask[198][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[200][0] - ((mask[198][0] * mask[198][0]) + (mask[199][0] * mask[199][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[201][0] - ((mask[199][0] * mask[199][0]) + (mask[200][0] * mask[200][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[202][0] - ((mask[200][0] * mask[200][0]) + (mask[201][0] * mask[201][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[203][0] - ((mask[201][0] * mask[201][0]) + (mask[202][0] * mask[202][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[204][0] - ((mask[202][0] * mask[202][0]) + (mask[203][0] * mask[203][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[205][0] - ((mask[203][0] * mask[203][0]) + (mask[204][0] * mask[204][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[206][0] - ((mask[204][0] * mask[204][0]) + (mask[205][0] * mask[205][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[207][0] - ((mask[205][0] * mask[205][0]) + (mask[206][0] * mask[206][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[208][0] - ((mask[206][0] * mask[206][0]) + (mask[207][0] * mask[207][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[209][0] - ((mask[207][0] * mask[207][0]) + (mask[208][0] * mask[208][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[210][0] - ((mask[208][0] * mask[208][0]) + (mask[209][0] * mask[209][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[211][0] - ((mask[209][0] * mask[209][0]) + (mask[210][0] * mask[210][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[212][0] - ((mask[210][0] * mask[210][0]) + (mask[211][0] * mask[211][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[213][0] - ((mask[211][0] * mask[211][0]) + (mask[212][0] * mask[212][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[214][0] - ((mask[212][0] * mask[212][0]) + (mask[213][0] * mask[213][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[215][0] - ((mask[213][0] * mask[213][0]) + (mask[214][0] * mask[214][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[216][0] - ((mask[214][0] * mask[214][0]) + (mask[215][0] * mask[215][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[217][0] - ((mask[215][0] * mask[215][0]) + (mask[216][0] * mask[216][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[218][0] - ((mask[216][0] * mask[216][0]) + (mask[217][0] * mask[217][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[219][0] - ((mask[217][0] * mask[217][0]) + (mask[218][0] * mask[218][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[220][0] - ((mask[218][0] * mask[218][0]) + (mask[219][0] * mask[219][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[221][0] - ((mask[219][0] * mask[219][0]) + (mask[220][0] * mask[220][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[222][0] - ((mask[220][0] * mask[220][0]) + (mask[221][0] * mask[221][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[223][0] - ((mask[221][0] * mask[221][0]) + (mask[222][0] * mask[222][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[224][0] - ((mask[222][0] * mask[222][0]) + (mask[223][0] * mask[223][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[225][0] - ((mask[223][0] * mask[223][0]) + (mask[224][0] * mask[224][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[226][0] - ((mask[224][0] * mask[224][0]) + (mask[225][0] * mask[225][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[227][0] - ((mask[225][0] * mask[225][0]) + (mask[226][0] * mask[226][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[228][0] - ((mask[226][0] * mask[226][0]) + (mask[227][0] * mask[227][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[229][0] - ((mask[227][0] * mask[227][0]) + (mask[228][0] * mask[228][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[230][0] - ((mask[228][0] * mask[228][0]) + (mask[229][0] * mask[229][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[231][0] - ((mask[229][0] * mask[229][0]) + (mask[230][0] * mask[230][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[232][0] - ((mask[230][0] * mask[230][0]) + (mask[231][0] * mask[231][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[233][0] - ((mask[231][0] * mask[231][0]) + (mask[232][0] * mask[232][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[234][0] - ((mask[232][0] * mask[232][0]) + (mask[233][0] * mask[233][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[235][0] - ((mask[233][0] * mask[233][0]) + (mask[234][0] * mask[234][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[236][0] - ((mask[234][0] * mask[234][0]) + (mask[235][0] * mask[235][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[237][0] - ((mask[235][0] * mask[235][0]) + (mask[236][0] * mask[236][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[238][0] - ((mask[236][0] * mask[236][0]) + (mask[237][0] * mask[237][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[239][0] - ((mask[237][0] * mask[237][0]) + (mask[238][0] * mask[238][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[240][0] - ((mask[238][0] * mask[238][0]) + (mask[239][0] * mask[239][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[241][0] - ((mask[239][0] * mask[239][0]) + (mask[240][0] * mask[240][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[242][0] - ((mask[240][0] * mask[240][0]) + (mask[241][0] * mask[241][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[243][0] - ((mask[241][0] * mask[241][0]) + (mask[242][0] * mask[242][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[244][0] - ((mask[242][0] * mask[242][0]) + (mask[243][0] * mask[243][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[245][0] - ((mask[243][0] * mask[243][0]) + (mask[244][0] * mask[244][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[246][0] - ((mask[244][0] * mask[244][0]) + (mask[245][0] * mask[245][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[247][0] - ((mask[245][0] * mask[245][0]) + (mask[246][0] * mask[246][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[248][0] - ((mask[246][0] * mask[246][0]) + (mask[247][0] * mask[247][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[249][0] - ((mask[247][0] * mask[247][0]) + (mask[248][0] * mask[248][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[250][0] - ((mask[248][0] * mask[248][0]) + (mask[249][0] * mask[249][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[251][0] - ((mask[249][0] * mask[249][0]) + (mask[250][0] * mask[250][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[252][0] - ((mask[250][0] * mask[250][0]) + (mask[251][0] * mask[251][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[253][0] - ((mask[251][0] * mask[251][0]) + (mask[252][0] * mask[252][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[254][0] - ((mask[252][0] * mask[252][0]) + (mask[253][0] * mask[253][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
        let numerator =
            (mask[255][0] - ((mask[253][0] * mask[253][0]) + (mask[254][0] * mask[254][0])));
        evaluation_accumulator.accumulate(numerator / denominator);
    }
}
