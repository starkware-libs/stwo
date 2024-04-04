use num_traits::Zero;

use super::structs::WideFibComponent;
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentTrace, Mask};
use crate::core::backend::CPUBackend;
use crate::core::circle::CirclePoint;
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::CanonicCoset;
use crate::core::utils::bit_reverse_index;
use crate::core::ColumnVec;

impl Component<CPUBackend> for WideFibComponent {
    fn n_constraints(&self) -> usize {
        62
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    fn trace_log_degree_bounds(&self) -> Vec<u32> {
        vec![self.log_size; 256]
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

    // TODO(ShaharS), precompute random coeff powers.
    // TODO(ShaharS), use intermidiate value to save the computation of the squares.
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, CPUBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CPUBackend>,
    ) {
        let mut trace_evals = vec![];
        // TODO(ShaharS), Share this LDE with the commitment LDE.
        for poly_index in 0..64 {
            let poly = &trace.columns[poly_index];
            let trace_eval_domain =
                CanonicCoset::new(self.max_constraint_log_degree_bound()).circle_domain();
            trace_evals.push(poly.evaluate(trace_eval_domain).bit_reverse());
        }
        let zero_domain = CanonicCoset::new(self.log_size).coset;
        let eval_domain = CanonicCoset::new(self.log_size + 1).circle_domain();
        let mut denoms = vec![];
        for point in eval_domain.iter() {
            denoms.push(coset_vanishing(zero_domain, point));
        }
        let mut denom_inverses =
            vec![BaseField::zero(); 1 << (self.max_constraint_log_degree_bound())];
        BaseField::batch_inverse(&denoms, &mut denom_inverses);
        let mut numerators =
            vec![SecureField::zero(); 1 << (self.max_constraint_log_degree_bound())];
        let [mut accum] = evaluation_accumulator
            .columns([(self.max_constraint_log_degree_bound(), self.n_constraints())]);
        // TODO (ShaharS) Change to get the correct power of random coeff inside the loop.
        let random_coeff = accum.random_coeff_powers[1];
        for (i, point_index) in eval_domain.iter_indices().enumerate() {
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[2].get_at(point_index)
                    - ((trace_evals[0].get_at(point_index) * trace_evals[0].get_at(point_index))
                        + (trace_evals[1].get_at(point_index)
                            * trace_evals[1].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[3].get_at(point_index)
                    - ((trace_evals[1].get_at(point_index) * trace_evals[1].get_at(point_index))
                        + (trace_evals[2].get_at(point_index)
                            * trace_evals[2].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[4].get_at(point_index)
                    - ((trace_evals[2].get_at(point_index) * trace_evals[2].get_at(point_index))
                        + (trace_evals[3].get_at(point_index)
                            * trace_evals[3].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[5].get_at(point_index)
                    - ((trace_evals[3].get_at(point_index) * trace_evals[3].get_at(point_index))
                        + (trace_evals[4].get_at(point_index)
                            * trace_evals[4].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[6].get_at(point_index)
                    - ((trace_evals[4].get_at(point_index) * trace_evals[4].get_at(point_index))
                        + (trace_evals[5].get_at(point_index)
                            * trace_evals[5].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[7].get_at(point_index)
                    - ((trace_evals[5].get_at(point_index) * trace_evals[5].get_at(point_index))
                        + (trace_evals[6].get_at(point_index)
                            * trace_evals[6].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[8].get_at(point_index)
                    - ((trace_evals[6].get_at(point_index) * trace_evals[6].get_at(point_index))
                        + (trace_evals[7].get_at(point_index)
                            * trace_evals[7].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[9].get_at(point_index)
                    - ((trace_evals[7].get_at(point_index) * trace_evals[7].get_at(point_index))
                        + (trace_evals[8].get_at(point_index)
                            * trace_evals[8].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[10].get_at(point_index)
                    - ((trace_evals[8].get_at(point_index) * trace_evals[8].get_at(point_index))
                        + (trace_evals[9].get_at(point_index)
                            * trace_evals[9].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[11].get_at(point_index)
                    - ((trace_evals[9].get_at(point_index) * trace_evals[9].get_at(point_index))
                        + (trace_evals[10].get_at(point_index)
                            * trace_evals[10].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[12].get_at(point_index)
                    - ((trace_evals[10].get_at(point_index)
                        * trace_evals[10].get_at(point_index))
                        + (trace_evals[11].get_at(point_index)
                            * trace_evals[11].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[13].get_at(point_index)
                    - ((trace_evals[11].get_at(point_index)
                        * trace_evals[11].get_at(point_index))
                        + (trace_evals[12].get_at(point_index)
                            * trace_evals[12].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[14].get_at(point_index)
                    - ((trace_evals[12].get_at(point_index)
                        * trace_evals[12].get_at(point_index))
                        + (trace_evals[13].get_at(point_index)
                            * trace_evals[13].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[15].get_at(point_index)
                    - ((trace_evals[13].get_at(point_index)
                        * trace_evals[13].get_at(point_index))
                        + (trace_evals[14].get_at(point_index)
                            * trace_evals[14].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[16].get_at(point_index)
                    - ((trace_evals[14].get_at(point_index)
                        * trace_evals[14].get_at(point_index))
                        + (trace_evals[15].get_at(point_index)
                            * trace_evals[15].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[17].get_at(point_index)
                    - ((trace_evals[15].get_at(point_index)
                        * trace_evals[15].get_at(point_index))
                        + (trace_evals[16].get_at(point_index)
                            * trace_evals[16].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[18].get_at(point_index)
                    - ((trace_evals[16].get_at(point_index)
                        * trace_evals[16].get_at(point_index))
                        + (trace_evals[17].get_at(point_index)
                            * trace_evals[17].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[19].get_at(point_index)
                    - ((trace_evals[17].get_at(point_index)
                        * trace_evals[17].get_at(point_index))
                        + (trace_evals[18].get_at(point_index)
                            * trace_evals[18].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[20].get_at(point_index)
                    - ((trace_evals[18].get_at(point_index)
                        * trace_evals[18].get_at(point_index))
                        + (trace_evals[19].get_at(point_index)
                            * trace_evals[19].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[21].get_at(point_index)
                    - ((trace_evals[19].get_at(point_index)
                        * trace_evals[19].get_at(point_index))
                        + (trace_evals[20].get_at(point_index)
                            * trace_evals[20].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[22].get_at(point_index)
                    - ((trace_evals[20].get_at(point_index)
                        * trace_evals[20].get_at(point_index))
                        + (trace_evals[21].get_at(point_index)
                            * trace_evals[21].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[23].get_at(point_index)
                    - ((trace_evals[21].get_at(point_index)
                        * trace_evals[21].get_at(point_index))
                        + (trace_evals[22].get_at(point_index)
                            * trace_evals[22].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[24].get_at(point_index)
                    - ((trace_evals[22].get_at(point_index)
                        * trace_evals[22].get_at(point_index))
                        + (trace_evals[23].get_at(point_index)
                            * trace_evals[23].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[25].get_at(point_index)
                    - ((trace_evals[23].get_at(point_index)
                        * trace_evals[23].get_at(point_index))
                        + (trace_evals[24].get_at(point_index)
                            * trace_evals[24].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[26].get_at(point_index)
                    - ((trace_evals[24].get_at(point_index)
                        * trace_evals[24].get_at(point_index))
                        + (trace_evals[25].get_at(point_index)
                            * trace_evals[25].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[27].get_at(point_index)
                    - ((trace_evals[25].get_at(point_index)
                        * trace_evals[25].get_at(point_index))
                        + (trace_evals[26].get_at(point_index)
                            * trace_evals[26].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[28].get_at(point_index)
                    - ((trace_evals[26].get_at(point_index)
                        * trace_evals[26].get_at(point_index))
                        + (trace_evals[27].get_at(point_index)
                            * trace_evals[27].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[29].get_at(point_index)
                    - ((trace_evals[27].get_at(point_index)
                        * trace_evals[27].get_at(point_index))
                        + (trace_evals[28].get_at(point_index)
                            * trace_evals[28].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[30].get_at(point_index)
                    - ((trace_evals[28].get_at(point_index)
                        * trace_evals[28].get_at(point_index))
                        + (trace_evals[29].get_at(point_index)
                            * trace_evals[29].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[31].get_at(point_index)
                    - ((trace_evals[29].get_at(point_index)
                        * trace_evals[29].get_at(point_index))
                        + (trace_evals[30].get_at(point_index)
                            * trace_evals[30].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[32].get_at(point_index)
                    - ((trace_evals[30].get_at(point_index)
                        * trace_evals[30].get_at(point_index))
                        + (trace_evals[31].get_at(point_index)
                            * trace_evals[31].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[33].get_at(point_index)
                    - ((trace_evals[31].get_at(point_index)
                        * trace_evals[31].get_at(point_index))
                        + (trace_evals[32].get_at(point_index)
                            * trace_evals[32].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[34].get_at(point_index)
                    - ((trace_evals[32].get_at(point_index)
                        * trace_evals[32].get_at(point_index))
                        + (trace_evals[33].get_at(point_index)
                            * trace_evals[33].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[35].get_at(point_index)
                    - ((trace_evals[33].get_at(point_index)
                        * trace_evals[33].get_at(point_index))
                        + (trace_evals[34].get_at(point_index)
                            * trace_evals[34].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[36].get_at(point_index)
                    - ((trace_evals[34].get_at(point_index)
                        * trace_evals[34].get_at(point_index))
                        + (trace_evals[35].get_at(point_index)
                            * trace_evals[35].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[37].get_at(point_index)
                    - ((trace_evals[35].get_at(point_index)
                        * trace_evals[35].get_at(point_index))
                        + (trace_evals[36].get_at(point_index)
                            * trace_evals[36].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[38].get_at(point_index)
                    - ((trace_evals[36].get_at(point_index)
                        * trace_evals[36].get_at(point_index))
                        + (trace_evals[37].get_at(point_index)
                            * trace_evals[37].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[39].get_at(point_index)
                    - ((trace_evals[37].get_at(point_index)
                        * trace_evals[37].get_at(point_index))
                        + (trace_evals[38].get_at(point_index)
                            * trace_evals[38].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[40].get_at(point_index)
                    - ((trace_evals[38].get_at(point_index)
                        * trace_evals[38].get_at(point_index))
                        + (trace_evals[39].get_at(point_index)
                            * trace_evals[39].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[41].get_at(point_index)
                    - ((trace_evals[39].get_at(point_index)
                        * trace_evals[39].get_at(point_index))
                        + (trace_evals[40].get_at(point_index)
                            * trace_evals[40].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[42].get_at(point_index)
                    - ((trace_evals[40].get_at(point_index)
                        * trace_evals[40].get_at(point_index))
                        + (trace_evals[41].get_at(point_index)
                            * trace_evals[41].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[43].get_at(point_index)
                    - ((trace_evals[41].get_at(point_index)
                        * trace_evals[41].get_at(point_index))
                        + (trace_evals[42].get_at(point_index)
                            * trace_evals[42].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[44].get_at(point_index)
                    - ((trace_evals[42].get_at(point_index)
                        * trace_evals[42].get_at(point_index))
                        + (trace_evals[43].get_at(point_index)
                            * trace_evals[43].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[45].get_at(point_index)
                    - ((trace_evals[43].get_at(point_index)
                        * trace_evals[43].get_at(point_index))
                        + (trace_evals[44].get_at(point_index)
                            * trace_evals[44].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[46].get_at(point_index)
                    - ((trace_evals[44].get_at(point_index)
                        * trace_evals[44].get_at(point_index))
                        + (trace_evals[45].get_at(point_index)
                            * trace_evals[45].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[47].get_at(point_index)
                    - ((trace_evals[45].get_at(point_index)
                        * trace_evals[45].get_at(point_index))
                        + (trace_evals[46].get_at(point_index)
                            * trace_evals[46].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[48].get_at(point_index)
                    - ((trace_evals[46].get_at(point_index)
                        * trace_evals[46].get_at(point_index))
                        + (trace_evals[47].get_at(point_index)
                            * trace_evals[47].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[49].get_at(point_index)
                    - ((trace_evals[47].get_at(point_index)
                        * trace_evals[47].get_at(point_index))
                        + (trace_evals[48].get_at(point_index)
                            * trace_evals[48].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[50].get_at(point_index)
                    - ((trace_evals[48].get_at(point_index)
                        * trace_evals[48].get_at(point_index))
                        + (trace_evals[49].get_at(point_index)
                            * trace_evals[49].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[51].get_at(point_index)
                    - ((trace_evals[49].get_at(point_index)
                        * trace_evals[49].get_at(point_index))
                        + (trace_evals[50].get_at(point_index)
                            * trace_evals[50].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[52].get_at(point_index)
                    - ((trace_evals[50].get_at(point_index)
                        * trace_evals[50].get_at(point_index))
                        + (trace_evals[51].get_at(point_index)
                            * trace_evals[51].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[53].get_at(point_index)
                    - ((trace_evals[51].get_at(point_index)
                        * trace_evals[51].get_at(point_index))
                        + (trace_evals[52].get_at(point_index)
                            * trace_evals[52].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[54].get_at(point_index)
                    - ((trace_evals[52].get_at(point_index)
                        * trace_evals[52].get_at(point_index))
                        + (trace_evals[53].get_at(point_index)
                            * trace_evals[53].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[55].get_at(point_index)
                    - ((trace_evals[53].get_at(point_index)
                        * trace_evals[53].get_at(point_index))
                        + (trace_evals[54].get_at(point_index)
                            * trace_evals[54].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[56].get_at(point_index)
                    - ((trace_evals[54].get_at(point_index)
                        * trace_evals[54].get_at(point_index))
                        + (trace_evals[55].get_at(point_index)
                            * trace_evals[55].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[57].get_at(point_index)
                    - ((trace_evals[55].get_at(point_index)
                        * trace_evals[55].get_at(point_index))
                        + (trace_evals[56].get_at(point_index)
                            * trace_evals[56].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[58].get_at(point_index)
                    - ((trace_evals[56].get_at(point_index)
                        * trace_evals[56].get_at(point_index))
                        + (trace_evals[57].get_at(point_index)
                            * trace_evals[57].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[59].get_at(point_index)
                    - ((trace_evals[57].get_at(point_index)
                        * trace_evals[57].get_at(point_index))
                        + (trace_evals[58].get_at(point_index)
                            * trace_evals[58].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[60].get_at(point_index)
                    - ((trace_evals[58].get_at(point_index)
                        * trace_evals[58].get_at(point_index))
                        + (trace_evals[59].get_at(point_index)
                            * trace_evals[59].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[61].get_at(point_index)
                    - ((trace_evals[59].get_at(point_index)
                        * trace_evals[59].get_at(point_index))
                        + (trace_evals[60].get_at(point_index)
                            * trace_evals[60].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[62].get_at(point_index)
                    - ((trace_evals[60].get_at(point_index)
                        * trace_evals[60].get_at(point_index))
                        + (trace_evals[61].get_at(point_index)
                            * trace_evals[61].get_at(point_index))));
            numerators[i] = numerators[i] * random_coeff
                + (trace_evals[63].get_at(point_index)
                    - ((trace_evals[61].get_at(point_index)
                        * trace_evals[61].get_at(point_index))
                        + (trace_evals[62].get_at(point_index)
                            * trace_evals[62].get_at(point_index))));
        }
        for (i, (num, denom)) in numerators.iter().zip(denom_inverses.iter()).enumerate() {
            accum.accumulate(
                bit_reverse_index(i, self.max_constraint_log_degree_bound()),
                *num * *denom,
            );
        }
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        _point: CirclePoint<SecureField>,
        _mask: &ColumnVec<Vec<SecureField>>,
        _evaluation_accumulator: &mut PointEvaluationAccumulator,
    ) {
        unimplemented!("not implemented")
    }
}
