use num_traits::{One, Zero};

use super::structs::{WideFibAir, WideFibComponent, LOG_N_COLUMNS};
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Air, Component, ComponentTrace, Mask};
use crate::core::backend::CPUBackend;
use crate::core::circle::{CirclePoint, Coset};
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::CanonicCoset;
use crate::core::utils::bit_reverse_index;
use crate::core::ColumnVec;
use crate::examples::wide_fibonacci::structs::N_COLUMNS;

impl Air<CPUBackend> for WideFibAir {
    fn components(&self) -> Vec<&dyn Component<CPUBackend>> {
        vec![&self.component]
    }
}

impl Component<CPUBackend> for WideFibComponent {
    fn n_constraints(&self) -> usize {
        N_COLUMNS - 1
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_column_size() + 1
    }

    fn trace_log_degree_bounds(&self) -> Vec<u32> {
        vec![self.log_column_size(); N_COLUMNS]
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> ColumnVec<Vec<CirclePoint<SecureField>>> {
        let mask = Mask(vec![vec![0_usize]; N_COLUMNS]);
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
        let constraint_log_degree = Component::<CPUBackend>::max_constraint_log_degree_bound(self);
        let n_constraints = Component::<CPUBackend>::n_constraints(self);
        let mut trace_evals = vec![];
        // TODO(ShaharS), Share this LDE with the commitment LDE.
        for poly_index in 0..64 {
            let poly = &trace.columns[poly_index];
            let trace_eval_domain = CanonicCoset::new(constraint_log_degree).circle_domain();
            trace_evals.push(poly.evaluate(trace_eval_domain).bit_reverse());
        }
        let step_zero_domain = CanonicCoset::new(self.log_column_size()).coset;
        let boundary_zero_domain = Coset::new(
            step_zero_domain.initial_index,
            step_zero_domain.log_size >> (self.log_fibonacci_size - LOG_N_COLUMNS as u32),
        );
        let eval_domain = CanonicCoset::new(constraint_log_degree).circle_domain();
        let mut step_denoms = vec![];
        let mut boundary_denoms = vec![];
        for point in eval_domain.iter() {
            step_denoms.push(coset_vanishing(step_zero_domain, point));
            boundary_denoms.push(coset_vanishing(boundary_zero_domain, point));
        }
        let mut step_denom_inverses = vec![BaseField::zero(); 1 << (constraint_log_degree)];
        let mut boundary_denom_inverses = vec![BaseField::zero(); 1 << (constraint_log_degree)];
        BaseField::batch_inverse(&step_denoms, &mut step_denom_inverses);
        BaseField::batch_inverse(&boundary_denoms, &mut boundary_denom_inverses);
        let [mut accum] = evaluation_accumulator.columns([(constraint_log_degree, n_constraints)]);
        let mut boundary_numerators = vec![SecureField::zero(); 1 << constraint_log_degree];
        let mut step_numerators = vec![SecureField::zero(); 1 << constraint_log_degree];

        // TODO(AlonH): Add the public input boundary constraint.
        for (i, point_index) in eval_domain.iter_indices().enumerate() {
            boundary_numerators[i] = trace_evals[0].get_at(point_index) - SecureField::one();
        }
        for (i, (num, denom_inverse)) in boundary_numerators
            .iter()
            .zip(boundary_denom_inverses.iter())
            .enumerate()
        {
            accum.accumulate(
                bit_reverse_index(i, constraint_log_degree),
                (*num * *denom_inverse) * accum.random_coeff_powers[N_COLUMNS - 2],
            );
        }
        // TODO (ShaharS) Change to get the correct power of random coeff inside the loop.
        let random_coeff = accum.random_coeff_powers[1];
        for (i, point_index) in eval_domain.iter_indices().enumerate() {
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[2].get_at(point_index)
                    - ((trace_evals[0].get_at(point_index) * trace_evals[0].get_at(point_index))
                        + (trace_evals[1].get_at(point_index)
                            * trace_evals[1].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[3].get_at(point_index)
                    - ((trace_evals[1].get_at(point_index) * trace_evals[1].get_at(point_index))
                        + (trace_evals[2].get_at(point_index)
                            * trace_evals[2].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[4].get_at(point_index)
                    - ((trace_evals[2].get_at(point_index) * trace_evals[2].get_at(point_index))
                        + (trace_evals[3].get_at(point_index)
                            * trace_evals[3].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[5].get_at(point_index)
                    - ((trace_evals[3].get_at(point_index) * trace_evals[3].get_at(point_index))
                        + (trace_evals[4].get_at(point_index)
                            * trace_evals[4].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[6].get_at(point_index)
                    - ((trace_evals[4].get_at(point_index) * trace_evals[4].get_at(point_index))
                        + (trace_evals[5].get_at(point_index)
                            * trace_evals[5].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[7].get_at(point_index)
                    - ((trace_evals[5].get_at(point_index) * trace_evals[5].get_at(point_index))
                        + (trace_evals[6].get_at(point_index)
                            * trace_evals[6].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[8].get_at(point_index)
                    - ((trace_evals[6].get_at(point_index) * trace_evals[6].get_at(point_index))
                        + (trace_evals[7].get_at(point_index)
                            * trace_evals[7].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[9].get_at(point_index)
                    - ((trace_evals[7].get_at(point_index) * trace_evals[7].get_at(point_index))
                        + (trace_evals[8].get_at(point_index)
                            * trace_evals[8].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[10].get_at(point_index)
                    - ((trace_evals[8].get_at(point_index) * trace_evals[8].get_at(point_index))
                        + (trace_evals[9].get_at(point_index)
                            * trace_evals[9].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[11].get_at(point_index)
                    - ((trace_evals[9].get_at(point_index) * trace_evals[9].get_at(point_index))
                        + (trace_evals[10].get_at(point_index)
                            * trace_evals[10].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[12].get_at(point_index)
                    - ((trace_evals[10].get_at(point_index)
                        * trace_evals[10].get_at(point_index))
                        + (trace_evals[11].get_at(point_index)
                            * trace_evals[11].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[13].get_at(point_index)
                    - ((trace_evals[11].get_at(point_index)
                        * trace_evals[11].get_at(point_index))
                        + (trace_evals[12].get_at(point_index)
                            * trace_evals[12].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[14].get_at(point_index)
                    - ((trace_evals[12].get_at(point_index)
                        * trace_evals[12].get_at(point_index))
                        + (trace_evals[13].get_at(point_index)
                            * trace_evals[13].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[15].get_at(point_index)
                    - ((trace_evals[13].get_at(point_index)
                        * trace_evals[13].get_at(point_index))
                        + (trace_evals[14].get_at(point_index)
                            * trace_evals[14].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[16].get_at(point_index)
                    - ((trace_evals[14].get_at(point_index)
                        * trace_evals[14].get_at(point_index))
                        + (trace_evals[15].get_at(point_index)
                            * trace_evals[15].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[17].get_at(point_index)
                    - ((trace_evals[15].get_at(point_index)
                        * trace_evals[15].get_at(point_index))
                        + (trace_evals[16].get_at(point_index)
                            * trace_evals[16].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[18].get_at(point_index)
                    - ((trace_evals[16].get_at(point_index)
                        * trace_evals[16].get_at(point_index))
                        + (trace_evals[17].get_at(point_index)
                            * trace_evals[17].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[19].get_at(point_index)
                    - ((trace_evals[17].get_at(point_index)
                        * trace_evals[17].get_at(point_index))
                        + (trace_evals[18].get_at(point_index)
                            * trace_evals[18].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[20].get_at(point_index)
                    - ((trace_evals[18].get_at(point_index)
                        * trace_evals[18].get_at(point_index))
                        + (trace_evals[19].get_at(point_index)
                            * trace_evals[19].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[21].get_at(point_index)
                    - ((trace_evals[19].get_at(point_index)
                        * trace_evals[19].get_at(point_index))
                        + (trace_evals[20].get_at(point_index)
                            * trace_evals[20].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[22].get_at(point_index)
                    - ((trace_evals[20].get_at(point_index)
                        * trace_evals[20].get_at(point_index))
                        + (trace_evals[21].get_at(point_index)
                            * trace_evals[21].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[23].get_at(point_index)
                    - ((trace_evals[21].get_at(point_index)
                        * trace_evals[21].get_at(point_index))
                        + (trace_evals[22].get_at(point_index)
                            * trace_evals[22].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[24].get_at(point_index)
                    - ((trace_evals[22].get_at(point_index)
                        * trace_evals[22].get_at(point_index))
                        + (trace_evals[23].get_at(point_index)
                            * trace_evals[23].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[25].get_at(point_index)
                    - ((trace_evals[23].get_at(point_index)
                        * trace_evals[23].get_at(point_index))
                        + (trace_evals[24].get_at(point_index)
                            * trace_evals[24].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[26].get_at(point_index)
                    - ((trace_evals[24].get_at(point_index)
                        * trace_evals[24].get_at(point_index))
                        + (trace_evals[25].get_at(point_index)
                            * trace_evals[25].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[27].get_at(point_index)
                    - ((trace_evals[25].get_at(point_index)
                        * trace_evals[25].get_at(point_index))
                        + (trace_evals[26].get_at(point_index)
                            * trace_evals[26].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[28].get_at(point_index)
                    - ((trace_evals[26].get_at(point_index)
                        * trace_evals[26].get_at(point_index))
                        + (trace_evals[27].get_at(point_index)
                            * trace_evals[27].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[29].get_at(point_index)
                    - ((trace_evals[27].get_at(point_index)
                        * trace_evals[27].get_at(point_index))
                        + (trace_evals[28].get_at(point_index)
                            * trace_evals[28].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[30].get_at(point_index)
                    - ((trace_evals[28].get_at(point_index)
                        * trace_evals[28].get_at(point_index))
                        + (trace_evals[29].get_at(point_index)
                            * trace_evals[29].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[31].get_at(point_index)
                    - ((trace_evals[29].get_at(point_index)
                        * trace_evals[29].get_at(point_index))
                        + (trace_evals[30].get_at(point_index)
                            * trace_evals[30].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[32].get_at(point_index)
                    - ((trace_evals[30].get_at(point_index)
                        * trace_evals[30].get_at(point_index))
                        + (trace_evals[31].get_at(point_index)
                            * trace_evals[31].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[33].get_at(point_index)
                    - ((trace_evals[31].get_at(point_index)
                        * trace_evals[31].get_at(point_index))
                        + (trace_evals[32].get_at(point_index)
                            * trace_evals[32].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[34].get_at(point_index)
                    - ((trace_evals[32].get_at(point_index)
                        * trace_evals[32].get_at(point_index))
                        + (trace_evals[33].get_at(point_index)
                            * trace_evals[33].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[35].get_at(point_index)
                    - ((trace_evals[33].get_at(point_index)
                        * trace_evals[33].get_at(point_index))
                        + (trace_evals[34].get_at(point_index)
                            * trace_evals[34].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[36].get_at(point_index)
                    - ((trace_evals[34].get_at(point_index)
                        * trace_evals[34].get_at(point_index))
                        + (trace_evals[35].get_at(point_index)
                            * trace_evals[35].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[37].get_at(point_index)
                    - ((trace_evals[35].get_at(point_index)
                        * trace_evals[35].get_at(point_index))
                        + (trace_evals[36].get_at(point_index)
                            * trace_evals[36].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[38].get_at(point_index)
                    - ((trace_evals[36].get_at(point_index)
                        * trace_evals[36].get_at(point_index))
                        + (trace_evals[37].get_at(point_index)
                            * trace_evals[37].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[39].get_at(point_index)
                    - ((trace_evals[37].get_at(point_index)
                        * trace_evals[37].get_at(point_index))
                        + (trace_evals[38].get_at(point_index)
                            * trace_evals[38].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[40].get_at(point_index)
                    - ((trace_evals[38].get_at(point_index)
                        * trace_evals[38].get_at(point_index))
                        + (trace_evals[39].get_at(point_index)
                            * trace_evals[39].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[41].get_at(point_index)
                    - ((trace_evals[39].get_at(point_index)
                        * trace_evals[39].get_at(point_index))
                        + (trace_evals[40].get_at(point_index)
                            * trace_evals[40].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[42].get_at(point_index)
                    - ((trace_evals[40].get_at(point_index)
                        * trace_evals[40].get_at(point_index))
                        + (trace_evals[41].get_at(point_index)
                            * trace_evals[41].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[43].get_at(point_index)
                    - ((trace_evals[41].get_at(point_index)
                        * trace_evals[41].get_at(point_index))
                        + (trace_evals[42].get_at(point_index)
                            * trace_evals[42].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[44].get_at(point_index)
                    - ((trace_evals[42].get_at(point_index)
                        * trace_evals[42].get_at(point_index))
                        + (trace_evals[43].get_at(point_index)
                            * trace_evals[43].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[45].get_at(point_index)
                    - ((trace_evals[43].get_at(point_index)
                        * trace_evals[43].get_at(point_index))
                        + (trace_evals[44].get_at(point_index)
                            * trace_evals[44].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[46].get_at(point_index)
                    - ((trace_evals[44].get_at(point_index)
                        * trace_evals[44].get_at(point_index))
                        + (trace_evals[45].get_at(point_index)
                            * trace_evals[45].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[47].get_at(point_index)
                    - ((trace_evals[45].get_at(point_index)
                        * trace_evals[45].get_at(point_index))
                        + (trace_evals[46].get_at(point_index)
                            * trace_evals[46].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[48].get_at(point_index)
                    - ((trace_evals[46].get_at(point_index)
                        * trace_evals[46].get_at(point_index))
                        + (trace_evals[47].get_at(point_index)
                            * trace_evals[47].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[49].get_at(point_index)
                    - ((trace_evals[47].get_at(point_index)
                        * trace_evals[47].get_at(point_index))
                        + (trace_evals[48].get_at(point_index)
                            * trace_evals[48].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[50].get_at(point_index)
                    - ((trace_evals[48].get_at(point_index)
                        * trace_evals[48].get_at(point_index))
                        + (trace_evals[49].get_at(point_index)
                            * trace_evals[49].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[51].get_at(point_index)
                    - ((trace_evals[49].get_at(point_index)
                        * trace_evals[49].get_at(point_index))
                        + (trace_evals[50].get_at(point_index)
                            * trace_evals[50].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[52].get_at(point_index)
                    - ((trace_evals[50].get_at(point_index)
                        * trace_evals[50].get_at(point_index))
                        + (trace_evals[51].get_at(point_index)
                            * trace_evals[51].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[53].get_at(point_index)
                    - ((trace_evals[51].get_at(point_index)
                        * trace_evals[51].get_at(point_index))
                        + (trace_evals[52].get_at(point_index)
                            * trace_evals[52].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[54].get_at(point_index)
                    - ((trace_evals[52].get_at(point_index)
                        * trace_evals[52].get_at(point_index))
                        + (trace_evals[53].get_at(point_index)
                            * trace_evals[53].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[55].get_at(point_index)
                    - ((trace_evals[53].get_at(point_index)
                        * trace_evals[53].get_at(point_index))
                        + (trace_evals[54].get_at(point_index)
                            * trace_evals[54].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[56].get_at(point_index)
                    - ((trace_evals[54].get_at(point_index)
                        * trace_evals[54].get_at(point_index))
                        + (trace_evals[55].get_at(point_index)
                            * trace_evals[55].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[57].get_at(point_index)
                    - ((trace_evals[55].get_at(point_index)
                        * trace_evals[55].get_at(point_index))
                        + (trace_evals[56].get_at(point_index)
                            * trace_evals[56].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[58].get_at(point_index)
                    - ((trace_evals[56].get_at(point_index)
                        * trace_evals[56].get_at(point_index))
                        + (trace_evals[57].get_at(point_index)
                            * trace_evals[57].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[59].get_at(point_index)
                    - ((trace_evals[57].get_at(point_index)
                        * trace_evals[57].get_at(point_index))
                        + (trace_evals[58].get_at(point_index)
                            * trace_evals[58].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[60].get_at(point_index)
                    - ((trace_evals[58].get_at(point_index)
                        * trace_evals[58].get_at(point_index))
                        + (trace_evals[59].get_at(point_index)
                            * trace_evals[59].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[61].get_at(point_index)
                    - ((trace_evals[59].get_at(point_index)
                        * trace_evals[59].get_at(point_index))
                        + (trace_evals[60].get_at(point_index)
                            * trace_evals[60].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[62].get_at(point_index)
                    - ((trace_evals[60].get_at(point_index)
                        * trace_evals[60].get_at(point_index))
                        + (trace_evals[61].get_at(point_index)
                            * trace_evals[61].get_at(point_index))));
            step_numerators[i] = step_numerators[i] * random_coeff
                + (trace_evals[63].get_at(point_index)
                    - ((trace_evals[61].get_at(point_index)
                        * trace_evals[61].get_at(point_index))
                        + (trace_evals[62].get_at(point_index)
                            * trace_evals[62].get_at(point_index))));
        }
        for (i, (num, denom_inverse)) in step_numerators
            .iter()
            .zip(step_denom_inverses.iter())
            .enumerate()
        {
            accum.accumulate(
                bit_reverse_index(i, constraint_log_degree),
                *num * *denom_inverse,
            );
        }
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &ColumnVec<Vec<SecureField>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    ) {
        let step_zero_domain = CanonicCoset::new(self.log_column_size()).coset;
        let boundary_zero_domain = Coset::new(
            step_zero_domain.initial_index,
            step_zero_domain.log_size >> (self.log_fibonacci_size - LOG_N_COLUMNS as u32),
        );
        let step_denominator = coset_vanishing(step_zero_domain, point);
        let boundary_denominator = coset_vanishing(boundary_zero_domain, point);
        evaluation_accumulator.accumulate((mask[0][0] - SecureField::one()) / boundary_denominator);
        for i in 0..(N_COLUMNS - 2) {
            let numerator = mask[i + 2][0] - (mask[i][0].square() + mask[i + 1][0].square());
            evaluation_accumulator.accumulate(numerator / step_denominator);
        }
    }
}
