use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Sub};

use num_traits::{One, Zero};

use crate::core::air::accumulation::PointEvaluationAccumulator;
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Column, CpuBackend};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{CircleEvaluation, SecureCirclePoly};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::offset_bit_reversed_circle_domain_index;
use crate::core::ColumnVec;

pub trait EvalAtRow {
    type F: FieldExpOps
        + Copy
        + Debug
        + AddAssign<Self::F>
        + Add<Self::F, Output = Self::F>
        + Sub<Self::F, Output = Self::F>
        + Mul<BaseField, Output = Self::F>
        + Add<SecureField, Output = Self::EF>
        + Mul<SecureField, Output = Self::EF>;
    type EF: One
        + Copy
        + Debug
        + Zero
        + Add<SecureField, Output = Self::EF>
        + Sub<SecureField, Output = Self::EF>
        + Mul<SecureField, Output = Self::EF>
        + Add<Self::F, Output = Self::EF>
        + Mul<Self::F, Output = Self::EF>
        + Sub<Self::EF, Output = Self::EF>
        + Mul<Self::EF, Output = Self::EF>;

    fn next_mask(&mut self) -> Self::F {
        self.next_interaction_mask(0, [0])[0]
    }
    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N];
    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF>;
    fn pow2(&self, i: u32) -> Self::F;
    fn combine_ef(values: [Self::F; 4]) -> Self::EF;
}

pub struct EvalAtDomain<'a> {
    pub trace_eval:
        &'a TreeVec<Vec<&'a CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
    pub col_index: Vec<usize>,
    pub vec_row: usize,
    pub random_coeff_powers: &'a [SecureField],
    pub row_res: PackedSecureField,
    pub constraint_row_index: usize,
    pub domain_log_size: u32,
    pub eval_log_size: u32,
}
impl<'a> EvalAtDomain<'a> {
    pub fn new(
        trace_eval: &'a TreeVec<Vec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
        vec_row: usize,
        random_coeff_powers: &'a [SecureField],
        domain_log_size: u32,
        eval_log_size: u32,
    ) -> Self {
        Self {
            trace_eval,
            col_index: vec![0; trace_eval.len()],
            vec_row,
            random_coeff_powers,
            row_res: PackedSecureField::zero(),
            constraint_row_index: 0,
            domain_log_size,
            eval_log_size,
        }
    }
}
impl<'a> EvalAtRow for EvalAtDomain<'a> {
    type F = PackedBaseField;
    type EF = PackedSecureField;

    // TODO: Remove all boundary checks.
    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        let col_index = self.col_index[interaction];
        self.col_index[interaction] += 1;
        offsets.map(|off| {
            // TODO: Optimize.
            if off == 0 {
                return self.trace_eval[interaction][col_index].data[self.vec_row];
            }
            PackedBaseField::from_array(std::array::from_fn(|i| {
                let index = offset_bit_reversed_circle_domain_index(
                    (self.vec_row << LOG_N_LANES) + i,
                    self.domain_log_size,
                    self.eval_log_size,
                    off,
                );
                self.trace_eval[interaction][col_index].at(index)
            }))
        })
    }
    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF>,
    {
        self.row_res +=
            PackedSecureField::broadcast(self.random_coeff_powers[self.constraint_row_index])
                * constraint;
        self.constraint_row_index += 1;
    }

    fn pow2(&self, i: u32) -> Self::F {
        PackedBaseField::broadcast(BaseField::from_u32_unchecked(1 << i))
    }

    fn combine_ef(values: [Self::F; 4]) -> Self::EF {
        PackedSecureField::from_packed_m31s(values)
    }
}

pub struct EvalAtPoint<'a> {
    pub mask: TreeVec<&'a ColumnVec<Vec<SecureField>>>,
    pub evaluation_accumulator: &'a mut PointEvaluationAccumulator,
    pub col_index: Vec<usize>,
    pub denom_inverse: SecureField,
}
impl<'a> EvalAtPoint<'a> {
    pub fn new(
        mask: TreeVec<&'a ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &'a mut PointEvaluationAccumulator,
        denom_inverse: SecureField,
    ) -> Self {
        let col_index = vec![0; mask.len()];
        Self {
            mask,
            evaluation_accumulator,
            col_index,
            denom_inverse,
        }
    }
}
impl<'a> EvalAtRow for EvalAtPoint<'a> {
    type F = SecureField;
    type EF = SecureField;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        _offsets: [isize; N],
    ) -> [Self::F; N] {
        let col_index = self.col_index[interaction];
        self.col_index[interaction] += 1;
        let mask = self.mask[interaction][col_index].clone();
        assert_eq!(mask.len(), N);
        mask.try_into().unwrap()
    }
    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF>,
    {
        self.evaluation_accumulator
            .accumulate(self.denom_inverse * constraint);
    }
    fn pow2(&self, i: u32) -> Self::F {
        BaseField::from_u32_unchecked(1 << i).into()
    }

    fn combine_ef(values: [Self::F; 4]) -> Self::EF {
        SecureCirclePoly::<CpuBackend>::eval_from_partial_evals(values)
    }
}

#[derive(Default)]
pub struct ConstraintCounter {
    pub mask_offsets: TreeVec<Vec<Vec<isize>>>,
    pub n_constraints: usize,
}
impl EvalAtRow for ConstraintCounter {
    type F = BaseField;
    type EF = SecureField;
    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        if self.mask_offsets.len() <= interaction {
            self.mask_offsets.resize(interaction + 1, vec![]);
        }
        self.mask_offsets[interaction].push(offsets.into_iter().collect());
        [BaseField::one(); N]
    }
    fn add_constraint<G>(&mut self, _constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF>,
    {
        self.n_constraints += 1;
    }

    fn pow2(&self, _i: u32) -> Self::F {
        BaseField::one()
    }

    fn combine_ef(_values: [Self::F; 4]) -> Self::EF {
        SecureField::one()
    }
}

pub struct AssertEvalAtRow<'a> {
    pub trace: &'a TreeVec<Vec<Vec<BaseField>>>,
    pub col_index: Vec<usize>,
    pub row: usize,
}
impl<'a> EvalAtRow for AssertEvalAtRow<'a> {
    type F = BaseField;
    type EF = SecureField;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        let col_index = self.col_index[interaction];
        self.col_index[interaction] += 1;
        offsets.map(|off| {
            self.trace[interaction][col_index][(self.row as isize + off)
                .rem_euclid(self.trace[interaction][col_index].len() as isize)
                as usize]
        })
    }

    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: std::ops::Mul<G, Output = Self::EF>,
    {
        let res = SecureField::one() * constraint;
        assert_eq!(res, SecureField::zero(), "row: {}", self.row);
    }

    fn pow2(&self, i: u32) -> Self::F {
        BaseField::from_u32_unchecked(1 << i)
    }

    fn combine_ef(values: [Self::F; 4]) -> Self::EF {
        SecureField::from_m31_array(values)
    }
}
