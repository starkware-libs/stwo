use std::array;
use std::cell::{RefCell, RefMut};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};
use std::rc::Rc;

use num_traits::{One, Zero};

use super::logup::LogupAtRow;
use super::preprocessed_columns::PreProcessedColumnId;
use super::{EvalAtRow, INTERACTION_TRACE_IDX};
use crate::constraint_framework::PREPROCESSED_TRACE_IDX;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::lookups::utils::Fraction;
use crate::core::pcs::TreeVec;

/// Collects information about the constraints.
/// This includes mask offsets and columns at each interaction, the number of constraints and number
/// of arithmetic operations.
#[derive(Default)]
pub struct InfoEvaluator {
    pub mask_offsets: TreeVec<Vec<Vec<isize>>>,
    pub n_constraints: usize,
    pub preprocessed_columns: Vec<PreProcessedColumnId>,
    pub logup: LogupAtRow<Self>,
    pub arithmetic_counts: ArithmeticCounts,
}
impl InfoEvaluator {
    pub fn new(
        log_size: u32,
        preprocessed_columns: Vec<PreProcessedColumnId>,
        claimed_sum: SecureField,
    ) -> Self {
        Self {
            mask_offsets: Default::default(),
            n_constraints: Default::default(),
            preprocessed_columns,
            logup: LogupAtRow::new(INTERACTION_TRACE_IDX, claimed_sum, log_size),
            arithmetic_counts: Default::default(),
        }
    }

    /// Create an empty `InfoEvaluator`, to measure components before their size and logup sums are
    /// available.
    pub fn empty() -> Self {
        Self::new(16, vec![], SecureField::default())
    }
}
impl EvalAtRow for InfoEvaluator {
    type F = FieldCounter;
    type EF = ExtensionFieldCounter;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        assert!(
            interaction != PREPROCESSED_TRACE_IDX,
            "Preprocessed should be accesses with `get_preprocessed_column`",
        );

        // Check if requested a mask from a new interaction
        if self.mask_offsets.len() <= interaction {
            // Extend `mask_offsets` so that `interaction` is the last index.
            self.mask_offsets.resize(interaction + 1, vec![]);
        }
        self.mask_offsets[interaction].push(offsets.into_iter().collect());
        array::from_fn(|_| FieldCounter::one())
    }

    fn get_preprocessed_column(&mut self, column: PreProcessedColumnId) -> Self::F {
        self.preprocessed_columns.push(column);
        FieldCounter::one()
    }

    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF>,
    {
        let lin_combination =
            ExtensionFieldCounter::one() + ExtensionFieldCounter::one() * constraint;
        self.arithmetic_counts.merge(lin_combination.drain());
        self.n_constraints += 1;
    }

    fn combine_ef(values: [Self::F; 4]) -> Self::EF {
        let mut res = ExtensionFieldCounter::zero();
        values.map(|v| res.merge(v));
        res
    }

    super::logup_proxy!();
}

/// Stores a count of field operations.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ArithmeticCounts {
    /// Number of [`EvalAtRow::EF`] * [`EvalAtRow::EF`] operations.
    pub n_ef_mul_ef: usize,
    /// Number of [`EvalAtRow::EF`] * [`EvalAtRow::F`] operations.
    pub n_ef_mul_f: usize,
    /// Number of [`EvalAtRow::F`] * [`BaseField`] operations.
    pub n_ef_mul_base_field: usize,
    /// Number of [`EvalAtRow::EF`] + [`EvalAtRow::EF`] operations.
    pub n_ef_add_ef: usize,
    /// Number of [`EvalAtRow::EF`] + [`EvalAtRow::F`] operations.
    pub n_ef_add_f: usize,
    /// Number of [`EvalAtRow::EF`] * [`BaseField`] operations.
    pub n_ef_add_base_field: usize,
    /// Number of [`EvalAtRow::F`] * [`EvalAtRow::F`] operations.
    pub n_f_mul_f: usize,
    /// Number of [`EvalAtRow::F`] * [`BaseField`] operations.
    pub n_f_mul_base_field: usize,
    /// Number of [`EvalAtRow::F`] + [`EvalAtRow::F`] operations.
    pub n_f_add_f: usize,
    /// Number of [`EvalAtRow::F`] + [`BaseField`] operations.
    pub n_f_add_base_field: usize,
}

impl ArithmeticCounts {
    fn merge(&mut self, other: ArithmeticCounts) {
        let Self {
            n_ef_mul_ef,
            n_ef_mul_f,
            n_ef_mul_base_field,
            n_ef_add_ef,
            n_ef_add_f,
            n_ef_add_base_field,
            n_f_mul_f,
            n_f_mul_base_field,
            n_f_add_f,
            n_f_add_base_field,
        } = self;

        *n_ef_mul_ef += other.n_ef_mul_ef;
        *n_ef_mul_f += other.n_ef_mul_f;
        *n_ef_mul_base_field += other.n_ef_mul_base_field;
        *n_ef_add_f += other.n_ef_add_f;
        *n_ef_add_base_field += other.n_ef_add_base_field;
        *n_ef_add_ef += other.n_ef_add_ef;
        *n_f_mul_f += other.n_f_mul_f;
        *n_f_mul_base_field += other.n_f_mul_base_field;
        *n_f_add_f += other.n_f_add_f;
        *n_f_add_base_field += other.n_f_add_base_field;
    }
}

#[derive(Debug, Default, Clone)]
pub struct ArithmeticCounter<const IS_EXT_FIELD: bool>(Rc<RefCell<ArithmeticCounts>>);

/// Counts operations on [`EvalAtRow::F`].
pub type FieldCounter = ArithmeticCounter<false>;

/// Counts operations on [`EvalAtRow::EF`].
pub type ExtensionFieldCounter = ArithmeticCounter<true>;

impl<const IS_EXT_FIELD: bool> ArithmeticCounter<IS_EXT_FIELD> {
    fn merge<const OTHER_IS_EXT_FIELD: bool>(
        &mut self,
        other: ArithmeticCounter<OTHER_IS_EXT_FIELD>,
    ) {
        // Skip if they come from the same source.
        if Rc::ptr_eq(&self.0, &other.0) {
            return;
        }

        self.counts().merge(other.drain());
    }

    fn drain(self) -> ArithmeticCounts {
        self.0.take()
    }

    fn counts(&mut self) -> RefMut<'_, ArithmeticCounts> {
        self.0.borrow_mut()
    }
}

impl<const IS_EXT_FIELD: bool> Zero for ArithmeticCounter<IS_EXT_FIELD> {
    fn zero() -> Self {
        Self::default()
    }

    fn is_zero(&self) -> bool {
        // TODO(andrew): Consider removing Zero from EvalAtRow::F, EvalAtRow::EF since is_zero
        // doesn't make sense. Creating zero elements does though.
        panic!()
    }
}

impl<const IS_EXT_FIELD: bool> One for ArithmeticCounter<IS_EXT_FIELD> {
    fn one() -> Self {
        Self::default()
    }
}

impl<const IS_EXT_FIELD: bool> Add for ArithmeticCounter<IS_EXT_FIELD> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        self.merge(rhs);
        {
            let mut counts = self.counts();
            match IS_EXT_FIELD {
                true => counts.n_ef_add_ef += 1,
                false => counts.n_f_add_f += 1,
            }
        }
        self
    }
}

impl<const IS_EXT_FIELD: bool> Sub for ArithmeticCounter<IS_EXT_FIELD> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        // Treat as addition.
        self + rhs
    }
}

impl Add<FieldCounter> for ExtensionFieldCounter {
    type Output = Self;

    fn add(mut self, rhs: FieldCounter) -> Self {
        self.merge(rhs);
        self.counts().n_ef_add_f += 1;
        self
    }
}

impl<const IS_EXT_FIELD: bool> Mul for ArithmeticCounter<IS_EXT_FIELD> {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self {
        self.merge(rhs);
        {
            let mut counts = self.counts();
            match IS_EXT_FIELD {
                true => counts.n_ef_mul_ef += 1,
                false => counts.n_f_mul_f += 1,
            }
        }
        self
    }
}

impl Mul<FieldCounter> for ExtensionFieldCounter {
    type Output = ExtensionFieldCounter;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(mut self, rhs: FieldCounter) -> Self {
        self.merge(rhs);
        self.counts().n_ef_mul_f += 1;
        self
    }
}

impl<const IS_EXT_FIELD: bool> MulAssign for ArithmeticCounter<IS_EXT_FIELD> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs
    }
}

impl<const IS_EXT_FIELD: bool> AddAssign for ArithmeticCounter<IS_EXT_FIELD> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs
    }
}

impl AddAssign<BaseField> for FieldCounter {
    fn add_assign(&mut self, _rhs: BaseField) {
        self.counts().n_f_add_base_field += 1;
    }
}

impl Mul<BaseField> for FieldCounter {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(mut self, _rhs: BaseField) -> Self {
        self.counts().n_f_mul_base_field += 1;
        self
    }
}

impl Mul<BaseField> for ExtensionFieldCounter {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(mut self, _rhs: BaseField) -> Self {
        self.counts().n_ef_mul_base_field += 1;
        self
    }
}

impl Mul<SecureField> for ExtensionFieldCounter {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, _rhs: SecureField) -> Self {
        self * ExtensionFieldCounter::zero()
    }
}

impl Add<SecureField> for FieldCounter {
    type Output = ExtensionFieldCounter;

    fn add(self, _rhs: SecureField) -> ExtensionFieldCounter {
        ExtensionFieldCounter::zero() + self
    }
}

impl Add<BaseField> for ExtensionFieldCounter {
    type Output = Self;

    fn add(mut self, _rhs: BaseField) -> Self {
        self.counts().n_ef_add_base_field += 1;
        self
    }
}

impl Add<SecureField> for ExtensionFieldCounter {
    type Output = Self;

    fn add(self, _rhs: SecureField) -> Self {
        self + ExtensionFieldCounter::zero()
    }
}

impl Sub<SecureField> for ExtensionFieldCounter {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: SecureField) -> Self {
        // Tread subtraction as addition
        self + rhs
    }
}

impl Mul<SecureField> for FieldCounter {
    type Output = ExtensionFieldCounter;

    fn mul(self, _rhs: SecureField) -> ExtensionFieldCounter {
        ExtensionFieldCounter::zero() * self
    }
}

impl From<BaseField> for FieldCounter {
    fn from(_value: BaseField) -> Self {
        Self::one()
    }
}

impl From<SecureField> for ExtensionFieldCounter {
    fn from(_value: SecureField) -> Self {
        Self::one()
    }
}

impl From<FieldCounter> for ExtensionFieldCounter {
    fn from(value: FieldCounter) -> Self {
        Self(value.0)
    }
}

impl<const IS_EXT_FIELD: bool> Neg for ArithmeticCounter<IS_EXT_FIELD> {
    type Output = Self;

    fn neg(self) -> Self {
        // Treat as addition.
        self + ArithmeticCounter::<IS_EXT_FIELD>::zero()
    }
}

impl<const IS_EXT_FIELD: bool> FieldExpOps for ArithmeticCounter<IS_EXT_FIELD> {
    fn inverse(&self) -> Self {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use num_traits::{One, Zero};

    use super::ExtensionFieldCounter;
    use crate::constraint_framework::info::{ArithmeticCounts, FieldCounter};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;

    #[test]
    fn test_arithmetic_counter() {
        const N_EF_MUL_EF: usize = 1;
        const N_EF_MUL_F: usize = 2;
        const N_EF_MUL_BASE_FIELD: usize = 3;
        const N_EF_MUL_ASSIGN_EF: usize = 4;
        const N_EF_MUL_SECURE_FIELD: usize = 5;
        const N_EF_ADD_EF: usize = 6;
        const N_EF_ADD_ASSIGN_EF: usize = 7;
        const N_EF_ADD_F: usize = 8;
        const N_EF_NEG: usize = 9;
        const N_EF_SUB_EF: usize = 10;
        const N_EF_ADD_BASE_FIELD: usize = 11;
        const N_F_MUL_F: usize = 12;
        const N_F_MUL_ASSIGN_F: usize = 13;
        const N_F_MUL_BASE_FIELD: usize = 14;
        const N_F_ADD_F: usize = 15;
        const N_F_ADD_ASSIGN_F: usize = 16;
        const N_F_ADD_ASSIGN_BASE_FIELD: usize = 17;
        const N_F_NEG: usize = 18;
        const N_F_SUB_F: usize = 19;
        let mut ef = ExtensionFieldCounter::zero();
        let mut f = FieldCounter::zero();

        (0..N_EF_MUL_EF).for_each(|_| ef = ef.clone() * ef.clone());
        (0..N_EF_MUL_F).for_each(|_| ef = ef.clone() * f.clone());
        (0..N_EF_MUL_BASE_FIELD).for_each(|_| ef = ef.clone() * BaseField::one());
        (0..N_EF_MUL_SECURE_FIELD).for_each(|_| ef = ef.clone() * SecureField::one());
        (0..N_EF_MUL_ASSIGN_EF).for_each(|_| ef *= ef.clone());
        (0..N_EF_ADD_EF).for_each(|_| ef = ef.clone() + ef.clone());
        (0..N_EF_ADD_ASSIGN_EF).for_each(|_| ef += ef.clone());
        (0..N_EF_ADD_F).for_each(|_| ef = ef.clone() + f.clone());
        (0..N_EF_ADD_BASE_FIELD).for_each(|_| ef = ef.clone() + BaseField::one());
        (0..N_EF_NEG).for_each(|_| ef = -ef.clone());
        (0..N_EF_SUB_EF).for_each(|_| ef = ef.clone() - ef.clone());
        (0..N_F_MUL_F).for_each(|_| f = f.clone() * f.clone());
        (0..N_F_MUL_ASSIGN_F).for_each(|_| f *= f.clone());
        (0..N_F_MUL_BASE_FIELD).for_each(|_| f = f.clone() * BaseField::one());
        (0..N_F_ADD_F).for_each(|_| f = f.clone() + f.clone());
        (0..N_F_ADD_ASSIGN_F).for_each(|_| f += f.clone());
        (0..N_F_ADD_ASSIGN_BASE_FIELD).for_each(|_| f += BaseField::one());
        (0..N_F_NEG).for_each(|_| f = -f.clone());
        (0..N_F_SUB_F).for_each(|_| f = f.clone() - f.clone());
        let mut res = f.drain();
        res.merge(ef.drain());

        assert_eq!(
            res,
            ArithmeticCounts {
                n_ef_mul_ef: N_EF_MUL_EF + N_EF_MUL_SECURE_FIELD + N_EF_MUL_ASSIGN_EF,
                n_ef_mul_base_field: N_EF_MUL_BASE_FIELD,
                n_ef_mul_f: N_EF_MUL_F,
                n_ef_add_ef: N_EF_ADD_EF + N_EF_NEG + N_EF_SUB_EF + N_EF_ADD_ASSIGN_EF,
                n_ef_add_f: N_EF_ADD_F,
                n_ef_add_base_field: N_EF_ADD_BASE_FIELD,
                n_f_mul_f: N_F_MUL_F + N_F_MUL_ASSIGN_F,
                n_f_mul_base_field: N_F_MUL_BASE_FIELD,
                n_f_add_f: N_F_ADD_F + N_F_NEG + N_F_SUB_F + N_F_ADD_ASSIGN_F,
                n_f_add_base_field: N_F_ADD_ASSIGN_BASE_FIELD,
            }
        );
    }
}
