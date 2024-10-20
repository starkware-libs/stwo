use std::array;
use std::cell::{RefCell, RefMut};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};
use std::rc::Rc;

use num_traits::{One, Zero};

use super::EvalAtRow;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::pcs::TreeVec;

/// Collects information about the constraints.
/// This includes mask offsets and columns at each interaction, the number of constraints and number
/// of arithmetic operations.
#[derive(Default)]
pub struct InfoEvaluator {
    pub mask_offsets: TreeVec<Vec<Vec<isize>>>,
    pub n_constraints: usize,
    pub arithmetic_counts: ArithmeticCounts,
}
impl InfoEvaluator {
    pub fn new() -> Self {
        Self::default()
    }
}
impl EvalAtRow for InfoEvaluator {
    type F = BaseFieldCounter;
    type EF = SecureFieldCounter;
    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        // Check if requested a mask from a new interaction
        if self.mask_offsets.len() <= interaction {
            // Extend `mask_offsets` so that `interaction` is the last index.
            self.mask_offsets.resize(interaction + 1, vec![]);
        }
        self.mask_offsets[interaction].push(offsets.into_iter().collect());
        array::from_fn(|_| BaseFieldCounter::one())
    }
    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF>,
    {
        let lin_combination = SecureFieldCounter::one() + SecureFieldCounter::one() * constraint;
        self.arithmetic_counts.merge(lin_combination.drain());
        self.n_constraints += 1;
    }

    fn combine_ef(values: [Self::F; 4]) -> Self::EF {
        let mut res = SecureFieldCounter::zero();
        values.map(|v| res.merge(v));
        res
    }
}

/// Stores a count of field operations.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ArithmeticCounts {
    /// Number of `ExtensionField * ExtensionField` operations.
    pub n_ef_mul_ef: usize,
    /// Number of `ExtensionField * BaseField` operations.
    pub n_ef_mul_f: usize,
    /// Number of `ExtensionField + ExtensionField` operations.
    pub n_ef_add_ef: usize,
    /// Number of `ExtensionField + BaseField` operations.
    pub n_ef_add_f: usize,
    /// Number of `BaseField * BaseField` operations.
    pub n_f_mul_f: usize,
    /// Number of `BaseField + BaseField` operations.
    pub n_f_add_f: usize,
}

impl ArithmeticCounts {
    fn merge(&mut self, other: ArithmeticCounts) {
        self.n_ef_mul_ef += other.n_ef_mul_ef;
        self.n_ef_mul_f += other.n_ef_mul_f;
        self.n_ef_add_f += other.n_ef_add_f;
        self.n_ef_add_ef += other.n_ef_add_ef;
        self.n_f_mul_f += other.n_f_mul_f;
        self.n_f_add_f += other.n_f_add_f;
    }
}

#[derive(Debug, Default, Clone)]
pub struct ArithmeticCounter<const IS_EXT_FIELD: bool>(Rc<RefCell<ArithmeticCounts>>);

pub type BaseFieldCounter = ArithmeticCounter<false>;

pub type SecureFieldCounter = ArithmeticCounter<true>;

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

impl Add<BaseFieldCounter> for SecureFieldCounter {
    type Output = Self;

    fn add(mut self, rhs: BaseFieldCounter) -> Self {
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

impl Mul<BaseFieldCounter> for SecureFieldCounter {
    type Output = SecureFieldCounter;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(mut self, rhs: BaseFieldCounter) -> Self {
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

impl AddAssign<BaseField> for BaseFieldCounter {
    fn add_assign(&mut self, _rhs: BaseField) {
        *self = self.clone() + BaseFieldCounter::zero()
    }
}

impl Mul<BaseField> for BaseFieldCounter {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, _rhs: BaseField) -> Self {
        self * BaseFieldCounter::zero()
    }
}

impl Mul<SecureField> for SecureFieldCounter {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, _rhs: SecureField) -> Self {
        self * SecureFieldCounter::zero()
    }
}

impl Add<SecureField> for BaseFieldCounter {
    type Output = SecureFieldCounter;

    fn add(self, _rhs: SecureField) -> SecureFieldCounter {
        SecureFieldCounter::zero() + self
    }
}

impl Add<SecureField> for SecureFieldCounter {
    type Output = Self;

    fn add(self, _rhs: SecureField) -> Self {
        self + SecureFieldCounter::zero()
    }
}

impl Sub<SecureField> for SecureFieldCounter {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: SecureField) -> Self {
        // Tread subtraction as addition
        self + rhs
    }
}

impl Mul<SecureField> for BaseFieldCounter {
    type Output = SecureFieldCounter;

    fn mul(self, _rhs: SecureField) -> SecureFieldCounter {
        SecureFieldCounter::zero() * self
    }
}

impl From<BaseField> for BaseFieldCounter {
    fn from(_value: BaseField) -> Self {
        Self::one()
    }
}

impl From<SecureField> for SecureFieldCounter {
    fn from(_value: SecureField) -> Self {
        Self::one()
    }
}

impl From<BaseFieldCounter> for SecureFieldCounter {
    fn from(value: BaseFieldCounter) -> Self {
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

    use super::SecureFieldCounter;
    use crate::constraint_framework::info::{ArithmeticCounts, BaseFieldCounter};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;

    #[test]
    fn test_arithmetic_counter() {
        const N_EF_MUL_EF: usize = 1;
        const N_EF_MUL_F: usize = 2;
        const N_EF_MUL_ASSIGN_EF: usize = 1;
        const N_EF_MUL_SECURE_FIELD: usize = 3;
        const N_EF_ADD_EF: usize = 4;
        const N_EF_ADD_ASSIGN_EF: usize = 4;
        const N_EF_ADD_F: usize = 5;
        const N_EF_NEG: usize = 6;
        const N_EF_SUB_EF: usize = 7;
        const N_F_MUL_F: usize = 8;
        const N_F_MUL_ASSIGN_F: usize = 8;
        const N_F_MUL_BASE_FIELD: usize = 9;
        const N_F_ADD_F: usize = 10;
        const N_F_ADD_ASSIGN_F: usize = 4;
        const N_F_ADD_ASSIGN_BASE_FIELD: usize = 4;
        const N_F_NEG: usize = 11;
        const N_F_SUB_F: usize = 12;
        let mut ef = SecureFieldCounter::zero();
        let mut f = BaseFieldCounter::zero();

        (0..N_EF_MUL_EF).for_each(|_| ef = ef.clone() * ef.clone());
        (0..N_EF_MUL_F).for_each(|_| ef = ef.clone() * f.clone());
        (0..N_EF_MUL_SECURE_FIELD).for_each(|_| ef = ef.clone() * SecureField::one());
        (0..N_EF_MUL_ASSIGN_EF).for_each(|_| ef *= ef.clone());
        (0..N_EF_ADD_EF).for_each(|_| ef = ef.clone() + ef.clone());
        (0..N_EF_ADD_ASSIGN_EF).for_each(|_| ef += ef.clone());
        (0..N_EF_ADD_F).for_each(|_| ef = ef.clone() + f.clone());
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
                n_ef_mul_f: N_EF_MUL_F,
                n_ef_add_ef: N_EF_ADD_EF + N_EF_NEG + N_EF_SUB_EF + N_EF_ADD_ASSIGN_EF,
                n_ef_add_f: N_EF_ADD_F,
                n_f_mul_f: N_F_MUL_F + N_F_MUL_BASE_FIELD + N_F_MUL_ASSIGN_F,
                n_f_add_f: N_F_ADD_F
                    + N_F_NEG
                    + N_F_SUB_F
                    + N_F_ADD_ASSIGN_BASE_FIELD
                    + N_F_ADD_ASSIGN_F,
            }
        );
    }
}
