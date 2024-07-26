use std::ops::{Add, AddAssign, Mul, MulAssign, Sub};
use std::sync::RwLock;

use num_traits::{One, Zero};
use stwo_prover::constraint_framework::EvalAtRow;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::FieldExpOps;
use stwo_prover::examples::poseidon::PoseidonEval;

use crate::cost_counter::CostCount;

/// Since uses static mem for counting make sure it only runs the computation once.
static HAS_RUN: RwLock<bool> = RwLock::new(false);

static BASE_ADD_BASE_COUNTER: RwLock<usize> = RwLock::new(0);
static BASE_MUL_BASE_COUNTER: RwLock<usize> = RwLock::new(0);
static EXT_MUL_EXT_COUNTER: RwLock<usize> = RwLock::new(0);
static EXT_MUL_BASE_COUNTER: RwLock<usize> = RwLock::new(0);
static EXT_ADD_EXT_COUNTER: RwLock<usize> = RwLock::new(0);
static EXT_ADD_BASE_COUNTER: RwLock<usize> = RwLock::new(0);
static COLUMN_COUNTER: RwLock<usize> = RwLock::new(0);

#[derive(Debug, Clone, Copy)]
struct BaseFieldCounter;

impl Add for BaseFieldCounter {
    type Output = Self;

    fn add(self, _rhs: Self) -> Self::Output {
        let mut counter = BASE_ADD_BASE_COUNTER.write().unwrap();
        *counter += 1;
        Self
    }
}

impl Sub for BaseFieldCounter {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, _rhs: Self) -> Self::Output {
        let mut counter = BASE_ADD_BASE_COUNTER.write().unwrap();
        *counter += 1;
        Self
    }
}

impl AddAssign for BaseFieldCounter {
    fn add_assign(&mut self, _rhs: Self) {
        let mut counter = BASE_ADD_BASE_COUNTER.write().unwrap();
        *counter += 1;
    }
}

impl AddAssign<BaseField> for BaseFieldCounter {
    fn add_assign(&mut self, _rhs: BaseField) {
        let mut counter = BASE_ADD_BASE_COUNTER.write().unwrap();
        *counter += 1;
    }
}

impl Mul<SecureField> for BaseFieldCounter {
    type Output = ExtFieldCounter;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, _rhs: SecureField) -> Self::Output {
        let mut counter = EXT_MUL_BASE_COUNTER.write().unwrap();
        *counter += 1;
        ExtFieldCounter
    }
}

impl Mul for BaseFieldCounter {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, _rhs: Self) -> Self::Output {
        let mut counter = BASE_MUL_BASE_COUNTER.write().unwrap();
        *counter += 1;
        Self
    }
}

impl Mul<BaseField> for BaseFieldCounter {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, _rhs: BaseField) -> Self::Output {
        let mut counter = BASE_MUL_BASE_COUNTER.write().unwrap();
        *counter += 1;
        Self
    }
}

impl Add<SecureField> for BaseFieldCounter {
    type Output = ExtFieldCounter;

    fn add(self, _rhs: SecureField) -> Self::Output {
        let mut counter = EXT_MUL_BASE_COUNTER.write().unwrap();
        *counter += 1;
        ExtFieldCounter
    }
}

impl From<BaseField> for BaseFieldCounter {
    fn from(_value: BaseField) -> Self {
        Self
    }
}

impl MulAssign for BaseFieldCounter {
    #[allow(clippy::suspicious_op_assign_impl)]
    fn mul_assign(&mut self, _rhs: Self) {
        let mut counter = BASE_MUL_BASE_COUNTER.write().unwrap();
        *counter += 1;
    }
}

impl One for BaseFieldCounter {
    fn one() -> Self {
        Self
    }
}

impl Zero for BaseFieldCounter {
    fn zero() -> Self {
        Self
    }

    fn is_zero(&self) -> bool {
        unimplemented!()
    }
}

impl FieldExpOps for BaseFieldCounter {
    fn inverse(&self) -> Self {
        unimplemented!()
    }
}

#[derive(Debug, Clone, Copy)]
struct ExtFieldCounter;

impl Add<SecureField> for ExtFieldCounter {
    type Output = Self;

    fn add(self, _rhs: SecureField) -> Self::Output {
        let mut counter = EXT_ADD_EXT_COUNTER.write().unwrap();
        *counter += 1;
        Self
    }
}

impl Add for ExtFieldCounter {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, _rhs: Self) -> Self::Output {
        let mut counter = EXT_ADD_EXT_COUNTER.write().unwrap();
        *counter += 1;
        Self
    }
}

impl Sub for ExtFieldCounter {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, _rhs: Self) -> Self::Output {
        let mut counter = EXT_ADD_EXT_COUNTER.write().unwrap();
        *counter += 1;
        Self
    }
}

impl Mul for ExtFieldCounter {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, _rhs: Self) -> Self::Output {
        let mut counter = EXT_MUL_EXT_COUNTER.write().unwrap();
        *counter += 1;
        Self
    }
}

impl Mul<SecureField> for ExtFieldCounter {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, _rhs: SecureField) -> Self::Output {
        let mut counter = EXT_MUL_EXT_COUNTER.write().unwrap();
        *counter += 1;
        Self
    }
}

impl Mul<BaseFieldCounter> for ExtFieldCounter {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, _rhs: BaseFieldCounter) -> Self::Output {
        let mut counter = EXT_MUL_BASE_COUNTER.write().unwrap();
        *counter += 1;
        Self
    }
}

impl Add<BaseFieldCounter> for ExtFieldCounter {
    type Output = Self;

    fn add(self, _rhs: BaseFieldCounter) -> Self::Output {
        let mut counter = EXT_ADD_BASE_COUNTER.write().unwrap();
        *counter += 1;
        Self
    }
}

impl Sub<SecureField> for ExtFieldCounter {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, _rhs: SecureField) -> Self::Output {
        let mut counter = EXT_ADD_EXT_COUNTER.write().unwrap();
        *counter += 1;
        Self
    }
}

impl Zero for ExtFieldCounter {
    fn zero() -> Self {
        Self
    }

    fn is_zero(&self) -> bool {
        unimplemented!()
    }
}

impl One for ExtFieldCounter {
    fn one() -> Self {
        Self
    }
}

impl From<BaseFieldCounter> for ExtFieldCounter {
    fn from(_value: BaseFieldCounter) -> Self {
        Self
    }
}

#[derive(Debug, Clone, Copy)]
struct ArithmeticCounter;

impl EvalAtRow for ArithmeticCounter {
    type F = BaseFieldCounter;
    type EF = ExtFieldCounter;

    // TODO(spapini): Remove all boundary checks.
    fn next_interaction_mask<const N: usize>(
        &mut self,
        _interaction: usize,
        _offsets: [isize; N],
    ) -> [Self::F; N] {
        *COLUMN_COUNTER.write().unwrap() += 1;
        [BaseFieldCounter; N]
    }

    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: std::ops::Mul<G, Output = Self::EF>,
    {
        let acc = ExtFieldCounter;
        let random_coeff = ExtFieldCounter;
        let _ = acc + random_coeff * constraint;
    }

    fn combine_ef(_values: [Self::F; 4]) -> Self::EF {
        ExtFieldCounter
    }
}

pub fn poseidon_constraint_cost() -> CostCount {
    if !*HAS_RUN.read().unwrap() {
        *HAS_RUN.write().unwrap() = true;
        PoseidonEval {
            eval: ArithmeticCounter,
        }
        .eval();
    }

    let res = CostCount {
        base_add_base: *BASE_ADD_BASE_COUNTER.read().unwrap(),
        base_mul_base: *BASE_MUL_BASE_COUNTER.read().unwrap(),
        ext_mul_ext: *EXT_MUL_EXT_COUNTER.read().unwrap(),
        ext_mul_base: *EXT_MUL_BASE_COUNTER.read().unwrap(),
        ext_add_ext: *EXT_ADD_EXT_COUNTER.read().unwrap(),
        ext_add_base: *EXT_ADD_BASE_COUNTER.read().unwrap(),
        hash_compression: 0,
    };

    // Add quotient cost (amortized accross all constraints).
    // Includes calculating: `numer * denom_inv`, `denom_inv` from denom
    let quotient_cost = CostCount::base_mul_base(1) + CostCount::batch_inverse_base_values(1);

    res + quotient_cost
}
