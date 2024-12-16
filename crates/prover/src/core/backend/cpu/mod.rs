mod accumulation;
mod blake2s;
pub mod circle;
mod fri;
mod grind;
pub mod lookups;
#[cfg(not(target_arch = "wasm32"))]
mod poseidon252;
pub mod quotients;

use std::fmt::Debug;

use num_traits::One;
use serde::{Deserialize, Serialize};

use super::{Backend, BackendForChannel, Column, ColumnOps, FieldOps};
use crate::core::backend::PowersGeneration;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;
use crate::core::lookups::mle::Mle;
use crate::core::poly::circle::{CircleEvaluation, CirclePoly};
use crate::core::utils::bit_reverse_index;
use crate::core::vcs::blake2_merkle::Blake2sMerkleChannel;
#[cfg(not(target_arch = "wasm32"))]
use crate::core::vcs::poseidon252_merkle::Poseidon252MerkleChannel;

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub struct CpuBackend;

impl Backend for CpuBackend {}
impl BackendForChannel<Blake2sMerkleChannel> for CpuBackend {}
#[cfg(not(target_arch = "wasm32"))]
impl BackendForChannel<Poseidon252MerkleChannel> for CpuBackend {}

/// Performs a naive bit-reversal permutation inplace.
///
/// # Panics
///
/// Panics if the length of the slice is not a power of two.
pub fn bit_reverse<T>(v: &mut [T]) {
    let n = v.len();
    assert!(n.is_power_of_two());
    let log_n = n.ilog2();
    for i in 0..n {
        let j = bit_reverse_index(i, log_n);
        if j > i {
            v.swap(i, j);
        }
    }
}

impl PowersGeneration for CpuBackend {
    fn generate_secure_powers(felt: SecureField, n_powers: usize) -> Vec<SecureField> {
        generate_secure_powers(felt, n_powers)
    }
}

/// Generates the first `n_powers` powers of `felt`.
pub fn generate_secure_powers(felt: SecureField, n_powers: usize) -> Vec<SecureField> {
    (0..n_powers)
        .scan(SecureField::one(), |acc, _| {
            let res = *acc;
            *acc *= felt;
            Some(res)
        })
        .collect()
}

impl<T: Debug + Clone + Default> ColumnOps<T> for CpuBackend {
    type Column = Vec<T>;

    fn bit_reverse_column(column: &mut Self::Column) {
        bit_reverse(column)
    }
}

impl<F: Field> FieldOps<F> for CpuBackend {
    /// Batch inversion using the Montgomery's trick.
    // TODO(Ohad): Benchmark this function.
    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
        F::batch_inverse(column, &mut dst[..]);
    }
}

impl<T: Debug + Clone + Default> Column<T> for Vec<T> {
    fn zeros(len: usize) -> Self {
        vec![T::default(); len]
    }
    #[allow(clippy::uninit_vec)]
    unsafe fn uninitialized(length: usize) -> Self {
        let mut data = Vec::with_capacity(length);
        data.set_len(length);
        data
    }
    fn to_cpu(&self) -> Vec<T> {
        self.clone()
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn at(&self, index: usize) -> T {
        self[index].clone()
    }
    fn set(&mut self, index: usize, value: T) {
        self[index] = value;
    }
}

pub type CpuCirclePoly = CirclePoly<CpuBackend>;
pub type CpuCircleEvaluation<F, EvalOrder> = CircleEvaluation<CpuBackend, F, EvalOrder>;
pub type CpuMle<F> = Mle<CpuBackend, F>;

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::One;
    use rand::prelude::*;
    use rand::rngs::SmallRng;

    use crate::core::backend::cpu::bit_reverse;
    use crate::core::backend::{Column, CpuBackend, FieldOps};
    use crate::core::fields::qm31::{SecureField, QM31};
    use crate::core::fields::FieldExpOps;
    use crate::qm31;

    #[test]
    fn bit_reverse_works() {
        let mut data = [0, 1, 2, 3, 4, 5, 6, 7];
        bit_reverse(&mut data);
        assert_eq!(data, [0, 4, 2, 6, 1, 5, 3, 7]);
    }

    #[test]
    #[should_panic]
    fn bit_reverse_non_power_of_two_size_fails() {
        let mut data = [0, 1, 2, 3, 4, 5];
        bit_reverse(&mut data);
    }

    #[test]
    fn generate_secure_powers_works() {
        let felt = qm31!(1, 2, 3, 4);
        let n_powers = 10;

        let powers = super::generate_secure_powers(felt, n_powers);

        assert_eq!(powers.len(), n_powers);
        assert_eq!(powers[0], SecureField::one());
        assert_eq!(powers[1], felt);
        assert_eq!(powers[7], felt.pow(7));
    }

    #[test]
    fn generate_empty_secure_powers_works() {
        let felt = qm31!(1, 2, 3, 4);
        let max_log_size = 0;

        let powers = super::generate_secure_powers(felt, max_log_size);

        assert_eq!(powers, vec![]);
    }

    #[test]
    fn batch_inverse_test() {
        let mut rng = SmallRng::seed_from_u64(0);
        let column = rng.gen::<[QM31; 16]>().to_vec();
        let expected = column.iter().map(|e| e.inverse()).collect_vec();
        let mut dst = Column::zeros(column.len());

        CpuBackend::batch_inverse(&column, &mut dst);

        assert_eq!(expected, dst);
    }
}
