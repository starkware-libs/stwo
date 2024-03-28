use num_traits::One;

use super::fields::qm31::SecureField;

pub trait IteratorMutExt<'a, T: 'a>: Iterator<Item = &'a mut T> {
    fn assign(self, other: impl IntoIterator<Item = T>)
    where
        Self: Sized,
    {
        self.zip(other).for_each(|(a, b)| *a = b);
    }
}

impl<'a, T: 'a, I: Iterator<Item = &'a mut T>> IteratorMutExt<'a, T> for I {}

pub(crate) fn bit_reverse_index(i: usize, log_size: u32) -> usize {
    if log_size == 0 {
        return i;
    }
    i.reverse_bits() >> (usize::BITS - log_size)
}

/// Performs a naive bit-reversal permutation inplace.
///
/// # Panics
///
/// Panics if the length of the slice is not a power of two.
// TODO: Implement cache friendly implementation.
// TODO(spapini): Move this to the cpu backend.
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

pub fn generate_secure_powers(felt: SecureField, max_log_size: usize) -> Vec<SecureField> {
    let mut res = vec![SecureField::one()];
    for _ in 1..=max_log_size {
        res.push(*res.last().unwrap() * felt);
    }
    res
}

#[cfg(test)]
mod tests {
    use num_traits::One;

    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::FieldExpOps;
    use crate::core::utils::bit_reverse;
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
        let max_log_size = 10;

        let powers = super::generate_secure_powers(felt, max_log_size);

        assert_eq!(powers.len(), max_log_size + 1);
        assert_eq!(powers[0], SecureField::one());
        assert_eq!(powers[1], felt);
        assert_eq!(powers[7], felt.pow(7));
    }
}
