use crate::core::fields::fraction::Frac;
use crate::core::fields::m31::M31;
use more_asserts::assert_gt;
use num_traits::{One, Zero};

pub fn generate_multiplicities(column: &[M31], max_value: u32) -> Vec<M31> {
    let mut multiplicities = vec![M31::zero(); max_value as usize];
    for value in column.iter() {
        assert_gt!(max_value, value.to_u32());
        multiplicities[value.to_u32() as usize] += M31::one();
    }
    multiplicities
}

pub fn sum_inverses(column: &[M31], random_elem: M31) -> Frac<M31> {
    let mut sum = Frac::new(M31::zero(), M31::one());
    for elem in column.iter() {
        let shifted_elem = random_elem - *elem;
        sum += shifted_elem;
    }
    sum
}

pub fn sum_multiplicities(multiplicities: &[M31], random_elem: M31) -> Frac<M31> {
    let mut sum = Frac::new(M31::zero(), M31::one());
    for (index, multiplicity) in multiplicities.iter().enumerate() {
        // Element is the index of the element in the column.
        let elem = M31::from_u32_unchecked(index as u32);
        let shifted_elem = random_elem - elem;
        sum += Frac::new(*multiplicity, shifted_elem);
    }
    sum
}

#[test]
fn test_logup() {
    let column = [
        M31::from_u32_unchecked(1),
        M31::from_u32_unchecked(2),
        M31::from_u32_unchecked(1),
        M31::from_u32_unchecked(1),
    ];
    let max_value = 3;
    let multiplicities = generate_multiplicities(&column, max_value);

    let random_elem = M31::from_u32_unchecked(13);
    let original_logup = sum_inverses(&column, random_elem);
    let multiplicities_logup = sum_multiplicities(&multiplicities, random_elem);

    assert_eq!(original_logup, multiplicities_logup);
}
