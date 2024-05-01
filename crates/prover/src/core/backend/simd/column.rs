use bytemuck::{cast_slice, cast_slice_mut, Zeroable};
use itertools::Itertools;
use num_traits::Zero;

use super::m31::{PackedBaseField, N_LANES};
use super::qm31::PackedSecureField;
use crate::core::backend::Column;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;

#[derive(Clone, Debug)]
pub struct BaseFieldVec {
    pub data: Vec<PackedBaseField>,
    pub length: usize,
}

impl AsRef<[BaseField]> for BaseFieldVec {
    fn as_ref(&self) -> &[BaseField] {
        &cast_slice(&self.data)[..self.length]
    }
}

impl AsMut<[BaseField]> for BaseFieldVec {
    fn as_mut(&mut self) -> &mut [BaseField] {
        &mut cast_slice_mut(&mut self.data)[..self.length]
    }
}

impl Column<BaseField> for BaseFieldVec {
    fn zeros(length: usize) -> Self {
        let data = vec![PackedBaseField::zeroed(); length.div_ceil(N_LANES)];
        Self { data, length }
    }

    fn to_cpu(&self) -> Vec<BaseField> {
        self.as_ref().to_vec()
    }

    fn len(&self) -> usize {
        self.length
    }

    fn at(&self, index: usize) -> BaseField {
        self.data[index / N_LANES].to_array()[index % N_LANES]
    }
}

impl FromIterator<BaseField> for BaseFieldVec {
    fn from_iter<I: IntoIterator<Item = BaseField>>(iter: I) -> Self {
        let mut chunks = iter.into_iter().array_chunks();
        let mut data = (&mut chunks).map(PackedBaseField::from_array).collect_vec();
        let mut length = data.len() * N_LANES;

        if let Some(remainder) = chunks.into_remainder() {
            if !remainder.is_empty() {
                length += remainder.len();
                let mut last = [BaseField::zero(); N_LANES];
                last[..remainder.len()].copy_from_slice(remainder.as_slice());
                data.push(PackedBaseField::from_array(last));
            }
        }

        Self { data, length }
    }
}

#[derive(Clone, Debug)]
pub struct SecureFieldVec {
    pub data: Vec<PackedSecureField>,
    pub length: usize,
}

impl Column<SecureField> for SecureFieldVec {
    fn zeros(length: usize) -> Self {
        Self {
            data: vec![PackedSecureField::zeroed(); length.div_ceil(N_LANES)],
            length,
        }
    }

    fn to_cpu(&self) -> Vec<SecureField> {
        self.data
            .iter()
            .flat_map(|x| x.to_array())
            .take(self.length)
            .collect()
    }

    fn len(&self) -> usize {
        self.length
    }

    fn at(&self, index: usize) -> SecureField {
        self.data[index / N_LANES].to_array()[index % N_LANES]
    }
}

impl FromIterator<SecureField> for SecureFieldVec {
    fn from_iter<I: IntoIterator<Item = SecureField>>(iter: I) -> Self {
        let mut chunks = iter.into_iter().array_chunks();
        let mut data = (&mut chunks)
            .map(PackedSecureField::from_array)
            .collect_vec();
        let mut length = data.len() * N_LANES;

        if let Some(remainder) = chunks.into_remainder() {
            if !remainder.is_empty() {
                length += remainder.len();
                let mut last = [SecureField::zero(); N_LANES];
                last[..remainder.len()].copy_from_slice(remainder.as_slice());
                data.push(PackedSecureField::from_array(last));
            }
        }

        Self { data, length }
    }
}

impl FromIterator<PackedSecureField> for SecureFieldVec {
    fn from_iter<I: IntoIterator<Item = PackedSecureField>>(iter: I) -> Self {
        let data = (&mut iter.into_iter()).collect_vec();
        let length = data.len() * N_LANES;

        Self { data, length }
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::BaseFieldVec;
    use crate::core::backend::simd::column::SecureFieldVec;
    use crate::core::backend::Column;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;

    #[test]
    fn base_field_vec_from_iter_works() {
        let values: [BaseField; 30] = array::from_fn(BaseField::from);

        let res = values.into_iter().collect::<BaseFieldVec>();

        assert_eq!(res.to_cpu(), values);
    }

    #[test]
    fn secure_field_vec_from_iter_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let values: [SecureField; 30] = rng.gen();

        let res = values.into_iter().collect::<SecureFieldVec>();

        assert_eq!(res.to_cpu(), values);
    }
}
