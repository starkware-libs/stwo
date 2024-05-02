use self::bit_reverse::bit_reverse_m31;
use self::column::{BaseFieldVec, SecureFieldVec};
use self::m31::PackedBaseField;
use self::qm31::PackedSecureField;
use super::ColumnOps;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{FieldExpOps, FieldOps};
use crate::core::utils::bit_reverse as cpu_bit_reverse;

pub mod bit_reverse;
pub mod blake2s;
pub mod circle;
pub mod cm31;
pub mod column;
pub mod fft;
pub mod fri;
pub mod m31;
pub mod qm31;
mod utils;

#[derive(Copy, Clone, Debug)]
pub struct SimdBackend;

impl ColumnOps<BaseField> for SimdBackend {
    type Column = BaseFieldVec;

    fn bit_reverse_column(column: &mut Self::Column) {
        // Fallback to cpu bit_reverse.
        if column.data.len().ilog2() < bit_reverse::MIN_LOG_SIZE {
            cpu_bit_reverse(column.as_mut());
            return;
        }

        bit_reverse_m31(&mut column.data);
    }
}

impl FieldOps<BaseField> for SimdBackend {
    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
        PackedBaseField::batch_inverse(&column.data, &mut dst.data);
    }
}

impl ColumnOps<SecureField> for SimdBackend {
    type Column = SecureFieldVec;

    fn bit_reverse_column(_column: &mut Self::Column) {
        todo!()
    }
}

impl FieldOps<SecureField> for SimdBackend {
    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
        PackedSecureField::batch_inverse(&column.data, &mut dst.data);
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::bit_reverse;
    use crate::core::backend::simd::column::BaseFieldVec;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::{Column, ColumnOps};
    use crate::core::fields::m31::BaseField;
    use crate::core::utils::bit_reverse as ground_truth_bit_reverse;

    #[test]
    fn bit_reverse_small_column_works() {
        const LOG_SIZE: u32 = bit_reverse::MIN_LOG_SIZE - 1;
        let column = (0..1 << LOG_SIZE).map(BaseField::from).collect_vec();
        let mut expected = column.clone();
        ground_truth_bit_reverse(&mut expected);

        let mut res = column.iter().copied().collect::<BaseFieldVec>();
        <SimdBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut res);

        assert_eq!(res.to_cpu(), expected);
    }

    #[test]
    fn bit_reverse_large_column_works() {
        const LOG_SIZE: u32 = bit_reverse::MIN_LOG_SIZE;
        let column = (0..1 << LOG_SIZE).map(BaseField::from).collect_vec();
        let mut expected = column.clone();
        ground_truth_bit_reverse(&mut expected);

        let mut res = column.iter().copied().collect::<BaseFieldVec>();
        <SimdBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut res);

        assert_eq!(res.to_cpu(), expected);
    }
}
