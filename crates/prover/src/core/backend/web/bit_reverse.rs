use super::WebBackend;
use crate::core::backend::cpu::bit_reverse as cpu_bit_reverse;
use crate::core::backend::simd::bit_reverse::bit_reverse_m31;
use crate::core::backend::simd::column::{BaseColumn, SecureColumn};
use crate::core::backend::ColumnOps;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;

const VEC_BITS: u32 = 4;

const W_BITS: u32 = 3;

pub const MIN_LOG_SIZE: u32 = 2 * W_BITS + VEC_BITS;

impl ColumnOps<BaseField> for WebBackend {
    type Column = BaseColumn;

    fn bit_reverse_column(column: &mut Self::Column) {
        // Fallback to cpu bit_reverse.
        if column.data.len().ilog2() < MIN_LOG_SIZE {
            cpu_bit_reverse(column.as_mut_slice());
            return;
        }

        bit_reverse_m31(&mut column.data);
    }
}

impl ColumnOps<SecureField> for WebBackend {
    type Column = SecureColumn;

    fn bit_reverse_column(_column: &mut SecureColumn) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::MIN_LOG_SIZE;
    use crate::core::backend::cpu::bit_reverse as cpu_bit_reverse;
    use crate::core::backend::simd::bit_reverse::{bit_reverse16, bit_reverse_m31};
    use crate::core::backend::simd::column::BaseColumn;
    use crate::core::backend::simd::m31::{PackedM31, N_LANES};
    use crate::core::backend::web::WebBackend;
    use crate::core::backend::{Column, ColumnOps};
    use crate::core::fields::m31::BaseField;

    #[test]
    fn test_bit_reverse16() {
        let values: BaseColumn = (0..N_LANES * 16).map(BaseField::from).collect();
        let mut expected = values.to_cpu();
        cpu_bit_reverse(&mut expected);

        let res = bit_reverse16(values.data.try_into().unwrap());

        assert_eq!(res.map(PackedM31::to_array).as_flattened(), expected);
    }

    #[test]
    fn bit_reverse_m31_works() {
        const SIZE: usize = 1 << 15;
        let data: Vec<_> = (0..SIZE).map(BaseField::from).collect();
        let mut expected = data.clone();
        cpu_bit_reverse(&mut expected);

        let mut res: BaseColumn = data.into_iter().collect();
        bit_reverse_m31(&mut res.data[..]);

        assert_eq!(res.to_cpu(), expected);
    }

    #[test]
    fn bit_reverse_small_column_works() {
        const LOG_SIZE: u32 = MIN_LOG_SIZE - 1;
        let column = (0..1 << LOG_SIZE).map(BaseField::from).collect_vec();
        let mut expected = column.clone();
        cpu_bit_reverse(&mut expected);

        let mut res = column.iter().copied().collect::<BaseColumn>();
        <WebBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut res);

        assert_eq!(res.to_cpu(), expected);
    }

    #[test]
    fn bit_reverse_large_column_works() {
        const LOG_SIZE: u32 = MIN_LOG_SIZE;
        let column = (0..1 << LOG_SIZE).map(BaseField::from).collect_vec();
        let mut expected = column.clone();
        cpu_bit_reverse(&mut expected);

        let mut res = column.iter().copied().collect::<BaseColumn>();
        <WebBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut res);

        assert_eq!(res.to_cpu(), expected);
    }
}
