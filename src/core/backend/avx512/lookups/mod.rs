use num_traits::Zero;

use super::{AVX512Backend, BaseFieldVec, PackedBaseField};
use crate::core::air::evaluation::SecureColumn;
use crate::core::backend::{avx512, Column};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::mle::{ColumnOpsV2, ColumnV2};

mod gkr;
mod grand_product;
mod logup;
mod mle;

impl ColumnOpsV2<BaseField> for AVX512Backend {
    type Column = BaseFieldVec;
}

impl ColumnV2<BaseField> for BaseFieldVec {
    fn to_vec(&self) -> Vec<BaseField> {
        Column::to_vec(self)
    }

    fn len(&self) -> usize {
        Column::len(self)
    }
}

impl ColumnOpsV2<SecureField> for AVX512Backend {
    type Column = SecureColumn<Self>;
}

impl ColumnV2<SecureField> for SecureColumn<AVX512Backend> {
    fn to_vec(&self) -> Vec<SecureField> {
        let c0 = self.cols[0].as_slice();
        let c1 = self.cols[1].as_slice();
        let c2 = self.cols[2].as_slice();
        let c3 = self.cols[3].as_slice();

        let mut res = Vec::new();

        for i in 0..self.len() {
            res.push(SecureField::from_m31_array([c0[i], c1[i], c2[i], c3[i]]))
        }

        res
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl FromIterator<SecureField> for SecureColumn<AVX512Backend> {
    fn from_iter<T: IntoIterator<Item = SecureField>>(iter: T) -> Self {
        let mut col0 = Vec::new();
        let mut col1 = Vec::new();
        let mut col2 = Vec::new();
        let mut col3 = Vec::new();

        let mut chunks = iter
            .into_iter()
            .array_chunks::<{ avx512::m31::K_BLOCK_SIZE }>();

        for chunk in &mut chunks {
            let mut v0 = [BaseField::zero(); avx512::m31::K_BLOCK_SIZE];
            let mut v1 = [BaseField::zero(); avx512::m31::K_BLOCK_SIZE];
            let mut v2 = [BaseField::zero(); avx512::m31::K_BLOCK_SIZE];
            let mut v3 = [BaseField::zero(); avx512::m31::K_BLOCK_SIZE];

            chunk.iter().enumerate().for_each(|(i, vi)| {
                let [c0, c1, c2, c3] = vi.to_m31_array();
                v0[i] = c0;
                v1[i] = c1;
                v2[i] = c2;
                v3[i] = c3;
            });

            col0.push(PackedBaseField::from_array(v0));
            col1.push(PackedBaseField::from_array(v1));
            col2.push(PackedBaseField::from_array(v2));
            col3.push(PackedBaseField::from_array(v3));
        }

        let mut length = col0.len() * avx512::m31::K_BLOCK_SIZE;

        if let Some(remainder) = chunks.into_remainder() {
            // TODO: Add `transpose` function/closure to DRY the code here and in the inner loop.
            let mut v0 = [BaseField::zero(); avx512::m31::K_BLOCK_SIZE];
            let mut v1 = [BaseField::zero(); avx512::m31::K_BLOCK_SIZE];
            let mut v2 = [BaseField::zero(); avx512::m31::K_BLOCK_SIZE];
            let mut v3 = [BaseField::zero(); avx512::m31::K_BLOCK_SIZE];

            length += remainder.len();

            remainder.into_iter().enumerate().for_each(|(i, vi)| {
                let [c0, c1, c2, c3] = vi.to_m31_array();
                v0[i] = c0;
                v1[i] = c1;
                v2[i] = c2;
                v3[i] = c3;
            });

            col0.push(PackedBaseField::from_array(v0));
            col1.push(PackedBaseField::from_array(v1));
            col2.push(PackedBaseField::from_array(v2));
            col3.push(PackedBaseField::from_array(v3));
        }

        println!("length: {length}");

        SecureColumn {
            cols: [
                BaseFieldVec { data: col0, length },
                BaseFieldVec { data: col1, length },
                BaseFieldVec { data: col2, length },
                BaseFieldVec { data: col3, length },
            ],
        }
    }
}
