use super::CPUBackend;
use crate::core::air::evaluation::SecureColumn;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::mle::{ColumnOpsV2, ColumnV2};

mod gkr;
mod grand_product;
mod logup;
pub mod mle;

impl ColumnOpsV2<BaseField> for CPUBackend {
    type Column = Vec<BaseField>;
}

impl ColumnV2<BaseField> for Vec<BaseField> {
    fn to_vec(&self) -> Vec<BaseField> {
        self.clone()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl ColumnOpsV2<SecureField> for CPUBackend {
    type Column = SecureColumn<Self>;
}

impl ColumnV2<SecureField> for SecureColumn<CPUBackend> {
    fn to_vec(&self) -> Vec<SecureField> {
        let mut res = Vec::new();

        for i in 0..self.len() {
            res.push(self.at(i))
        }

        res
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl Default for SecureColumn<CPUBackend> {
    fn default() -> Self {
        Self {
            cols: Default::default(),
        }
    }
}

impl FromIterator<SecureField> for SecureColumn<CPUBackend> {
    fn from_iter<T: IntoIterator<Item = SecureField>>(iter: T) -> Self {
        let mut res = Self::default();

        for v in iter {
            res.push(v);

            // let [v0, v1, v2, v3] = v.to_m31_array();
            // res.cols[0].push(v0);
            // res.cols[1].push(v1);
            // res.cols[2].push(v2);
            // res.cols[3].push(v3);
        }

        res
    }
}
