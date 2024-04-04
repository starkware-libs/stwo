use num_traits::Zero;

use crate::core::fields::m31::BaseField;

pub fn assert_constraints_on_row(row: &[BaseField]) {
    for i in 2..row.len() {
        assert_eq!(
            (row[i] - (row[i - 1] * row[i - 1] + row[i - 2] * row[i - 2])),
            BaseField::zero()
        );
    }
}
