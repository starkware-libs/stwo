#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::SimdBackend;
use crate::core::air::accumulation::AccumulationOps;
use crate::core::fields::secure_column::SecureColumn;

impl AccumulationOps for SimdBackend {
    fn accumulate(column: &mut SecureColumn<Self>, other: &SecureColumn<Self>) {
        #[cfg(not(feature = "parallel"))]
        let iter = 0..column.packed_len();
        #[cfg(feature = "parallel")]
        let iter = (0..column.packed_len()).into_par_iter();

        iter.for_each(|i| {
            let column_ptr = column as *const SecureColumn<Self>;
            let column_ptr = column_ptr as *mut SecureColumn<Self>;
            let res_coeff = unsafe { column.packed_at(i) + other.packed_at(i) };
            unsafe { (*column_ptr).set_packed(i, res_coeff) };
        });
    }
}
