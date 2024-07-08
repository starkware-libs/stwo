use std::ops::{Add, Mul, Sub};

use itertools::Itertools;
use num_traits::Zero;

use crate::core::backend::simd::column::SecureColumn;
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::prefix_sum::inclusive_prefix_sum;
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumnByCoords;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::shifted_secure_combination;
use crate::core::ColumnVec;

/// Interaction elements for the logup protocol.
#[derive(Copy, Clone, Debug)]
pub struct LookupElements {
    pub z: SecureField,
    pub alpha: SecureField,
}
impl LookupElements {
    pub fn draw(channel: &mut Blake2sChannel) -> Self {
        let [z, alpha] = channel.draw_felts(2).try_into().unwrap();
        Self { z, alpha }
    }
    pub fn combine<F: Copy, EF>(&self, values: &[F]) -> EF
    where
        EF: Copy
            + Zero
            + Mul<EF, Output = EF>
            + Add<F, Output = EF>
            + Sub<EF, Output = EF>
            + From<SecureField>,
    {
        shifted_secure_combination(values, EF::from(self.z), EF::from(self.alpha))
    }
}

// SIMD backend generator for logup interaction trace.
pub struct LogupTraceGenerator {
    log_size: u32,
    /// Current allocated interaction columns.
    trace: Vec<SecureColumnByCoords<SimdBackend>>,
    /// Denominator expressions (z + sum_i alpha^i * x_i) being generated for the current lookup.
    denom: SecureColumn,
    /// Preallocated buffer for the Inverses of the denominators.
    denom_inv: SecureColumn,
}
impl LogupTraceGenerator {
    pub fn new(log_size: u32) -> Self {
        let trace = vec![];
        let denom = SecureColumn::zeros(1 << log_size);
        let denom_inv = SecureColumn::zeros(1 << log_size);
        Self {
            log_size,
            trace,
            denom,
            denom_inv,
        }
    }

    /// Allocate a new lookup column.
    pub fn new_col(&mut self) -> LogupColGenerator<'_> {
        let log_size = self.log_size;
        LogupColGenerator {
            gen: self,
            numerator: SecureColumnByCoords::<SimdBackend>::zeros(1 << log_size),
        }
    }

    /// Finalize the trace. Returns the trace and the claimed sum of the last column.
    pub fn finalize(
        mut self,
    ) -> (
        ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        SecureField,
    ) {
        // Prefix sum the last column.
        let last_col_coords = self.trace.pop().unwrap().columns;
        let coord_prefix_sum = last_col_coords.map(inclusive_prefix_sum);
        self.trace.push(SecureColumnByCoords {
            columns: coord_prefix_sum,
        });
        let claimed_sum = self.trace.last().unwrap().at(1);

        let trace = self
            .trace
            .into_iter()
            .flat_map(|eval| {
                eval.columns.map(|c| {
                    CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(
                        CanonicCoset::new(self.log_size).circle_domain(),
                        c,
                    )
                })
            })
            .collect_vec();
        (trace, claimed_sum)
    }
}

/// Trace generator for a single lookup column.
pub struct LogupColGenerator<'a> {
    gen: &'a mut LogupTraceGenerator,
    /// Numerator expressions (i.e. multiplicities) being generated for the current lookup.
    numerator: SecureColumnByCoords<SimdBackend>,
}
impl<'a> LogupColGenerator<'a> {
    /// Write a fraction to the column at a row.
    pub fn write_frac(
        &mut self,
        vec_row: usize,
        numerator: PackedSecureField,
        denom: PackedSecureField,
    ) {
        unsafe {
            self.numerator.set_packed(vec_row, numerator);
            *self.gen.denom.data.get_unchecked_mut(vec_row) = denom;
        }
    }

    /// Finalizes generating the column.
    pub fn finalize_col(mut self) {
        FieldExpOps::batch_inverse(&self.gen.denom.data, &mut self.gen.denom_inv.data);

        #[allow(clippy::needless_range_loop)]
        for vec_row in 0..(1 << (self.gen.log_size - LOG_N_LANES)) {
            unsafe {
                let value = self.numerator.packed_at(vec_row)
                    * *self.gen.denom_inv.data.get_unchecked(vec_row);
                let prev_value = self
                    .gen
                    .trace
                    .last()
                    .map(|col| col.packed_at(vec_row))
                    .unwrap_or_else(PackedSecureField::zero);
                self.numerator.set_packed(vec_row, value + prev_value)
            };
        }

        self.gen.trace.push(self.numerator)
    }
}
