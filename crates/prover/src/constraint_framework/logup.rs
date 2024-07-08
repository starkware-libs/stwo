use itertools::Itertools;
use num_traits::Zero;
use tracing::{span, Level};

use crate::core::backend::simd::column::SecureFieldVec;
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Backend, Column};
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::bit_reverse_index;
use crate::core::ColumnVec;

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
}

// SIMD backend generator.
pub struct LogupTraceGenerator {
    log_size: u32,
    trace: Vec<SecureColumn<SimdBackend>>,
    denom: SecureFieldVec,
    denom_inv: SecureFieldVec,
}
impl LogupTraceGenerator {
    pub fn new(log_size: u32) -> Self {
        let trace = vec![];
        let denom = SecureFieldVec::zeros(1 << log_size);
        let denom_inv = SecureFieldVec::zeros(1 << log_size);
        Self {
            log_size,
            trace,
            denom,
            denom_inv,
        }
    }

    pub fn new_col(&mut self) -> LogupColGenerator<'_> {
        let log_size = self.log_size;
        LogupColGenerator {
            gen: self,
            numerator: SecureColumn::<SimdBackend>::zeros(1 << log_size),
        }
    }

    pub fn finalize(
        mut self,
    ) -> (
        ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        SecureField,
    ) {
        let claimed_xor_sum = eval_order_prefix_sum(self.trace.last_mut().unwrap(), self.log_size);

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
        (trace, claimed_xor_sum)
    }
}

pub struct LogupColGenerator<'a> {
    gen: &'a mut LogupTraceGenerator,
    numerator: SecureColumn<SimdBackend>,
}
impl<'a> LogupColGenerator<'a> {
    pub fn write_frac(&mut self, vec_row: usize, p: PackedSecureField, q: PackedSecureField) {
        unsafe {
            self.numerator.set_packed(vec_row, p);
            *self.gen.denom.data.get_unchecked_mut(vec_row) = q;
        }
    }

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

// TODO(spapini): Consider adding optional Ops.
pub fn eval_order_prefix_sum<B: Backend>(col: &mut SecureColumn<B>, log_size: u32) -> SecureField {
    let _span = span!(Level::INFO, "Prefix sum").entered();

    let mut cur = SecureField::zero();
    for i in 0..(1 << log_size) {
        let index = if i & 1 == 0 {
            i / 2
        } else {
            (1 << (log_size - 1)) + ((1 << log_size) - 1 - i) / 2
        };
        let index = bit_reverse_index(index, log_size);
        cur += col.at(index);
        col.set(index, cur);
    }
    cur
}
