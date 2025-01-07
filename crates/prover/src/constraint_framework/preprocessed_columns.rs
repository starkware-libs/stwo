use std::any::Any;
use std::fmt::Debug;
use std::hash::Hash;
use std::rc::Rc;
use std::simd::Simd;

use num_traits::{One, Zero};

use crate::core::backend::simd::m31::{PackedM31, N_LANES};
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column};
use crate::core::fields::m31::{BaseField, M31};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::{bit_reverse_index, coset_index_to_circle_domain_index};

const SIMD_ENUMERATION_0: PackedM31 = unsafe {
    PackedM31::from_simd_unchecked(Simd::from_array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    ]))
};

// TODO(ilya): Where should this enum be placed?
pub trait PreprocessedColumn: Debug + Any {
    fn name(&self) -> &'static str;
    fn to_string(&self) -> String;
    fn log_size(&self) -> u32;
    fn packed_at(&self, vec_row: usize) -> PackedM31;
    fn gen_preprocessed_column_simd(
        &self,
    ) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>;
}

impl PartialEq for dyn PreprocessedColumn {
    fn eq(&self, other: &Self) -> bool {
        self.to_string() == other.to_string()
    }
}

impl Eq for dyn PreprocessedColumn {}

impl Hash for dyn PreprocessedColumn {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.to_string().hash(state);
    }
}

/// A column with `1` at the first position, and `0` elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IsFirst {
    pub log_size: u32,
}
impl IsFirst {
    pub const fn new(log_size: u32) -> Self {
        Self { log_size }
    }
}
impl PreprocessedColumn for IsFirst {
    fn name(&self) -> &'static str {
        "preprocessed_is_first"
    }

    fn to_string(&self) -> String {
        format!("IsFirst(log_size: {})", self.log_size)
    }

    fn log_size(&self) -> u32 {
        self.log_size
    }
    fn packed_at(&self, vec_row: usize) -> PackedM31 {
        assert!(vec_row < (1 << self.log_size) / N_LANES);
        if vec_row == 0 {
            unsafe {
                PackedM31::from_simd_unchecked(Simd::from_array(std::array::from_fn(|i| {
                    if i == 0 {
                        1
                    } else {
                        0
                    }
                })))
            }
        } else {
            PackedM31::zero()
        }
    }

    fn gen_preprocessed_column_simd(
        &self,
    ) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
        let mut col = Col::<SimdBackend, BaseField>::zeros(1 << self.log_size);
        col.set(0, BaseField::one());
        CircleEvaluation::new(CanonicCoset::new(self.log_size).circle_domain(), col)
    }
}

/// A column with the numbers [0..2^log_size-1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Seq {
    pub log_size: u32,
}
impl Seq {
    pub const fn new(log_size: u32) -> Self {
        Self { log_size }
    }
}
impl PreprocessedColumn for Seq {
    fn name(&self) -> &'static str {
        "preprocessed_seq"
    }

    fn to_string(&self) -> String {
        format!("Seq(log_size: {})", self.log_size)
    }

    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn packed_at(&self, vec_row: usize) -> PackedM31 {
        assert!(vec_row < (1 << self.log_size) / N_LANES);
        PackedM31::broadcast(M31::from(vec_row * N_LANES)) + SIMD_ENUMERATION_0
    }

    fn gen_preprocessed_column_simd(
        &self,
    ) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
        let col = Col::<SimdBackend, BaseField>::from_iter(
            (0..(1 << self.log_size)).map(BaseField::from),
        );
        CircleEvaluation::new(CanonicCoset::new(self.log_size).circle_domain(), col)
    }
}

// TODO(Gali): Add documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct XorTable {
    pub xor_var_1: u32,
    pub xor_var_2: u32,
    pub xor_var_3: usize,
}
impl XorTable {
    pub const fn new(xor_var_1: u32, xor_var_2: u32, xor_var_3: usize) -> Self {
        Self {
            xor_var_1,
            xor_var_2,
            xor_var_3,
        }
    }
}
impl PreprocessedColumn for XorTable {
    fn name(&self) -> &'static str {
        "preprocessed_xor_table"
    }

    fn to_string(&self) -> String {
        format!(
            "XorTable(xor_var_1: {}, xor_var_2: {}, xor_var_3: {})",
            self.xor_var_1, self.xor_var_2, self.xor_var_3
        )
    }

    fn log_size(&self) -> u32 {
        2 * (self.xor_var_1 - self.xor_var_2)
    }

    fn packed_at(&self, _vec_row: usize) -> PackedM31 {
        unimplemented!("XorTable::packed_at")
    }

    fn gen_preprocessed_column_simd(
        &self,
    ) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
        unimplemented!("XorTable::gen_preprocessed_column")
    }
}

// TODO(Gali): Add documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Plonk {
    pub plonk_var: usize,
}
impl Plonk {
    pub const fn new(plonk_var: usize) -> Self {
        Self { plonk_var }
    }
}
impl PreprocessedColumn for Plonk {
    fn name(&self) -> &'static str {
        "preprocessed_plonk"
    }

    fn to_string(&self) -> String {
        format!("Plonk(plonk_var: {})", self.plonk_var)
    }

    fn log_size(&self) -> u32 {
        unimplemented!("Plonk::log_size")
    }

    fn packed_at(&self, _vec_row: usize) -> PackedM31 {
        unimplemented!("Plonk::packed_at")
    }

    fn gen_preprocessed_column_simd(
        &self,
    ) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
        unimplemented!("Plonk::gen_preprocessed_column")
    }
}

/// A column with `1` at every `2^log_step` positions, `0` elsewhere, shifted by offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IsStepWithOffset {
    log_size: u32,
    log_step: u32,
    offset: usize,
}
impl IsStepWithOffset {
    pub const fn new(log_size: u32, log_step: u32, offset: usize) -> Self {
        Self {
            log_size,
            log_step,
            offset,
        }
    }
}
impl PreprocessedColumn for IsStepWithOffset {
    fn name(&self) -> &'static str {
        "preprocessed_is_step_with_offset"
    }

    fn to_string(&self) -> String {
        format!(
            "IsStepWithOffset(log_size: {}, log_step: {}, offset: {})",
            self.log_size, self.log_step, self.offset
        )
    }

    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn packed_at(&self, _vec_row: usize) -> PackedM31 {
        unimplemented!("IsStepWithOffset::packed_at")
    }

    // TODO(andrew): Consider optimizing. Is a quotients of two coset_vanishing (use succinct rep
    // for verifier).
    fn gen_preprocessed_column_simd(
        &self,
    ) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
        let mut col = Col::<SimdBackend, BaseField>::zeros(1 << self.log_size);

        let size = 1 << self.log_size;
        let step = 1 << self.log_step;
        let step_offset = self.offset % step;

        for i in (step_offset..size).step_by(step) {
            let circle_domain_index = coset_index_to_circle_domain_index(i, self.log_size);
            let circle_domain_index_bit_rev = bit_reverse_index(circle_domain_index, self.log_size);
            col.set(circle_domain_index_bit_rev, BaseField::one());
        }

        CircleEvaluation::new(CanonicCoset::new(self.log_size).circle_domain(), col)
    }
}

pub fn gen_preprocessed_columns<'a>(
    columns: impl Iterator<Item = &'a Rc<dyn PreprocessedColumn>>,
) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    columns
        .map(|col| col.gen_preprocessed_column_simd())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{IsFirst, PreprocessedColumn, Seq};
    use crate::core::backend::simd::m31::N_LANES;
    use crate::core::backend::Column;
    use crate::core::fields::m31::{BaseField, M31};
    const LOG_SIZE: u32 = 8;

    #[test]
    fn test_gen_seq() {
        let seq = Seq::new(LOG_SIZE).gen_preprocessed_column_simd();

        for i in 0..(1 << LOG_SIZE) {
            assert_eq!(seq.at(i), BaseField::from_u32_unchecked(i as u32));
        }
    }

    // TODO(Gali): Add packed_at tests for xor_table and plonk.
    #[test]
    fn test_packed_at_is_first() {
        let is_first = IsFirst::new(LOG_SIZE);
        let expected_is_first = is_first.gen_preprocessed_column_simd().to_cpu();

        for i in 0..(1 << LOG_SIZE) / N_LANES {
            assert_eq!(
                is_first.packed_at(i).to_array(),
                expected_is_first[i * N_LANES..(i + 1) * N_LANES]
            );
        }
    }

    #[test]
    fn test_packed_at_seq() {
        let seq = Seq::new(LOG_SIZE);
        let expected_seq: [_; 1 << LOG_SIZE] = std::array::from_fn(|i| M31::from(i as u32));

        let packed_seq = std::array::from_fn::<_, { (1 << LOG_SIZE) / N_LANES }, _>(|i| {
            seq.packed_at(i).to_array()
        })
        .concat();

        assert_eq!(packed_seq, expected_seq);
    }
}
