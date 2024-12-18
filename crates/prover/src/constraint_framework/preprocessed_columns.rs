use std::any::Any;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use num_traits::One;

use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Backend, Col, Column, CpuBackend};
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::{bit_reverse_index, coset_index_to_circle_domain_index};

/// XorTable, etc will be implementation of this trait.
pub trait PreprocessedColumnOps: Debug + Any {
    fn get_type_id(&self) -> std::any::TypeId {
        self.type_id()
    }
    fn name(&self) -> &'static str;
    fn log_size(&self) -> u32;
    fn gen_preprocessed_column_cpu(
        &self,
    ) -> CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>;
    fn gen_preprocessed_column_simd(
        &self,
    ) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>;
    fn as_bytes(&self) -> Vec<u8>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IsFirst {
    pub log_size: u32,
}

impl PreprocessedColumnOps for IsFirst {
    fn name(&self) -> &'static str {
        "preprocessed.is_first"
    }
    fn log_size(&self) -> u32 {
        self.log_size
    }
    fn gen_preprocessed_column_cpu(
        &self,
    ) -> CircleEvaluation<CpuBackend, BaseField, BitReversedOrder> {
        gen_is_first(self.log_size)
    }
    fn gen_preprocessed_column_simd(
        &self,
    ) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
        gen_is_first(self.log_size)
    }
    fn as_bytes(&self) -> Vec<u8> {
        self.log_size.to_le_bytes().to_vec()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct XorTable {
    pub elem_bits: u32,
    pub expand_bits: u32,
    pub kind: usize,
}

impl PreprocessedColumnOps for XorTable {
    fn name(&self) -> &'static str {
        "preprocessed.xor_table"
    }
    fn log_size(&self) -> u32 {
        assert!(self.elem_bits >= self.expand_bits);
        2 * (self.elem_bits - self.expand_bits)
    }
    fn gen_preprocessed_column_cpu(
        &self,
    ) -> CircleEvaluation<CpuBackend, BaseField, BitReversedOrder> {
        unimplemented!("XorTable is not supported.")
    }
    fn gen_preprocessed_column_simd(
        &self,
    ) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
        unimplemented!("XorTable is not supported.")
    }
    fn as_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![];
        bytes.extend_from_slice(&self.elem_bits.to_le_bytes());
        bytes.extend_from_slice(&self.expand_bits.to_le_bytes());
        bytes.extend_from_slice(&self.kind.to_le_bytes());
        bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Plonk {
    pub kind: u32,
}

impl PreprocessedColumnOps for Plonk {
    fn name(&self) -> &'static str {
        "preprocessed.plonk"
    }
    fn log_size(&self) -> u32 {
        unimplemented!("Plonk is not supported.")
    }
    fn gen_preprocessed_column_cpu(
        &self,
    ) -> CircleEvaluation<CpuBackend, BaseField, BitReversedOrder> {
        unimplemented!("Plonk is not supported.")
    }
    fn gen_preprocessed_column_simd(
        &self,
    ) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
        unimplemented!("Plonk is not supported.")
    }
    fn as_bytes(&self) -> Vec<u8> {
        self.kind.to_le_bytes().to_vec()
    }
}

impl PartialEq for dyn PreprocessedColumnOps {
    fn eq(&self, other: &Self) -> bool {
        self.get_type_id() == other.get_type_id() && self.as_bytes() == other.as_bytes()
    }
}

impl Eq for dyn PreprocessedColumnOps {}

impl Hash for dyn PreprocessedColumnOps {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.get_type_id().hash(state);
        self.as_bytes().hash(state);
    }
}

/// Generates a column with a single one at the first position, and zeros elsewhere.
pub fn gen_is_first<B: Backend>(log_size: u32) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
    let mut col = Col::<B, BaseField>::zeros(1 << log_size);
    col.set(0, BaseField::one());
    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

/// Generates a column with `1` at every `2^log_step` positions, `0` elsewhere, shifted by offset.
// TODO(andrew): Consider optimizing. Is a quotients of two coset_vanishing (use succinct rep for
// verifier).
pub fn gen_is_step_with_offset<B: Backend>(
    log_size: u32,
    log_step: u32,
    offset: usize,
) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
    let mut col = Col::<B, BaseField>::zeros(1 << log_size);

    let size = 1 << log_size;
    let step = 1 << log_step;
    let step_offset = offset % step;

    for i in (step_offset..size).step_by(step) {
        let circle_domain_index = coset_index_to_circle_domain_index(i, log_size);
        let circle_domain_index_bit_rev = bit_reverse_index(circle_domain_index, log_size);
        col.set(circle_domain_index_bit_rev, BaseField::one());
    }

    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

pub fn gen_preprocessed_columns_simd<'a>(
    columns: impl Iterator<Item = Arc<dyn PreprocessedColumnOps>>,
) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    columns
        .map(|col| col.gen_preprocessed_column_simd())
        .collect()
}
