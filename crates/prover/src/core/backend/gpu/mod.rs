pub mod error;
pub mod m31;

use std::fmt::Debug;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};
// use error::Error;
use once_cell::sync::Lazy;

use self::m31::LoadPackedBaseField;
use super::{Backend, Col, ColumnOps};
use crate::core::air::accumulation::AccumulationOps;
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::{BaseField, M31, P};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;
use crate::core::fields::FieldOps;
use crate::core::fri::FriOps;
use crate::core::pcs::quotients::{ColumnSampleBatch, QuotientOps};
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps, SecureEvaluation,
};
use crate::core::poly::line::LineEvaluation;
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::BitReversedOrder;
const VECTOR_SIZE: usize = 16;

// TODO:: cleanup unwraps with error handling?
// (We can replace lazy statics with unsafe global references)
static DEVICE: Lazy<Arc<CudaDevice>> = Lazy::new(|| CudaDevice::new(0).unwrap().load());
static MODULUS: Lazy<CudaSlice<u32>> =
    Lazy::new(|| DEVICE.htod_copy([P; VECTOR_SIZE].to_vec()).unwrap());

type Device = Arc<CudaDevice>;

trait Load {
    fn load(self) -> Self;
}

impl Load for Device {
    fn load(self) -> Self {
        LoadPackedBaseField::load(&self);
        self
    }
}

#[derive(Copy, Clone, Debug)]
struct GpuBackend;

impl Backend for GpuBackend {}

impl<T: Debug + Clone + Default> ColumnOps<T> for GpuBackend {
    type Column = Vec<T>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        todo!()
    }
}

impl FieldOps<M31> for GpuBackend {
    fn batch_inverse(_column: &Self::Column, _dst: &mut Self::Column) {
        todo!()
    }
}

impl FieldOps<SecureField> for GpuBackend {
    fn batch_inverse(_column: &Self::Column, _dst: &mut Self::Column) {
        todo!()
    }
}

impl AccumulationOps for GpuBackend {
    fn accumulate(_column: &mut SecureColumn<Self>, _other: &SecureColumn<Self>) {
        todo!()
    }
}

impl PolyOps for GpuBackend {
    type Twiddles = Vec<BaseField>;

    fn new_canonical_ordered(
        _coset: CanonicCoset,
        _values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        todo!()
    }

    fn interpolate(
        _eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        _itwiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        todo!()
    }

    fn eval_at_point(_poly: &CirclePoly<Self>, _point: CirclePoint<SecureField>) -> SecureField {
        todo!()
    }

    fn extend(_poly: &CirclePoly<Self>, _log_size: u32) -> CirclePoly<Self> {
        todo!()
    }

    fn evaluate(
        _poly: &CirclePoly<Self>,
        _domain: CircleDomain,
        _twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        todo!()
    }

    fn precompute_twiddles(_coset: Coset) -> TwiddleTree<Self> {
        todo!()
    }
}

impl FriOps for GpuBackend {
    fn fold_line(
        _eval: &LineEvaluation<Self>,
        _alpha: SecureField,
        _twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        todo!()
    }

    fn fold_circle_into_line(
        _dst: &mut LineEvaluation<GpuBackend>,
        _src: &SecureEvaluation<Self>,
        _alpha: SecureField,
        _twiddles: &TwiddleTree<Self>,
    ) {
        todo!()
    }

    fn decompose(_eval: &SecureEvaluation<Self>) -> (SecureEvaluation<Self>, SecureField) {
        todo!()
    }
}

impl QuotientOps for GpuBackend {
    fn accumulate_quotients(
        _domain: CircleDomain,
        _columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        _random_coeff: SecureField,
        _sample_batches: &[ColumnSampleBatch],
    ) -> SecureEvaluation<Self> {
        todo!()
    }
}
