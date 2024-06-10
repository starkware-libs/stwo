use std::sync::Arc;

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use super::column::BaseFieldCudaColumn;
use super::{GpuBackend, DEVICE};
use crate::core::backend::{Col, Column, ColumnOps};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::BitReversedOrder;

impl PolyOps for GpuBackend {
    // TODO: This type may need to be changed
    type Twiddles = Vec<BaseField>;

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let domain = coset.circle_domain();
        let size = values.len();
        let config = LaunchConfig::for_num_elems(size as u32);
        let kernel = DEVICE.get_func("circle", "sort_values").unwrap();
        let mut sorted_values = BaseFieldCudaColumn::new(unsafe { DEVICE.alloc(size).unwrap() });
        unsafe {
            kernel.launch(
                config,
                (values.as_slice(), sorted_values.as_mut_slice(), size),
            )
        }
        .unwrap();
        DEVICE.synchronize().unwrap();

        <GpuBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut sorted_values);
        CircleEvaluation::new(domain, sorted_values)
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

pub fn load_circle(device: &Arc<CudaDevice>) {
    let ptx_src = include_str!("circle.cu");
    let ptx = compile_ptx(ptx_src).unwrap();
    device.load_ptx(ptx, "circle", &["sort_values"]).unwrap();
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::core::backend::gpu::column::BaseFieldCudaColumn;
    use crate::core::backend::gpu::GpuBackend;
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::fields::m31::M31;
    use crate::core::poly::circle::{CanonicCoset, PolyOps};

    #[test]
    fn test_new_canonical_ordered() {
        let log_size = 12;
        let coset = CanonicCoset::new(log_size);
        let size: usize = 1 << log_size;
        let column_data = (0..size as u32).map(|x| M31(x)).collect_vec();
        let cpu_values = column_data.clone();
        let expected_result = CpuBackend::new_canonical_ordered(coset.clone(), cpu_values);

        let column = BaseFieldCudaColumn::from_vec(column_data);
        let result = GpuBackend::new_canonical_ordered(coset, column);

        assert_eq!(result.values.to_cpu(), expected_result.values);
    }
}
