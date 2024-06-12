use std::sync::Arc;

use cudarc::driver::{CudaDevice, DeviceRepr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use super::column::BaseFieldCudaColumn;
use super::{GpuBackend, DEVICE};
use crate::core::backend::{Col, Column, ColumnOps};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldOps;
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::BitReversedOrder;

unsafe impl DeviceRepr for CirclePoint<BaseField> {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut _
    }
}

impl PolyOps for GpuBackend {
    // TODO: This type may need to be changed
    type Twiddles = BaseFieldCudaColumn;

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

    fn precompute_twiddles(mut coset: Coset) -> TwiddleTree<Self> {
        // Compute twiddles
        let config = LaunchConfig::for_num_elems(coset.size() as u32);
        let kernel = DEVICE.get_func("circle", "precompute_twiddles").unwrap();

        let mut twiddles = BaseFieldCudaColumn::new(unsafe { DEVICE.alloc(coset.size()).unwrap() });
        let mut itwiddles = BaseFieldCudaColumn::new(unsafe { DEVICE.alloc(coset.size()).unwrap() });

        let mut current_level_offset = 0;
        
        for _ in 0..coset.log_size() {
            // Compute each level of the twiddle tree.
            unsafe {
                kernel.clone().launch(
                    config,
                    (twiddles.as_mut_slice(),
                     coset.initial(),
                     coset.step_size.to_point(),
                     current_level_offset,
                     coset.size(),
                     coset.log_size()),
                )
            }
            .unwrap();
            coset = coset.double();
            current_level_offset += coset.size();
        }
        DEVICE.synchronize().unwrap();

        // Put a one in the last position.
        let kernel = DEVICE.get_func("circle", "put_one").unwrap();
        let config = LaunchConfig::for_num_elems(1);
        unsafe {
            kernel.clone().launch(config, (twiddles.as_mut_slice(), current_level_offset))
        }.unwrap();

        <Self as FieldOps<BaseField>>::batch_inverse(&twiddles, &mut itwiddles);

        TwiddleTree {
            root_coset: coset,
            twiddles, 
            itwiddles
        }
    }
}

pub fn load_circle(device: &Arc<CudaDevice>) {
    let ptx_src = include_str!("circle.cu");
    let ptx = compile_ptx(ptx_src).unwrap();
    device.load_ptx(ptx, "circle", &["sort_values", "precompute_twiddles", "put_one"]).unwrap();
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
        let log_size = 3;
        let coset = CanonicCoset::new(log_size);
        let size: usize = 1 << log_size;
        let column_data = (0..size as u32).map(|x| M31(x)).collect_vec();
        let cpu_values = column_data.clone();
        let expected_result = CpuBackend::new_canonical_ordered(coset.clone(), cpu_values);

        let column = BaseFieldCudaColumn::from_vec(column_data);
        let result = GpuBackend::new_canonical_ordered(coset, column);

        assert_eq!(result.values.to_cpu(), expected_result.values);
    }

    #[test]
    fn test_precompute_twiddles() {
        let log_size = 20;

        let coset = CanonicCoset::new(log_size).half_coset();
        let expected_result = CpuBackend::precompute_twiddles(coset.clone());
        let twiddles = GpuBackend::precompute_twiddles(coset);
        
        assert_eq!(twiddles.twiddles.to_cpu(), expected_result.twiddles);
        assert_eq!(twiddles.itwiddles.to_cpu(), expected_result.itwiddles);
    }
}
