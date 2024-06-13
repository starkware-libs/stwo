use std::sync::Arc;

use cudarc::driver::{CudaDevice, DeviceRepr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use super::column::BaseFieldCudaColumn;
use super::{GpuBackend, DEVICE};
use crate::core::backend::{Col, Column, ColumnOps};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{FieldExpOps, FieldOps};
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
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddle_tree: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        let mut values = eval.values;
        let values_size = values.len();
        let log_values_size = u32::BITS - (values_size as u32).leading_zeros() - 1;

        assert!(eval.domain.half_coset.is_doubling_of(twiddle_tree.root_coset));

        let mut layer_domain_size = (values_size as u32) >> 1;
        let config = LaunchConfig::for_num_elems(values_size as u32);
        let kernel = DEVICE.get_func("circle", "fft_circle_part").unwrap();
        unsafe {
            kernel.launch(
                config,
                (values.as_mut_slice(), twiddle_tree.itwiddles.as_slice(), values_size),
            )
        }
        .unwrap();
        DEVICE.synchronize().unwrap();

        let mut layer_domain_offset = 0;
        for i in 0..log_values_size {
            let config = LaunchConfig::for_num_elems(values_size as u32);
            let kernel = DEVICE.get_func("circle", "fft_line_part").unwrap();
            unsafe {
                kernel.launch(
                    config,
                    (values.as_mut_slice(), twiddle_tree.itwiddles.as_slice(), values_size, layer_domain_size, layer_domain_offset, i + 1),
                )
            }
            .unwrap();
            layer_domain_size >>= 1;
            layer_domain_offset += layer_domain_size;
            DEVICE.synchronize().unwrap();
        }

        // Divide all values by 2^log_size.
        let config = LaunchConfig::for_num_elems(values_size as u32);
        let kernel = DEVICE.get_func("circle", "rescale").unwrap();
        unsafe {
            kernel.launch(
                config,
                (values.as_mut_slice(), values_size, M31(2).pow(log_values_size as u128).inverse().0),
            )
        }
        .unwrap();

        DEVICE.synchronize().unwrap();
        CirclePoly::new(values)
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
        let root_coset = coset.clone();
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
            root_coset,
            twiddles, 
            itwiddles
        }
    }
}

pub fn load_circle(device: &Arc<CudaDevice>) {
    let ptx_src = include_str!("circle.cu");
    let ptx = compile_ptx(ptx_src).unwrap();
    device.load_ptx(ptx, "circle", &["sort_values", "precompute_twiddles", "put_one", "fft_circle_part", "fft_line_part", "rescale"]).unwrap();
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
        let log_size = 20;
        let coset = CanonicCoset::new(log_size);
        let size: usize = 1 << log_size;
        let column_data = (0..size as u32).map(|x| M31(x)).collect_vec();
        let cpu_values = column_data.clone();
        let expected_result = CpuBackend::new_canonical_ordered(coset.clone(), cpu_values);

        let column = BaseFieldCudaColumn::from_vec(column_data);
        let result = GpuBackend::new_canonical_ordered(coset, column);

        assert_eq!(result.values.to_cpu(), expected_result.values);
        assert_eq!(result.domain.iter().collect_vec(), expected_result.domain.iter().collect_vec());
    }

    #[test]
    fn test_precompute_twiddles() {
        let log_size = 7;

        let half_coset = CanonicCoset::new(log_size).half_coset();
        let expected_result = CpuBackend::precompute_twiddles(half_coset.clone());
        let twiddles = GpuBackend::precompute_twiddles(half_coset);
        
        assert_eq!(twiddles.twiddles.to_cpu(), expected_result.twiddles);
        assert_eq!(twiddles.itwiddles.to_cpu(), expected_result.itwiddles);
        assert_eq!(twiddles.root_coset.iter().collect_vec(), expected_result.root_coset.iter().collect_vec());
    }


    #[test]
    fn test_interpolate() {
        let log_size = 7;
        let size = 1 << log_size;

        let cpu_values = (1..(size+1) as u32).map(|x| M31(x)).collect_vec();
        let gpu_values = BaseFieldCudaColumn::from_vec(cpu_values.clone());

        let coset = CanonicCoset::new(log_size);
        let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
        let gpu_evaluations = GpuBackend::new_canonical_ordered(coset, gpu_values);

        let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
        let gpu_twiddles = GpuBackend::precompute_twiddles(coset.half_coset());

        let expected_result = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
        let result = GpuBackend::interpolate(gpu_evaluations, &gpu_twiddles);

        assert_eq!(result.coeffs.to_cpu(), expected_result.coeffs);
    }
}
