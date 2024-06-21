use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
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

unsafe impl DeviceRepr for CirclePoint<SecureField> {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut _
    }
}

impl PolyOps for GpuBackend {
    type Twiddles = BaseFieldCudaColumn;

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let domain = coset.circle_domain();
        let size = values.len();
        let config = Self::launch_config_for_num_elems(size as u32, 256, 0);
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
        assert!(eval
            .domain
            .half_coset
            .is_doubling_of(twiddle_tree.root_coset));

        Self::ifft_circle_part(&mut values, twiddle_tree);
        DEVICE.synchronize().unwrap();

        Self::ifft_line_part(&mut values, twiddle_tree);
        DEVICE.synchronize().unwrap();

        let exponent = u32::BITS - (values.len() as u32).leading_zeros() - 1;
        Self::rescale_by_power_of_two(&mut values, exponent as u128);
        DEVICE.synchronize().unwrap();

        CirclePoly::new(values)
    }

    fn eval_at_point(poly: &CirclePoly<Self>, point: CirclePoint<SecureField>) -> SecureField {
        let mut mappings = vec![point.y, point.x];
        let mut x = point.x;
        for _ in 2..poly.log_size() {
            x = CirclePoint::double_x(x);
            mappings.push(x);
        }
        mappings.reverse();
        let mappings_size = mappings.len();

        let temp_memory_size = {
            let mut size = poly.coeffs.len();
            let mut result = 0;
            while size > 1 {
                size = (size + 511) / 512;
                result += size;
            }
            result
        };

        unsafe {
            // Gpu slices
            let mut device_result: CudaSlice<SecureField> = DEVICE.alloc(1).unwrap();
            let device_mappings: CudaSlice<SecureField> = DEVICE.htod_copy(mappings).unwrap();
            let mut temp: CudaSlice<SecureField> = DEVICE.alloc(temp_memory_size).unwrap();

            let coeffs_length = poly.coeffs.len() as u32;
            let config =
                Self::launch_config_for_num_elems(coeffs_length >> 1, 256, 512 * 4 + 512 * 8);
            let mut num_blocks = config.grid_dim.0;
            let mut output_offset = temp_memory_size - num_blocks as usize;

            // First pass: starts from the coefficients of the polynomial as the leaves of the tree
            // and each cuda block computes an inner node in the folding tree. The leaves are
            // elements of `BaseField`, but all inner nodes are elements of
            // `SecureField`.
            let kernel = DEVICE
                .get_func("circle", "eval_at_point_first_pass")
                .unwrap();
            kernel
                .clone()
                .launch(
                    config,
                    (
                        poly.coeffs.as_slice(),
                        &mut temp,
                        &device_mappings,
                        poly.coeffs.len(),
                        mappings_size,
                        point,
                        output_offset,
                    ),
                )
                .unwrap();

            // Second pass. Constructs the upper levels of the tree until it reaches the root.
            // The difference with the first pass is that leaves in this part are elements of
            // `SecureField`.
            let mut mappings_offset = mappings_size - 1;
            let kernel = DEVICE
                .get_func("circle", "eval_at_point_second_pass")
                .unwrap();
            let mut level_offset = output_offset;
            while num_blocks > 1 {
                mappings_offset -= 9;
                let config = Self::launch_config_for_num_elems(num_blocks >> 1, 256, 512 * 4 * 4);
                output_offset = level_offset - config.grid_dim.0 as usize;
                kernel
                    .clone()
                    .launch(
                        config,
                        (
                            &mut temp,
                            &device_mappings,
                            num_blocks,
                            mappings_offset,
                            point,
                            level_offset,
                            output_offset,
                        ),
                    )
                    .unwrap();
                num_blocks = config.grid_dim.0;
                level_offset = output_offset;
            }
            assert_eq!(output_offset, 0);

            // Copy the root of the folding tree to `device_result`. This is done
            // to avoid copying the entire `temp` slice, which is slow.
            // TODO: find a better way of doing this without launching a kernel.
            let kernel = DEVICE.get_func("circle", "get_result_from_temp").unwrap();
            kernel
                .launch(
                    Self::launch_config_for_num_elems(1, 1, 0),
                    (&temp, &mut device_result),
                )
                .unwrap();
            DEVICE.synchronize().unwrap();
            let result = DEVICE.dtoh_sync_copy(&device_result).unwrap();
            assert_eq!(result.len(), 1);

            result[0]
        }
    }

    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self> {
        let new_size = 1 << log_size;
        assert!(
            new_size >= poly.coeffs.len(),
            "New size must be larger than the old size"
        );

        let mut new_slice: CudaSlice<M31> = DEVICE.alloc_zeros(new_size).unwrap();

        DEVICE
            .dtod_copy(poly.coeffs.as_slice(), &mut new_slice)
            .unwrap();

        CirclePoly::new(BaseFieldCudaColumn::new(new_slice))
    }

    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
        twiddle_tree: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let mut values = poly.extend(domain.log_size()).coeffs;

        assert!(domain.half_coset.is_doubling_of(twiddle_tree.root_coset));

        Self::rfft_line_part(&mut values, twiddle_tree);
        DEVICE.synchronize().unwrap();

        Self::rfft_circle_part(&mut values, twiddle_tree);
        DEVICE.synchronize().unwrap();

        CircleEvaluation::new(domain, values)
    }

    fn precompute_twiddles(mut coset: Coset) -> TwiddleTree<Self> {
        let root_coset = coset;
        // Compute twiddles
        let config = Self::launch_config_for_num_elems(coset.size() as u32, 256, 0);
        let kernel = DEVICE.get_func("circle", "precompute_twiddles").unwrap();

        let mut twiddles = BaseFieldCudaColumn::new(unsafe { DEVICE.alloc(coset.size()).unwrap() });
        let mut itwiddles =
            BaseFieldCudaColumn::new(unsafe { DEVICE.alloc(coset.size()).unwrap() });

        let mut current_level_offset = 0;

        for _ in 0..coset.log_size() {
            // Compute each level of the twiddle tree.
            unsafe {
                kernel.clone().launch(
                    config,
                    (
                        twiddles.as_mut_slice(),
                        coset.initial(),
                        coset.step_size.to_point(),
                        current_level_offset,
                        coset.size(),
                        coset.log_size(),
                    ),
                )
            }
            .unwrap();
            coset = coset.double();
            current_level_offset += coset.size();
        }
        DEVICE.synchronize().unwrap();

        // Put a one in the last position.
        let kernel = DEVICE.get_func("circle", "put_one").unwrap();
        let config = Self::launch_config_for_num_elems(1, 1, 0);
        unsafe {
            kernel
                .clone()
                .launch(config, (twiddles.as_mut_slice(), current_level_offset))
        }
        .unwrap();

        <Self as FieldOps<BaseField>>::batch_inverse(&twiddles, &mut itwiddles);

        TwiddleTree {
            root_coset,
            twiddles,
            itwiddles,
        }
    }
}

pub fn load_circle(device: &Arc<CudaDevice>) {
    let ptx_src = include_str!("circle.cu");
    let ptx = compile_ptx(ptx_src).unwrap();
    device
        .load_ptx(
            ptx,
            "circle",
            &[
                "sort_values",
                "precompute_twiddles",
                "put_one",
                "ifft_circle_part",
                "ifft_line_part",
                "rfft_circle_part",
                "rfft_line_part",
                "rescale",
                "eval_at_point_first_pass",
                "eval_at_point_second_pass",
                "get_result_from_temp",
            ],
        )
        .unwrap();
}

impl GpuBackend {
    fn ifft_circle_part(values: &mut BaseFieldCudaColumn, twiddle_tree: &TwiddleTree<GpuBackend>) {
        let size = values.len() as u32 ;
        let config = Self::launch_config_for_num_elems(size >> 1, 256, 0);
        let kernel = DEVICE.get_func("circle", "ifft_circle_part").unwrap();
        unsafe {
            kernel.launch(
                config,
                (
                    values.as_mut_slice(),
                    twiddle_tree.itwiddles.as_slice(),
                    size,
                ),
            )
        }
        .unwrap();
    }

    fn rfft_circle_part(values: &mut BaseFieldCudaColumn, twiddle_tree: &TwiddleTree<GpuBackend>) {
        let size = values.len() as u32;
        let config = Self::launch_config_for_num_elems(size >> 1, 256, 0);
        let kernel = DEVICE.get_func("circle", "rfft_circle_part").unwrap();
        unsafe {
            kernel.launch(
                config,
                (
                    values.as_mut_slice(),
                    twiddle_tree.twiddles.as_slice(),
                    size,
                ),
            )
        }
        .unwrap();
    }

    fn ifft_line_part(values: &mut BaseFieldCudaColumn, twiddle_tree: &TwiddleTree<GpuBackend>) {
        let size = values.len() as u32;
        let log_values_size = u32::BITS - size.leading_zeros() - 1;

        let mut layer_domain_size = size >> 1;
        let mut layer_domain_offset = 0;
        for i in 1..log_values_size {
            let config = Self::launch_config_for_num_elems(size >> 1, 256, 0);
            let kernel = DEVICE.get_func("circle", "ifft_line_part").unwrap();
            unsafe {
                kernel.launch(
                    config,
                    (
                        values.as_mut_slice(),
                        twiddle_tree.itwiddles.as_slice(),
                        size,
                        layer_domain_size,
                        layer_domain_offset,
                        i,
                    ),
                )
            }
            .unwrap();
            layer_domain_size >>= 1;
            layer_domain_offset += layer_domain_size;
        }
    }

    fn rfft_line_part(values: &mut BaseFieldCudaColumn, twiddle_tree: &TwiddleTree<GpuBackend>) {
        let size = values.len() as u32;
        let log_values_size = u32::BITS - size.leading_zeros() - 1;

        let mut layer_domain_size = 1;
        // TODO: fix this for size < 8
        let mut layer_domain_offset = (size >> 1) - 2;
        for i in (1..log_values_size).rev() {
            let config = Self::launch_config_for_num_elems(size >> 1, 256, 0);
            let kernel = DEVICE.get_func("circle", "rfft_line_part").unwrap();
            unsafe {
                kernel.launch(
                    config,
                    (
                        values.as_mut_slice(),
                        twiddle_tree.twiddles.as_slice(),
                        size,
                        layer_domain_size,
                        layer_domain_offset,
                        i,
                    ),
                )
            }
            .unwrap();
            if i > 1 {
                layer_domain_size <<= 1;
                layer_domain_offset -= layer_domain_size;
            }
        }
    }
    fn rescale_by_power_of_two(values: &mut BaseFieldCudaColumn, exponent: u128) {
        let size = values.len();
        let config = LaunchConfig::for_num_elems(size as u32);
        let kernel = DEVICE.get_func("circle", "rescale").unwrap();
        unsafe {
            kernel.launch(
                config,
                (
                    values.as_mut_slice(),
                    size,
                    M31(2).pow(exponent).inverse().0,
                ),
            )
        }
        .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::core::backend::gpu::column::BaseFieldCudaColumn;
    use crate::core::backend::gpu::GpuBackend;
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::fields::m31::{BaseField, M31};
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
        assert_eq!(
            result.domain.iter().collect_vec(),
            expected_result.domain.iter().collect_vec()
        );
    }

    #[test]
    fn test_precompute_twiddles() {
        let log_size = 20;

        let half_coset = CanonicCoset::new(log_size).half_coset();
        let expected_result = CpuBackend::precompute_twiddles(half_coset.clone());
        let twiddles = GpuBackend::precompute_twiddles(half_coset);

        assert_eq!(twiddles.twiddles.to_cpu(), expected_result.twiddles);
        assert_eq!(twiddles.itwiddles.to_cpu(), expected_result.itwiddles);
        assert_eq!(
            twiddles.root_coset.iter().collect_vec(),
            expected_result.root_coset.iter().collect_vec()
        );
    }

    #[test]
    fn test_interpolate() {
        let log_size = 20;

        let size = 1 << log_size;

        let cpu_values = (1..(size + 1) as u32).map(BaseField::from).collect_vec();
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

    #[test]
    fn test_evaluate() {
        let log_size = 3;

        let size = 1 << log_size;

        let cpu_values = (1..(size + 1) as u32).map(BaseField::from).collect_vec();
        let gpu_values = BaseFieldCudaColumn::from_vec(cpu_values.clone());

        let coset = CanonicCoset::new(log_size);
        let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
        let gpu_evaluations = GpuBackend::new_canonical_ordered(coset, gpu_values);

        let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
        let gpu_twiddles = GpuBackend::precompute_twiddles(coset.half_coset());

        let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
        let gpu_poly = GpuBackend::interpolate(gpu_evaluations, &gpu_twiddles);

        let expected_result = CpuBackend::evaluate(&cpu_poly, coset.circle_domain(), &cpu_twiddles);
        let result = GpuBackend::evaluate(&gpu_poly, coset.circle_domain(), &gpu_twiddles);

        assert_eq!(result.values.to_cpu(), expected_result.values);
    }

    #[test]
    fn test_eval_at_point() {
        let log_size = 23;

        let size = 1 << log_size;
        let coset = CanonicCoset::new(log_size);
        let point = SECURE_FIELD_CIRCLE_GEN;

        let cpu_values = (1..(size + 1) as u32).map(BaseField::from).collect_vec();

        let gpu_values = BaseFieldCudaColumn::from_vec(cpu_values.clone());
        let gpu_evaluations = GpuBackend::new_canonical_ordered(coset, gpu_values);
        let gpu_twiddles = GpuBackend::precompute_twiddles(coset.half_coset());
        let gpu_poly = GpuBackend::interpolate(gpu_evaluations, &gpu_twiddles);
        let result = GpuBackend::eval_at_point(&gpu_poly, point);

        let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
        let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
        let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);

        let expected_result = CpuBackend::eval_at_point(&cpu_poly, point.clone());

        assert_eq!(result, expected_result);
    }
}
