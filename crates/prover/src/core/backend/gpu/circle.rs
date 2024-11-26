use std::ops::{Add, AddAssign, Mul, Sub};

use wgpu::util::DeviceExt;
use {once_cell, wgpu};

pub struct GpuInterpolator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    interpolate_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InterpolateInput {
    values: [u32; 8],
    initial_x: u32,
    initial_y: u32,
    log_size: u32,
    circle_twiddles: [u32; 8],
    circle_twiddles_size: u32,
    line_twiddles_flat: [u32; 8],
    line_twiddles_layer_count: u32,
    line_twiddles_sizes: [u32; 8],
    line_twiddles_offsets: [u32; 8],
}

impl InterpolateInput {
    pub fn new_zero() -> Self {
        Self {
            values: [0u32; 8],
            initial_x: 0,
            initial_y: 0,
            log_size: 0,
            circle_twiddles: [0u32; 8],
            circle_twiddles_size: 0,
            line_twiddles_flat: [0u32; 8],
            line_twiddles_layer_count: 0,
            line_twiddles_sizes: [0; 8],
            line_twiddles_offsets: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InterpolateOutput {
    results: [u32; 8],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DebugData {
    index: [u32; 16],
    values: [u32; 16],
    counter: u32,
}

pub struct InterpolateInputF<F> {
    pub values: [F; 8],
    pub initial_x: F,
    pub initial_y: F,
    pub log_size: u32,
    pub circle_twiddles: [F; 8],
    pub circle_twiddles_size: u32,
    pub line_twiddles_flat: [F; 8],
    pub line_twiddles_layer_count: u32,
    pub line_twiddles_sizes: [u32; 8],
    pub line_twiddles_offsets: [u32; 8],
}

impl<F> InterpolateInputF<F>
where
    F: Into<u32> + Copy + From<u32>,
{
    pub fn new_zero() -> Self {
        Self {
            values: [F::from(0); 8],
            initial_x: F::from(0),
            initial_y: F::from(0),
            log_size: 0,
            circle_twiddles: [F::from(0); 8],
            circle_twiddles_size: 0,
            line_twiddles_flat: [F::from(0); 8],
            line_twiddles_layer_count: 0,
            line_twiddles_sizes: [0; 8],
            line_twiddles_offsets: [0; 8],
        }
    }
}

pub struct InterpolateOutputF<F> {
    pub results: [F; 8],
}

impl<F> InterpolateInputF<F>
where
    F: Into<u32> + Copy,
{
    pub fn to_gpu_input(self) -> InterpolateInput {
        InterpolateInput {
            values: self.values.map(|v| v.into()),
            initial_x: self.initial_x.into(),
            initial_y: self.initial_y.into(),
            log_size: self.log_size,
            circle_twiddles: self.circle_twiddles.map(|v| v.into()),
            circle_twiddles_size: self.circle_twiddles_size,
            line_twiddles_flat: self.line_twiddles_flat.map(|v| v.into()),
            line_twiddles_layer_count: self.line_twiddles_layer_count,
            line_twiddles_sizes: self.line_twiddles_sizes,
            line_twiddles_offsets: self.line_twiddles_offsets,
        }
    }
}

impl<F> InterpolateOutputF<F>
where
    F: From<u32> + Copy,
{
    pub fn from_gpu_output(output: InterpolateOutput) -> Self {
        Self {
            results: [
                F::from(output.results[0]),
                F::from(output.results[1]),
                F::from(output.results[2]),
                F::from(output.results[3]),
                F::from(output.results[4]),
                F::from(output.results[5]),
                F::from(output.results[6]),
                F::from(output.results[7]),
            ],
        }
    }
}

impl GpuInterpolator {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::SHADER_INT64,
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                },
                None,
            )
            .await
            .unwrap();

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("circle.wgsl").into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let interpolate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("interpolate_compute"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            device,
            queue,
            interpolate_pipeline,
            bind_group_layout,
        }
    }

    fn execute_interpolate(&self, input: InterpolateInput) -> InterpolateOutput {
        // Create input storage buffer
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[input]),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        // Create output storage buffer
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<InterpolateInput>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create debug storage buffer
        let debug_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<DebugData>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<InterpolateOutput>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // create storage buffer for debug data
        let debug_staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<DebugData>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: debug_buffer.as_entire_binding(),
                },
            ],
        });

        // Create and submit command buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.interpolate_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of::<InterpolateOutput>() as u64,
        );
        encoder.copy_buffer_to_buffer(
            &debug_buffer,
            0,
            &debug_staging_buffer,
            0,
            std::mem::size_of::<DebugData>() as u64,
        );
        self.queue.submit(Some(encoder.finish()));

        // Read back the debug data
        {
            let slice = debug_staging_buffer.slice(..);
            let (tx, rx) = flume::bounded(1);
            slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            self.device.poll(wgpu::Maintain::Wait);
            let _debug_result = pollster::block_on(async {
                rx.recv_async().await.unwrap().unwrap();
                let data = slice.get_mapped_range();
                let result = *bytemuck::from_bytes::<DebugData>(&data);
                drop(data);
                result
            });
            // println!("debug_result: {:?}", _debug_result);
        }

        // Read back the results
        let slice = staging_buffer.slice(..);
        let (tx, rx) = flume::bounded(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);

        pollster::block_on(async {
            rx.recv_async().await.unwrap().unwrap();
            let data = slice.get_mapped_range();
            let result = *bytemuck::from_bytes::<InterpolateOutput>(&data);
            drop(data);
            staging_buffer.unmap();
            result
        })
    }
}

pub fn interpolate_gpu<F>(input: InterpolateInputF<F>) -> InterpolateOutputF<F>
where
    F: AddAssign<F>
        + Add<F, Output = F>
        + Sub<F, Output = F>
        + Mul<F, Output = F>
        + Copy
        + Into<u32>
        + From<u32>,
{
    static GPU_INTERPOLATOR: once_cell::sync::Lazy<GpuInterpolator> =
        once_cell::sync::Lazy::new(|| pollster::block_on(GpuInterpolator::new()));

    let result = GPU_INTERPOLATOR.execute_interpolate(input.to_gpu_input());
    InterpolateOutputF::from_gpu_output(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::cpu::circle::circle_twiddles_from_line_twiddles;
    use crate::core::backend::cpu::{CpuCircleEvaluation, CpuCirclePoly};
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::fft::ibutterfly;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::FieldExpOps;
    use crate::core::poly::circle::{CanonicCoset, PolyOps};
    use crate::core::poly::utils::domain_line_twiddles_from_tree;
    use crate::core::poly::BitReversedOrder;

    #[test]
    fn test_interpolate2() {
        // CPU implementation for comparison
        let cpu_test = |v0: u32, v1: u32, y: u32| {
            let mut v0 = BaseField::partial_reduce(v0);
            let mut v1 = BaseField::partial_reduce(v1);
            let y = BaseField::partial_reduce(y);

            let n = BaseField::from(2);
            let yn_inv = (y * n).inverse();
            let y_inv = yn_inv * n;
            let n_inv = yn_inv * y;

            ibutterfly(&mut v0, &mut v1, y_inv);
            (v0 * n_inv, v1 * n_inv)
        };

        // Test cases
        let test_cases = [(1, 2, 3), (100, 200, 300), (1000000, 2000000, 3000000)];

        for (v0, v1, y) in test_cases {
            let (cpu_v0, cpu_v1) = cpu_test(v0, v1, y);

            let mut input = InterpolateInputF::new_zero();
            input.values[0] = BaseField::partial_reduce(v0);
            input.values[1] = BaseField::partial_reduce(v1);
            input.initial_y = BaseField::partial_reduce(y);
            input.log_size = 1;
            let gpu_output = interpolate_gpu(input);

            assert_eq!(
                cpu_v0.0, gpu_output.results[0].0,
                "v0 mismatch for input ({}, {}, {})",
                v0, v1, y
            );
            assert_eq!(
                cpu_v1.0, gpu_output.results[1].0,
                "v1 mismatch for input ({}, {}, {})",
                v0, v1, y
            );
        }
    }

    fn circle_eval_to_gpu_input(
        evals: CpuCircleEvaluation<BaseField, BitReversedOrder>,
        log_size: u32,
    ) -> InterpolateInputF<BaseField> {
        assert!(1 << log_size <= 8);

        let domain = evals.domain;
        let mut input = InterpolateInputF::new_zero();
        input.values = evals.values.to_cpu().try_into().unwrap();
        input.log_size = log_size;

        let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);

        // line twiddles
        let line_twiddles = domain_line_twiddles_from_tree(domain, &twiddles.itwiddles);
        input.line_twiddles_layer_count = line_twiddles.len() as u32;
        for (i, twiddle) in line_twiddles.iter().enumerate() {
            input.line_twiddles_sizes[i] = twiddle.len() as u32;
            // if i == 0, offset is 0, otherwise offset is sum of previous layer offset and layer
            // size
            input.line_twiddles_offsets[i] = if i == 0 {
                0
            } else {
                input.line_twiddles_offsets[i - 1] + input.line_twiddles_sizes[i - 1]
            };
            for (j, twiddle) in twiddle.iter().enumerate() {
                input.line_twiddles_flat[input.line_twiddles_offsets[i] as usize + j] = *twiddle;
            }
        }

        // circle twiddles
        let circle_twiddles: Vec<_> =
            circle_twiddles_from_line_twiddles(line_twiddles[0]).collect();
        input.circle_twiddles[..circle_twiddles.len()].copy_from_slice(&circle_twiddles);
        input.circle_twiddles_size = circle_twiddles.len() as u32;

        input
    }

    #[test]
    fn test_interpolate8() {
        let poly = CpuCirclePoly::new((1..=8).map(BaseField::from).collect());
        let domain = CanonicCoset::new(3).circle_domain();
        let evals = poly.evaluate(domain);

        // do interpolation on gpu
        let input = circle_eval_to_gpu_input(evals, 3);
        let gpu_output = interpolate_gpu(input);

        assert_eq!(gpu_output.results.to_vec(), poly.coeffs);
    }

    #[test]
    fn test_interpolate_n() {
        let max_log_size = 3;
        for log_size in 3..=max_log_size {
            let poly = CpuCirclePoly::new((1..=1 << log_size).map(BaseField::from).collect());
            let domain = CanonicCoset::new(log_size).circle_domain();
            let evals = poly.evaluate(domain);
            let input = circle_eval_to_gpu_input(evals, log_size);
            let gpu_output = interpolate_gpu(input);
            assert_eq!(gpu_output.results.to_vec(), poly.coeffs);
        }
    }

    #[test]
    fn test_interpolate4() {
        // CPU implementation for comparison
        let cpu_test = |v0: u32, v1: u32, v2: u32, v3: u32, x: u32, y: u32| {
            let mut v0 = BaseField::partial_reduce(v0);
            let mut v1 = BaseField::partial_reduce(v1);
            let mut v2 = BaseField::partial_reduce(v2);
            let mut v3 = BaseField::partial_reduce(v3);
            let x = BaseField::partial_reduce(x);
            let y = BaseField::partial_reduce(y);

            let n = BaseField::from(4);
            let xyn_inv = (x * y * n).inverse();
            let x_inv = xyn_inv * y * n;
            let y_inv = xyn_inv * x * n;
            let n_inv = xyn_inv * x * y;

            ibutterfly(&mut v0, &mut v1, y_inv);
            ibutterfly(&mut v2, &mut v3, -y_inv);
            ibutterfly(&mut v0, &mut v2, x_inv);
            ibutterfly(&mut v1, &mut v3, x_inv);
            (v0 * n_inv, v1 * n_inv, v2 * n_inv, v3 * n_inv)
        };

        // Test cases
        let test_cases = [(1, 2, 3, 4, 5, 6), (100, 200, 300, 400, 500, 600)];

        for (v0, v1, v2, v3, x, y) in test_cases {
            let (cpu_v0, cpu_v1, cpu_v2, cpu_v3) = cpu_test(v0, v1, v2, v3, x, y);

            let mut input = InterpolateInputF::new_zero();
            input.values[0] = BaseField::partial_reduce(v0);
            input.values[1] = BaseField::partial_reduce(v1);
            input.values[2] = BaseField::partial_reduce(v2);
            input.values[3] = BaseField::partial_reduce(v3);
            input.initial_x = BaseField::partial_reduce(x);
            input.initial_y = BaseField::partial_reduce(y);
            input.log_size = 2;

            let gpu_output = interpolate_gpu(input);

            assert_eq!(
                cpu_v0.0, gpu_output.results[0].0,
                "v0 mismatch for input ({}, {}, {}, {}, {}, {})",
                v0, v1, v2, v3, x, y
            );
            assert_eq!(
                cpu_v1.0, gpu_output.results[1].0,
                "v1 mismatch for input ({}, {}, {}, {}, {}, {})",
                v0, v1, v2, v3, x, y
            );
            assert_eq!(
                cpu_v2.0, gpu_output.results[2].0,
                "v2 mismatch for input ({}, {}, {}, {}, {}, {})",
                v0, v1, v2, v3, x, y
            );
            assert_eq!(
                cpu_v3.0, gpu_output.results[3].0,
                "v3 mismatch for input ({}, {}, {}, {}, {}, {})",
                v0, v1, v2, v3, x, y
            );
        }
    }
}
