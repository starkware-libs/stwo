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
    values: [u32; 4],
    initial_x: u32,
    initial_y: u32,
    log_size: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InterpolateOutput {
    results: [u32; 4],
}

pub struct InterpolateInputF<F> {
    pub values: [F; 4],
    pub initial_x: F,
    pub initial_y: F,
    pub log_size: u32,
}

pub struct InterpolateOutputF<F> {
    pub results: [F; 4],
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
        // Create input buffer
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[input]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create storage buffer for computation
        let storage_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<InterpolateInput>() as u64,
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
                    resource: storage_buffer.as_entire_binding(),
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
            &storage_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of::<InterpolateOutput>() as u64,
        );
        self.queue.submit(Some(encoder.finish()));

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
    use crate::core::fft::ibutterfly;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::FieldExpOps;

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

            let input = InterpolateInputF {
                values: [
                    BaseField::partial_reduce(v0),
                    BaseField::partial_reduce(v1),
                    BaseField::partial_reduce(0),
                    BaseField::partial_reduce(0),
                ],
                initial_x: BaseField::partial_reduce(0),
                initial_y: BaseField::partial_reduce(y),
                log_size: 1,
            };
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

            let input = InterpolateInputF {
                values: [
                    BaseField::partial_reduce(v0),
                    BaseField::partial_reduce(v1),
                    BaseField::partial_reduce(v2),
                    BaseField::partial_reduce(v3),
                ],
                initial_x: BaseField::partial_reduce(x),
                initial_y: BaseField::partial_reduce(y),
                log_size: 2,
            };
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
