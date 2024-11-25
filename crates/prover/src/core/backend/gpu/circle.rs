use std::ops::{Add, AddAssign, Mul, Sub};

use wgpu::util::DeviceExt;
use {once_cell, wgpu};

pub struct GpuInterpolator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    interpolate_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
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

    fn execute_interpolate(&self, values: [u32; 2], initial_y: u32) -> [u32; 2] {
        // Create input buffer
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[values[0], values[1], initial_y]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create storage buffer for computation
        let storage_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 8, // 2 * sizeof(u32)
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 8,
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
        encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, 8);
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
            let result: [u32; 2] = bytemuck::cast_slice(&data).try_into().unwrap();
            drop(data);
            staging_buffer.unmap();
            result
        })
    }
}

pub fn interpolate_gpu<F>(v0: F, v1: F, y: F) -> (F, F)
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

    let [result_v0, result_v1] =
        GPU_INTERPOLATOR.execute_interpolate([v0.into(), v1.into()], y.into());

    (F::from(result_v0), F::from(result_v1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fft::ibutterfly;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::FieldExpOps;

    #[test]
    fn test_interpolate() {
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
            let (gpu_v0, gpu_v1) = interpolate_gpu(
                BaseField::partial_reduce(v0).0,
                BaseField::partial_reduce(v1).0,
                BaseField::partial_reduce(y).0,
            );

            assert_eq!(
                cpu_v0.0, gpu_v0,
                "v0 mismatch for input ({}, {}, {})",
                v0, v1, y
            );
            assert_eq!(
                cpu_v1.0, gpu_v1,
                "v1 mismatch for input ({}, {}, {})",
                v0, v1, y
            );
        }
    }
}
