use std::ops::{Add, AddAssign, Mul, Sub};

use wgpu::util::DeviceExt;
use {once_cell, wgpu};

pub struct GpuFieldOps {
    device: wgpu::Device,
    queue: wgpu::Queue,
    butterfly_pipeline: wgpu::ComputePipeline,
    ibutterfly_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuFieldOps {
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

        // Load and create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("butterfly.wgsl").into()),
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

        let butterfly_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("butterfly_compute"),
            compilation_options: Default::default(),
            cache: None,
        });

        let ibutterfly_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("ibutterfly_compute"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            device,
            queue,
            butterfly_pipeline,
            ibutterfly_pipeline,
            bind_group_layout,
        }
    }

    fn execute_butterfly(&self, v0: u32, v1: u32, twiddle: u32, is_inverse: bool) -> (u32, u32) {
        // Create input buffer
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[v0, v1, twiddle]),
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
            compute_pass.set_pipeline(if is_inverse {
                &self.ibutterfly_pipeline
            } else {
                &self.butterfly_pipeline
            });
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
            (result[0], result[1])
        })
    }
}

// Generic butterfly operation interface
pub fn butterfly_gpu<F>(v0: &mut F, v1: &mut F, twid: F)
where
    F: AddAssign<F> + Sub<F, Output = F> + Mul<F, Output = F> + Copy + Into<u32> + From<u32>,
{
    static GPU_OPS: once_cell::sync::Lazy<GpuFieldOps> =
        once_cell::sync::Lazy::new(|| pollster::block_on(GpuFieldOps::new()));

    let (result_v0, result_v1) =
        GPU_OPS.execute_butterfly((*v0).into(), (*v1).into(), twid.into(), false);

    *v0 = F::from(result_v0);
    *v1 = F::from(result_v1);
}

// Generic inverse butterfly operation interface
pub fn ibutterfly_gpu<F>(v0: &mut F, v1: &mut F, itwid: F)
where
    F: AddAssign<F>
        + Add<F, Output = F>
        + Sub<F, Output = F>
        + Mul<F, Output = F>
        + Copy
        + Into<u32>
        + From<u32>,
{
    static GPU_OPS: once_cell::sync::Lazy<GpuFieldOps> =
        once_cell::sync::Lazy::new(|| pollster::block_on(GpuFieldOps::new()));

    let (result_v0, result_v1) =
        GPU_OPS.execute_butterfly((*v0).into(), (*v1).into(), itwid.into(), true);

    *v0 = F::from(result_v0);
    *v1 = F::from(result_v1);
}

mod tests {
    #[allow(unused_imports)]
    use rand::rngs::SmallRng;
    #[allow(unused_imports)]
    use rand::{Rng, SeedableRng};

    use super::{butterfly_gpu, ibutterfly_gpu};
    use crate::core::fft::{butterfly, ibutterfly};
    use crate::core::fields::m31::BaseField;

    #[allow(dead_code)]
    fn test_single_butterfly(v0: u32, v1: u32, twid: u32) -> bool {
        // CPU implementation
        let mut cpu_v0 = BaseField::partial_reduce(v0);
        let mut cpu_v1 = BaseField::partial_reduce(v1);
        let cpu_twid = BaseField::partial_reduce(twid);

        // GPU implementation
        let mut gpu_v0 = cpu_v0.0;
        let mut gpu_v1 = cpu_v1.0;
        let gpu_twid = cpu_twid.0;

        // Execute both implementations
        butterfly(&mut cpu_v0, &mut cpu_v1, cpu_twid);
        butterfly_gpu(&mut gpu_v0, &mut gpu_v1, gpu_twid);

        // Compare results
        cpu_v0.0 == gpu_v0 && cpu_v1.0 == gpu_v1
    }

    #[allow(dead_code)]
    fn test_single_ibutterfly(v0: u32, v1: u32, twid: u32) -> bool {
        // CPU implementation
        let mut cpu_v0 = BaseField::partial_reduce(v0);
        let mut cpu_v1 = BaseField::partial_reduce(v1);
        let cpu_twid = BaseField::partial_reduce(twid);

        // GPU implementation
        let mut gpu_v0 = cpu_v0.0;
        let mut gpu_v1 = cpu_v1.0;
        let gpu_twid = cpu_twid.0;

        // Execute both implementations
        ibutterfly(&mut cpu_v0, &mut cpu_v1, cpu_twid);
        ibutterfly_gpu(&mut gpu_v0, &mut gpu_v1, gpu_twid);

        // Compare results
        cpu_v0.0 == gpu_v0 && cpu_v1.0 == gpu_v1
    }

    #[test]
    fn test_butterfly_random() {
        let mut rng = SmallRng::seed_from_u64(0);

        // Generate random test case
        let v0: u32 = rng.gen();
        let v1: u32 = rng.gen();
        let twid: u32 = rng.gen();

        assert!(
            test_single_butterfly(v0, v1, twid),
            "Random butterfly test failed with v0={}, v1={}, twid={}",
            v0,
            v1,
            twid
        );

        assert!(
            test_single_ibutterfly(v0, v1, twid),
            "Random inverse butterfly test failed with v0={}, v1={}, itwid={}",
            v0,
            v1,
            twid
        );
    }
}
