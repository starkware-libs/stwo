use std::borrow::Cow;

use wgpu::util::DeviceExt;

/// Common trait for GPU input/output types
pub trait ByteSerialize: Sized {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= std::mem::size_of::<Self>());
        unsafe { std::ptr::read(bytes.as_ptr() as *const Self) }
    }
}

/// Base GPU instance for field computations
pub struct GpuComputeInstance {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub input_buffer: wgpu::Buffer,
    pub output_buffer: wgpu::Buffer,
    pub staging_buffer: wgpu::Buffer,
    pub debug_buffer: wgpu::Buffer,
    pub staging_buffer_debug: wgpu::Buffer,
}

impl GpuComputeInstance {
    pub async fn new<T: ByteSerialize>(input_data: &T, output_size: usize) -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Field Operations Device"),
                    required_features: wgpu::Features::SHADER_INT64,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .unwrap();

        // Create input buffer
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Field Input Buffer"),
            contents: input_data.as_bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Create output buffer
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Field Output Buffer"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer for reading results
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Field Staging Buffer"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create debug buffer
        let debug_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Field Debug Buffer"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer for debug buffer
        let staging_buffer_debug = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Field Staging Buffer Debug"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            input_buffer,
            output_buffer,
            staging_buffer,
            debug_buffer,
            staging_buffer_debug,
        }
    }

    pub fn create_pipeline(
        &self,
        shader_source: &str,
        entry_point: &str,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Field Operations Shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    label: Some("Field Operations Bind Group Layout"),
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Field Operations Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Field Operations Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.output_buffer.as_entire_binding(),
                },
            ],
            label: Some("Field Operations Bind Group"),
        });

        (pipeline, bind_group)
    }

    pub fn create_pipeline_debug(
        &self,
        shader_source: &str,
        entry_point: &str,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Field Operations Shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    label: Some("Field Operations Bind Group Layout"),
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Field Operations Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Field Operations Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.debug_buffer.as_entire_binding(),
                },
            ],
            label: Some("Field Operations Bind Group"),
        });

        (pipeline, bind_group)
    }

    pub async fn run_computation<T: ByteSerialize>(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroup_count: (u32, u32, u32),
    ) -> T {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Field Operations Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Field Operations Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(
                workgroup_count.0,
                workgroup_count.1,
                workgroup_count.2,
            );
        }

        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffer,
            0,
            self.staging_buffer.size(),
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.staging_buffer.slice(..);
        let (tx, rx) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::wait());

        let result = async {
            rx.recv_async().await.unwrap().unwrap();
            let data = buffer_slice.get_mapped_range();
            let result = T::from_bytes(&data);
            drop(data);
            self.staging_buffer.unmap();
            result
        };
        let result = result.await;

        result
    }

    pub async fn run_computation_debug<T: ByteSerialize, D: ByteSerialize>(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroup_count: (u32, u32, u32),
    ) -> (T, D) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Field Operations Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Field Operations Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(
                workgroup_count.0,
                workgroup_count.1,
                workgroup_count.2,
            );
        }

        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffer,
            0,
            self.staging_buffer.size(),
        );

        encoder.copy_buffer_to_buffer(
            &self.debug_buffer,
            0,
            &self.staging_buffer_debug,
            0,
            self.staging_buffer_debug.size(),
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.staging_buffer.slice(..);
        let (tx, rx) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        let buffer_slice_debug = self.staging_buffer_debug.slice(..);
        let (tx_debug, rx_debug) = flume::bounded(1);
        buffer_slice_debug.map_async(wgpu::MapMode::Read, move |result| {
            tx_debug.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::wait());

        let debug_result = async {
            rx_debug.recv_async().await.unwrap().unwrap();
            let data = buffer_slice_debug.get_mapped_range();
            let result = D::from_bytes(&data);
            drop(data);
            self.staging_buffer_debug.unmap();
            result
        };
        let debug_result = debug_result.await;

        let result = async {
            rx.recv_async().await.unwrap().unwrap();
            let data = buffer_slice.get_mapped_range();
            let result = T::from_bytes(&data);
            drop(data);
            self.staging_buffer.unmap();
            result
        };
        let result = result.await;

        (result, debug_result)
    }
}

/// Trait for GPU operations that require shader source generation
pub trait GpuOperation {
    fn shader_source(&self) -> Cow<'static, str>;

    fn entry_point(&self) -> &'static str {
        "main"
    }
}
