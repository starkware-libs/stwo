use wgpu::util::DeviceExt;

/// Input data for the GPU computation
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ComputeInput {
    pub i: u32,
    pub domain_log_size: u32,
    pub eval_log_size: u32,
    pub offset: i32,
}

/// Output data from the GPU computation
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ComputeOutput {
    pub result: u32,
}

impl From<ComputeOutput> for usize {
    fn from(output: ComputeOutput) -> Self {
        output.result as usize
    }
}

pub trait ByteSerialize: Sized {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    fn from_bytes(bytes: &[u8]) -> &Self {
        assert!(bytes.len() >= std::mem::size_of::<Self>());
        unsafe { &*(bytes.as_ptr() as *const Self) }
    }
}

impl ByteSerialize for ComputeInput {}
impl ByteSerialize for ComputeOutput {}

/// GPU instance for utility computations
pub struct GpuUtilsInstance {
    device: wgpu::Device,
    queue: wgpu::Queue,
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
}

impl GpuUtilsInstance {
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

        Self {
            device,
            queue,
            input_buffer,
            output_buffer,
            staging_buffer,
        }
    }

    /// Creates a compute pipeline for the operation
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

    /// Runs the computation on the GPU
    async fn run_computation<T: ByteSerialize + Copy>(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroup_count: (u32, u32, u32),
    ) -> T {
        // Create command encoder and compute pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(
                workgroup_count.0,
                workgroup_count.1,
                workgroup_count.2,
            );
        }

        // Copy results to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffer,
            0,
            self.staging_buffer.size(),
        );

        // Submit command buffer and wait for results
        self.queue.submit(Some(encoder.finish()));

        // Read results from staging buffer
        let slice = self.staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);

        receiver.recv_async().await.unwrap().unwrap();
        let data = slice.get_mapped_range();
        let result = *T::from_bytes(&data);
        drop(data);
        self.staging_buffer.unmap();

        result
    }
}

#[derive(Debug)]
pub enum GpuUtilsOperation {
    OffsetBitReversedCircleDomainIndex,
}

impl GpuUtilsOperation {
    pub fn shader_source(&self) -> String {
        let base_source = include_str!("utils.wgsl");
        let qm31_source = include_str!("qm31.wgsl");

        let inputs = r#"
            struct Inputs {
                i: u32,
                domain_log_size: u32,
                eval_log_size: u32,
                offset: i32,
            }

            @group(0) @binding(0) var<storage, read> inputs: Inputs;
        "#;

        let output = r#"
            struct Output {
                result: u32,
            }

            @group(0) @binding(1) var<storage, read_write> output: Output;
        "#;

        let operation = match self {
            GpuUtilsOperation::OffsetBitReversedCircleDomainIndex => {
                r#"
                @compute @workgroup_size(1)
                fn main() {{
                    let i = inputs.i;
                    let domain_log_size = inputs.domain_log_size;
                    let eval_log_size = inputs.eval_log_size;
                    let offset = inputs.offset;

                    let result = offset_bit_reversed_circle_domain_index(i, domain_log_size, eval_log_size, offset);
                    output.result = result;
                }}
                "#
            }
        };

        format!("{base_source}\n{qm31_source}\n{inputs}\n{output}\n{operation}")
    }
}

/// Computes the offset bit reversed circle domain index using the GPU
pub async fn compute_offset_bit_reversed_circle_domain_index(
    i: usize,
    domain_log_size: u32,
    eval_log_size: u32,
    offset: i32,
) -> usize {
    let input = ComputeInput {
        i: i as u32,
        domain_log_size,
        eval_log_size,
        offset,
    };

    let instance = GpuUtilsInstance::new(&input, std::mem::size_of::<ComputeOutput>()).await;

    let shader_source = GpuUtilsOperation::OffsetBitReversedCircleDomainIndex.shader_source();
    let (pipeline, bind_group) = instance.create_pipeline(&shader_source, "main");

    let gpu_result = instance
        .run_computation::<ComputeOutput>(&pipeline, &bind_group, (1, 1, 1))
        .await;
    gpu_result.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::utils::offset_bit_reversed_circle_domain_index as cpu_offset_bit_reversed_circle_domain_index;

    #[test]
    fn test_offset_bit_reversed_circle_domain_index() {
        // Test parameters from the CPU test
        let domain_log_size = 3;
        let eval_log_size = 6;
        let initial_index = 5;
        let offset = -2;

        let gpu_result = pollster::block_on(compute_offset_bit_reversed_circle_domain_index(
            initial_index,
            domain_log_size,
            eval_log_size,
            offset,
        ));
        println!("gpu_result: {}", gpu_result);

        let cpu_result = cpu_offset_bit_reversed_circle_domain_index(
            initial_index,
            domain_log_size,
            eval_log_size,
            offset as isize,
        );

        assert_eq!(gpu_result, cpu_result, "GPU and CPU results should match");
    }
}
