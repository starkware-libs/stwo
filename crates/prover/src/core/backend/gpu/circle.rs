use std::ops::{Add, AddAssign, Mul, Sub};

//#[cfg(target_family = "wasm")]
// use wasm_bindgen_test::console_log;
// use {once_cell, wgpu};
use wgpu;
use wgpu::util::DeviceExt;

use crate::core::backend::cpu::circle::circle_twiddles_from_line_twiddles;
use crate::core::backend::cpu::CpuCircleEvaluation;
use crate::core::backend::{Column, CpuBackend};
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::PolyOps;
use crate::core::poly::utils::domain_line_twiddles_from_tree;
use crate::core::poly::BitReversedOrder;

pub struct GpuInterpolator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    interpolate_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

unsafe impl Send for GpuInterpolator {}
unsafe impl Sync for GpuInterpolator {}

const MAX_ARRAY_LOG_SIZE: u32 = 22;
const MAX_ARRAY_SIZE: usize = 1 << MAX_ARRAY_LOG_SIZE;

#[allow(dead_code)]
pub struct InterpolateInput {
    values: [u32; MAX_ARRAY_SIZE],
    initial_x: u32,
    initial_y: u32,
    log_size: u32,
    circle_twiddles: [u32; MAX_ARRAY_SIZE],
    circle_twiddles_size: u32,
    line_twiddles_flat: [u32; MAX_ARRAY_SIZE],
    line_twiddles_layer_count: u32,
    line_twiddles_sizes: [u32; MAX_ARRAY_SIZE],
    line_twiddles_offsets: [u32; MAX_ARRAY_SIZE],
}

#[allow(dead_code)]
pub struct InterpolateOutput {
    results: [u32; MAX_ARRAY_SIZE],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct DebugData {
    index: [u32; 16],
    values: [u32; 16],
    counter: u32,
}

pub struct InterpolateInputF<F> {
    pub values: Vec<F>,
    pub initial_x: F,
    pub initial_y: F,
    pub log_size: u32,
    pub circle_twiddles: Vec<F>,
    pub circle_twiddles_size: u32,
    pub line_twiddles_flat: Vec<F>,
    pub line_twiddles_layer_count: u32,
    pub line_twiddles_sizes: Vec<u32>,
    pub line_twiddles_offsets: Vec<u32>,
}

impl<F> InterpolateInputF<F>
where
    F: Into<u32> + Copy + From<u32>,
{
    pub fn new_zero() -> Self {
        Self {
            values: vec![F::from(0); MAX_ARRAY_SIZE],
            initial_x: F::from(0),
            initial_y: F::from(0),
            log_size: 0,
            circle_twiddles: vec![F::from(0); MAX_ARRAY_SIZE],
            circle_twiddles_size: 0,
            line_twiddles_flat: vec![F::from(0); MAX_ARRAY_SIZE],
            line_twiddles_layer_count: 0,
            line_twiddles_sizes: vec![0; MAX_ARRAY_SIZE],
            line_twiddles_offsets: vec![0; MAX_ARRAY_SIZE],
        }
    }
}

pub struct InterpolateOutputF<F> {
    pub results: Vec<F>,
}

impl<F> InterpolateInputF<F>
where
    F: Into<u32> + From<u32> + Copy,
{
    fn as_bytes(&self) -> &[u8] {
        let total_size = std::mem::size_of::<InterpolateInput>();
        let mut bytes = Vec::with_capacity(total_size);

        let mut padded_values = vec![F::from(0u32); MAX_ARRAY_SIZE];
        padded_values[..self.values.len()].copy_from_slice(&self.values);
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                padded_values.as_ptr() as *const u8,
                MAX_ARRAY_SIZE * std::mem::size_of::<F>(),
            )
        });

        // initial_x, initial_y
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.initial_x as *const F as *const u8,
                std::mem::size_of::<F>(),
            )
        });
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.initial_y as *const F as *const u8,
                std::mem::size_of::<F>(),
            )
        });

        // log_size
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.log_size as *const u32 as *const u8,
                std::mem::size_of::<u32>(),
            )
        });

        let mut padded_circle_twiddles = vec![F::from(0u32); MAX_ARRAY_SIZE];
        padded_circle_twiddles[..self.circle_twiddles.len()].copy_from_slice(&self.circle_twiddles);
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                padded_circle_twiddles.as_ptr() as *const u8,
                MAX_ARRAY_SIZE * std::mem::size_of::<F>(),
            )
        });

        // circle_twiddles_size
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.circle_twiddles_size as *const u32 as *const u8,
                std::mem::size_of::<u32>(),
            )
        });

        let mut padded_line_twiddles = vec![F::from(0u32); MAX_ARRAY_SIZE];
        padded_line_twiddles[..self.line_twiddles_flat.len()]
            .copy_from_slice(&self.line_twiddles_flat);
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                padded_line_twiddles.as_ptr() as *const u8,
                MAX_ARRAY_SIZE * std::mem::size_of::<F>(),
            )
        });

        // line_twiddles_layer_count
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.line_twiddles_layer_count as *const u32 as *const u8,
                std::mem::size_of::<u32>(),
            )
        });

        let mut padded_sizes = vec![0u32; MAX_ARRAY_SIZE];
        padded_sizes[..self.line_twiddles_sizes.len()].copy_from_slice(&self.line_twiddles_sizes);
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                padded_sizes.as_ptr() as *const u8,
                MAX_ARRAY_SIZE * std::mem::size_of::<u32>(),
            )
        });

        let mut padded_offsets = vec![0u32; MAX_ARRAY_SIZE];
        padded_offsets[..self.line_twiddles_offsets.len()]
            .copy_from_slice(&self.line_twiddles_offsets);
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                padded_offsets.as_ptr() as *const u8,
                MAX_ARRAY_SIZE * std::mem::size_of::<u32>(),
            )
        });

        Box::leak(bytes.into_boxed_slice())
    }
}

impl<F> InterpolateOutputF<F>
where
    F: From<u32> + Copy,
{
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= std::mem::size_of::<[u32; MAX_ARRAY_SIZE]>());

        let results_slice =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, MAX_ARRAY_SIZE) };

        let mut results = Vec::with_capacity(MAX_ARRAY_SIZE);
        for &value in results_slice {
            results.push(F::from(value));
        }

        Self { results }
    }
}

pub trait ByteSerialize: Sized {
    fn from_bytes(bytes: &[u8]) -> &Self {
        assert!(bytes.len() >= std::mem::size_of::<Self>());
        unsafe { &*(bytes.as_ptr() as *const Self) }
    }
}

impl ByteSerialize for DebugData {}

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
                    required_features: wgpu::Features::empty(),
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

    async fn execute_interpolate<F>(&self, input: InterpolateInputF<F>) -> InterpolateOutputF<F>
    where
        F: Into<u32> + From<u32> + Copy,
    {
        // Create input storage buffer
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: input.as_bytes(),
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

        // create staging buffer for debug data
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

        // part 1 from here
        #[cfg(not(target_family = "wasm"))]
        let part1_start = std::time::Instant::now();

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
            let workgroup_size = 1;
            compute_pass.dispatch_workgroups(workgroup_size, 1, 1);
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
        // // Read back the debug data
        {
            let slice = debug_staging_buffer.slice(..);
            let (tx, rx) = flume::bounded(1);
            slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            self.device.poll(wgpu::Maintain::Wait);
            let _debug_result = async {
                rx.recv_async().await.unwrap().unwrap();
                let data = slice.get_mapped_range();
                let result = *DebugData::from_bytes(&data);
                drop(data);
                result
            };

            // println!("debug_result: {:?}", _debug_result.await);

            // #[cfg(target_family = "wasm")]
            // console_log!("debug_result: {:?}", _debug_result.await);
        }

        // Read back the results
        let slice = staging_buffer.slice(..);
        let (tx, rx) = flume::bounded(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);

        let result = async {
            rx.recv_async().await.unwrap().unwrap();
            let data = slice.get_mapped_range();
            let result = InterpolateOutputF::from_bytes(&data);
            drop(data);
            staging_buffer.unmap();
            result
        };

        #[cfg(not(target_family = "wasm"))]
        let part1_duration = part1_start.elapsed();
        #[cfg(not(target_family = "wasm"))]
        println!("interpolate elapsed time: {:?}", part1_duration);

        result.await

        // end part 2
    }
}

pub async fn interpolate_gpu<F>(input: InterpolateInputF<F>) -> InterpolateOutputF<F>
where
    F: AddAssign<F>
        + Add<F, Output = F>
        + Sub<F, Output = F>
        + Mul<F, Output = F>
        + Copy
        + Into<u32>
        + From<u32>,
{
    #[cfg(not(target_family = "wasm"))]
    {
        static GPU_INTERPOLATOR: once_cell::sync::Lazy<GpuInterpolator> =
            once_cell::sync::Lazy::new(|| pollster::block_on(GpuInterpolator::new()));

        GPU_INTERPOLATOR.execute_interpolate(input).await
    }

    #[cfg(target_family = "wasm")]
    {
        let gpu_interpolator = GpuInterpolator::new().await;
        let result = gpu_interpolator.execute_interpolate(input).await;
        result
    }
}

pub fn circle_eval_to_gpu_input(
    evals: CpuCircleEvaluation<BaseField, BitReversedOrder>,
    log_size: u32,
) -> InterpolateInputF<BaseField> {
    let domain = evals.domain;
    let mut input = InterpolateInputF::new_zero();
    let eval_values = evals.values.to_cpu();
    input.values[..eval_values.len()].copy_from_slice(&eval_values);
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
    let circle_twiddles: Vec<_> = circle_twiddles_from_line_twiddles(line_twiddles[0]).collect();
    input.circle_twiddles[..circle_twiddles.len()].copy_from_slice(&circle_twiddles);
    input.circle_twiddles_size = circle_twiddles.len() as u32;

    input
}

#[cfg(test)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg(test)]
mod tests {
    use wasm_bindgen_test::{console_log, wasm_bindgen_test};

    use super::*;
    use crate::core::backend::cpu::CpuCirclePoly;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::CanonicCoset;

    #[test]
    fn test_interpolate8() {
        let poly = CpuCirclePoly::new((1..=8).map(BaseField::from).collect());
        let domain = CanonicCoset::new(3).circle_domain();
        let evals = poly.evaluate(domain);

        // do interpolation on gpu
        let input = circle_eval_to_gpu_input(evals, 3);
        let gpu_output = pollster::block_on(interpolate_gpu(input));

        assert_eq!(
            gpu_output.results.to_vec()[..poly.coeffs.len()],
            poly.coeffs
        );
    }

    #[test]
    fn test_interpolate_n() {
        let _max_log_size = 22;
        for log_size in 5..=_max_log_size {
            let poly = CpuCirclePoly::new((1..=1 << log_size).map(BaseField::from).collect());
            let domain = CanonicCoset::new(log_size).circle_domain();
            let evals = poly.evaluate(domain);
            let input = circle_eval_to_gpu_input(evals, log_size);
            println!("log size: {}", log_size);
            let gpu_output = pollster::block_on(interpolate_gpu(input));

            assert_eq!(
                gpu_output.results.to_vec()[..poly.coeffs.len()],
                poly.coeffs
            );
        }
    }

    #[wasm_bindgen_test]
    async fn test_interpolate_n_wasm() {
        let _max_log_size = 12;
        // alert(&format!("max log size: {}", _max_log_size));
        console_log!("max log size: {}", _max_log_size);

        for log_size in 3..=_max_log_size {
            let poly = CpuCirclePoly::new((1..=1 << log_size).map(BaseField::from).collect());
            let domain = CanonicCoset::new(log_size).circle_domain();
            let evals = poly.evaluate(domain);
            let input = circle_eval_to_gpu_input(evals, log_size);
            // println!("log size: {}", log_size);
            // alert(&format!("log size: {}", log_size));

            let gpu_output = interpolate_gpu(input).await;

            assert_eq!(
                gpu_output.results.to_vec()[..poly.coeffs.len()],
                poly.coeffs
            );
        }
    }
}
