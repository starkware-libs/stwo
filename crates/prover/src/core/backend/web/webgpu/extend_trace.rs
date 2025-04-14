use std::collections::HashMap;

use itertools::Itertools;
use wgpu::util::DeviceExt;

use super::qm31::GpuQM31;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::web::webgpu::qm31::GpuM31;
use crate::core::backend::{Column, CpuBackend};
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::QM31;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{CircleDomain, CirclePoly, PolyOps};
use crate::core::poly::utils::domain_line_twiddles_from_tree;
use crate::examples::poseidon::PoseidonElements;

pub const N_ROWS: u32 = 256;
pub const N_STATE: u32 = 16;
pub const N_LOG_INSTANCES_PER_ROW: u32 = 3;
pub const N_INSTANCES_PER_ROW: u32 = 1 << N_LOG_INSTANCES_PER_ROW;
pub const N_LANES: u32 = 16;
pub const N_EXTENDED_ROWS: u32 = N_ROWS * 4;
pub const N_ORIGINAL_ROWS: u32 = N_ROWS;
pub const N_CONSTRAINTS: u32 = 1144;
pub const N_COLUMNS: u32 = 1264;
pub const N_INTERACTION_COLUMNS: u32 = N_INSTANCES_PER_ROW * 4;
pub const N_WORKGROUPS: u32 = N_EXTENDED_ROWS * N_LANES / THREADS_PER_WORKGROUP;
pub const THREADS_PER_WORKGROUP: u32 = 256;
pub const N_HALF_FULL_ROUNDS: u32 = 4;
pub const N_PARTIAL_ROUNDS: u32 = 14;
pub const N_ORIGINAL_COLUMN_SIZE: u32 = N_LANES * N_ROWS;
pub const N_EXTENDED_COLUMN_SIZE: u32 = N_LANES * N_EXTENDED_ROWS;

pub const N_LINE_TWIDDLES_SIZE: u32 = N_EXTENDED_ROWS * N_LANES;
pub const N_LINE_TWIDDLES_FLAT_SIZE: u32 = N_LINE_TWIDDLES_SIZE * 2;
pub const N_CIRCLE_TWIDDLES_SIZE: u32 = N_LINE_TWIDDLES_SIZE * 2;
pub const N_ORIGINAL_TRACE_COLUMNS: u32 = 1 + N_COLUMNS + N_INTERACTION_COLUMNS;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuExtendedColumn {
    pub data: [[GpuM31; N_LANES as usize]; N_EXTENDED_ROWS as usize],
}
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Twiddles {
    pub circle_twiddles: [GpuM31; N_CIRCLE_TWIDDLES_SIZE as usize],
    pub circle_twiddles_size: u32,
    pub line_twiddles_flat: [GpuM31; N_LINE_TWIDDLES_FLAT_SIZE as usize],
    pub line_twiddles_layer_count: u32,
    pub line_twiddles_sizes: [u32; N_LINE_TWIDDLES_SIZE as usize],
    pub line_twiddles_offsets: [u32; N_LINE_TWIDDLES_SIZE as usize],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuOriginalColumn {
    pub coeffs: [GpuM31; N_ORIGINAL_COLUMN_SIZE as usize],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuExtended1DColumn {
    pub data: [GpuM31; N_EXTENDED_COLUMN_SIZE as usize],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuLookupElements {
    pub z: GpuQM31,
    pub alpha: GpuQM31,
    pub alpha_powers: [GpuQM31; N_STATE as usize],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ComputeCompositionPolynomialInput {
    pub original_trace: [GpuOriginalColumn; N_ORIGINAL_TRACE_COLUMNS as usize],
    pub twiddles: Twiddles,
    pub denom_inv: [GpuM31; 4],
    pub random_coeff_powers: [GpuQM31; N_CONSTRAINTS as usize],
    pub lookup_elements: GpuLookupElements,
    pub trace_domain_log_size: u32,
    pub eval_domain_log_size: u32,
    pub total_sum: GpuQM31,
}

impl From<PoseidonElements> for GpuLookupElements {
    fn from(value: PoseidonElements) -> Self {
        GpuLookupElements {
            z: value.0.z.into(),
            alpha: value.0.alpha.into(),
            alpha_powers: value
                .0
                .alpha_powers
                .iter()
                .map(|&x| x.into())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }
}

impl From<&&&CirclePoly<SimdBackend>> for GpuOriginalColumn {
    fn from(value: &&&CirclePoly<SimdBackend>) -> Self {
        let coeffs: [GpuM31; N_ORIGINAL_COLUMN_SIZE as usize] = value
            .coeffs
            .to_cpu()
            .into_iter()
            .map(GpuM31::from)
            .collect::<Vec<_>>()
            .try_into()
            .expect("Wrong length");

        GpuOriginalColumn { coeffs }
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

#[derive(Debug, Clone, Copy)]
pub struct ComputeCompositionPolynomialOutput {
    pub poly: [[GpuQM31; N_LANES as usize]; N_EXTENDED_ROWS as usize],
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct ExtendTraceOutput {
    pub extended_trace: [GpuExtended1DColumn; N_ORIGINAL_TRACE_COLUMNS as usize],
}

#[derive(Debug, Clone)]
pub struct ExtendTraceResults {
    pub output: ExtendTraceOutput,
}

impl ByteSerialize for GpuExtendedColumn {}
impl ByteSerialize for GpuExtended1DColumn {}
impl ByteSerialize for GpuOriginalColumn {}
impl ByteSerialize for ExtendTraceOutput {}
impl ByteSerialize for ComputeCompositionPolynomialOutput {}
impl ByteSerialize for ComputeCompositionPolynomialInput {}

impl ExtendTraceOutput {
    fn from_bytes(bytes: &[u8]) -> Self {
        unsafe { *(bytes.as_ptr() as *const Self) }
    }
}

pub struct WgpuInstance {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub staging_buffer: wgpu::Buffer,
    pub encoder: wgpu::CommandEncoder,
}

async fn init(
    original_trace: TreeVec<Vec<&&CirclePoly<SimdBackend>>>,
    eval_domain: CircleDomain,
    denom_inv: Vec<M31>,
    random_coeff_powers: Vec<QM31>,
    lookup_elements: PoseidonElements,
    trace_domain_log_size: u32,
    eval_domain_log_size: u32,
    total_sum: QM31,
) -> WgpuInstance {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    let mut limit = wgpu::Limits::default();
    limit.max_storage_buffer_binding_size = 128 << 22; // (512 MiB)
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::SHADER_INT64,
                required_limits: limit,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .unwrap();

    let input_data = create_extend_trace_gpu_input(
        original_trace,
        eval_domain,
        denom_inv,
        random_coeff_powers,
        lookup_elements,
        trace_domain_log_size,
        eval_domain_log_size,
        total_sum,
    );

    // Create buffers
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Extend Trace Input Buffer"),
        contents: input_data.as_bytes(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let buffer_size = std::mem::size_of::<ComputeCompositionPolynomialOutput>();
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Composition Polynomial Output Buffer"),
        size: buffer_size as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let extend_trace_output_buffer_size = std::mem::size_of::<ExtendTraceOutput>();
    let extend_trace_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Extend Trace Output Buffer"),
        size: extend_trace_output_buffer_size as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Load shader
    let qm31_shader = include_str!("qm31.wgsl");
    let fraction_shader = include_str!("fraction.wgsl");
    let utils_shader = include_str!("utils.wgsl");
    let extend_trace_shader = include_str!("extend_trace.wgsl");
    let combined_shader = format!(
        "{}\n
        {}\n
        {}\n
        {}",
        qm31_shader, fraction_shader, utils_shader, extend_trace_shader
    );
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Extend Trace Shader"),
        source: wgpu::ShaderSource::Wgsl(combined_shader.into()),
    });

    // Bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            // Binding 0: Input buffer
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
            // Binding 1: Output buffer
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
            // Binding 2: Extend trace output buffer
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
        label: Some("Extend Trace Bind Group Layout"),
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
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
                resource: extend_trace_output_buffer.as_entire_binding(),
            },
        ],
        label: Some("Extend Trace Bind Group"),
    });

    // Pipeline layout
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
        label: Some("Extend Trace Pipeline Layout"),
    });

    // Compute pipeline
    let evaluate_line_twiddle_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Extend Trace Line Twiddle Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("evaluate_line_twiddle"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &HashMap::from([]),
                zero_initialize_workgroup_memory: true,
            },
        });

    let evaluate_circle_twiddle_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Extend Trace Circle Twiddle Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("evaluate_circle_twiddle"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &HashMap::from([]),
                zero_initialize_workgroup_memory: true,
            },
        });

    // Create encoder
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Extend Trace Command Encoder"),
    });

    // Dispatch the compute shader
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Extend Trace Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.set_pipeline(&evaluate_line_twiddle_pipeline);
        compute_pass.dispatch_workgroups(1, 256, 1);

        compute_pass.set_pipeline(&evaluate_circle_twiddle_pipeline);
        compute_pass.dispatch_workgroups(1, 256, 1);
    }

    // Copy extend_trace_output_buffer to staging buffer for read access
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: extend_trace_output_buffer_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(
        &extend_trace_output_buffer,
        0,
        &staging_buffer,
        0,
        staging_buffer.size(),
    );

    WgpuInstance {
        instance,
        adapter,
        device,
        queue,
        staging_buffer,
        encoder,
    }
}

fn create_extend_trace_gpu_input(
    original_trace: TreeVec<Vec<&&CirclePoly<SimdBackend>>>,
    eval_domain: CircleDomain,
    denom_inv: Vec<M31>,
    random_coeff_powers: Vec<QM31>,
    lookup_elements: PoseidonElements,
    trace_domain_log_size: u32,
    eval_domain_log_size: u32,
    total_sum: QM31,
) -> ComputeCompositionPolynomialInput {
    // flatten original trace
    let original_trace_gpu: [GpuOriginalColumn; N_ORIGINAL_TRACE_COLUMNS as usize] = original_trace
        .iter()
        .flatten()
        .map(|eval| GpuOriginalColumn::from(eval))
        .collect_vec()
        .try_into()
        .expect("Wrong length");

    // flatten twiddles
    let twiddles = CpuBackend::precompute_twiddles(eval_domain.half_coset);
    let line_twiddles = domain_line_twiddles_from_tree(eval_domain, &twiddles.twiddles);
    let mut twiddle_input = Twiddles {
        line_twiddles_layer_count: line_twiddles.len() as u32,
        line_twiddles_sizes: [0; N_LINE_TWIDDLES_SIZE as usize],
        line_twiddles_offsets: [0; N_LINE_TWIDDLES_SIZE as usize],
        line_twiddles_flat: [GpuM31 { data: 0 }; N_LINE_TWIDDLES_FLAT_SIZE as usize],
        circle_twiddles: [GpuM31 { data: 0 }; N_CIRCLE_TWIDDLES_SIZE as usize],
        circle_twiddles_size: 0,
    };
    for (i, twiddle) in line_twiddles.iter().enumerate() {
        twiddle_input.line_twiddles_sizes[i] = twiddle.len() as u32;
        twiddle_input.line_twiddles_offsets[i] = if i == 0 {
            0
        } else {
            twiddle_input.line_twiddles_offsets[i - 1] + twiddle_input.line_twiddles_sizes[i - 1]
        };
        for (j, &twiddle) in twiddle.iter().enumerate() {
            twiddle_input.line_twiddles_flat[twiddle_input.line_twiddles_offsets[i] as usize + j] =
                twiddle.into();
        }
    }

    // circle twiddles
    let circle_twiddles: Vec<GpuM31> = line_twiddles[0]
        .iter()
        .array_chunks()
        .flat_map(|[&x, &y]| [y, -y, -x, x])
        .map(|twiddle| twiddle.into())
        .collect();
    twiddle_input.circle_twiddles[..circle_twiddles.len()].copy_from_slice(&circle_twiddles);
    twiddle_input.circle_twiddles_size = circle_twiddles.len() as u32;

    let denom_inv_gpu: [GpuM31; 4] = denom_inv
        .into_iter()
        .map(GpuM31::from)
        .collect::<Vec<_>>()
        .try_into()
        .expect("Wrong length");

    let random_coeff_powers_gpu: [GpuQM31; N_CONSTRAINTS as usize] = random_coeff_powers
        .into_iter()
        .map(GpuQM31::from)
        .collect::<Vec<_>>()
        .try_into()
        .expect("Wrong length");

    let lookup_elements_gpu = GpuLookupElements::from(lookup_elements);

    ComputeCompositionPolynomialInput {
        original_trace: original_trace_gpu,
        twiddles: twiddle_input,
        denom_inv: denom_inv_gpu,
        random_coeff_powers: random_coeff_powers_gpu,
        lookup_elements: lookup_elements_gpu,
        trace_domain_log_size,
        eval_domain_log_size,
        total_sum: total_sum.into(),
    }
}

pub async fn extended_trace_gpu<'a>(
    original_trace: TreeVec<Vec<&&CirclePoly<SimdBackend>>>,
    eval_domain: CircleDomain,
    denom_inv: Vec<M31>,
    random_coeff_powers: Vec<QM31>,
    lookup_elements: PoseidonElements,
    trace_domain_log_size: u32,
    eval_domain_log_size: u32,
    total_sum: QM31,
) -> ExtendTraceResults {
    let instance = init(
        original_trace,
        eval_domain,
        denom_inv,
        random_coeff_powers,
        lookup_elements,
        trace_domain_log_size,
        eval_domain_log_size,
        total_sum,
    )
    .await;
    instance.queue.submit(Some(instance.encoder.finish()));
    let output_slice = instance.staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    output_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    instance
        .device
        .poll(wgpu::Maintain::wait())
        .panic_on_timeout();
    let result = async {
        receiver.recv_async().await.unwrap().unwrap();
        let data = output_slice.get_mapped_range();
        let output = ExtendTraceOutput::from_bytes(&data);
        drop(data);
        instance.staging_buffer.unmap();
        output
    };

    let output = result.await;
    ExtendTraceResults { output }
}
