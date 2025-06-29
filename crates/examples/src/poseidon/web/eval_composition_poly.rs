use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::ptr;

use itertools::Itertools;
use stwo_constraint_framework::WebDomainEvaluator;
use stwo_prover::core::backend::cpu::circle::circle_twiddles_from_line_twiddles;
use stwo_prover::core::backend::web::webgpu::qm31::{GpuM31, GpuQM31};
use stwo_prover::core::backend::web::webgpu::ByteSerialize;
use stwo_prover::core::backend::{Column, CpuBackend};
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::poly::circle::PolyOps;
use stwo_prover::core::poly::utils::domain_line_twiddles_from_tree;

use crate::poseidon::web::*;
use crate::poseidon::PoseidonElements;

#[allow(dead_code)]
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    pub input_buffer: wgpu::Buffer,
    pub output_buffer: wgpu::Buffer,
    pub staging_buffer: wgpu::Buffer,

    pub bind_group: wgpu::BindGroup,

    pub extend_trace_pipeline: wgpu::ComputePipeline,
    pub composition_polynomial_pipeline: wgpu::ComputePipeline,
}

impl GpuContext {
    /// Create a fully‑initialised GPU context ready for compute work.
    pub async fn new() -> Self {
        // ── adapter / device ────────────────────────────────────────────────
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let mut limits = wgpu::Limits::default();
        let ext_size = std::mem::size_of::<ExtendTraceOutput>() as u64;
        limits.max_storage_buffer_binding_size = limits
            .max_storage_buffer_binding_size
            .max(ext_size as u32 + 1);
        limits.max_buffer_size = limits.max_buffer_size.max(ext_size + 1);
        limits.max_compute_workgroup_storage_size = 32 << 10; // 32 KiB.

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("stwo‑device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .unwrap();

        // ── shaders ─────────────────────────────────────────────────────────
        let constants = include_str!("constants.wgsl")
            .replace("${N_ROWS}", &N_ROWS.to_string())
            .replace("${N_CONSTRAINTS}", &N_CONSTRAINTS.to_string());
        let qm31 = include_str!("../../../../prover/src/core/backend/web/webgpu/qm31.wgsl");
        let utils = include_str!("../../../../prover/src/core/backend/web/webgpu/utils.wgsl");
        let extend = include_str!("extend_trace.wgsl");
        let comp_poly = include_str!("eval_composition_poly.wgsl");

        let mk_shader = |label: &str, src: String| -> wgpu::ShaderModule {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };

        let extend_mod = mk_shader(
            "extend-trace-shader",
            format!("{constants}\n{qm31}\n{utils}\n{extend}"),
        );
        let comp_mod = mk_shader(
            "comp-poly-shader",
            format!("{constants}\n{qm31}\n{utils}\n{comp_poly}"),
        );

        // ── buffers ─────────────────────────────────────────────────────────
        let input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("input-buffer"),
            size: std::mem::size_of::<ComputeCompositionPolynomialInput>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output-buffer"),
            size: std::mem::size_of::<ComputeCompositionPolynomialOutput>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging-buffer"),
            size: output.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let extend_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("extend-trace-buffer"),
            size: ext_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // ── bind group & layout ─────────────────────────────────────────────
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute-layout"),
            entries: &[
                Self::storage_entry(0, true),  // input
                Self::storage_entry(1, false), // output
                Self::storage_entry(2, false), // extend trace
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: extend_buf.as_entire_binding(),
                },
            ],
        });

        // ── pipelines ───────────────────────────────────────────────────────
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute-pl-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let opts = wgpu::PipelineCompilationOptions {
            constants: &HashMap::new(),
            zero_initialize_workgroup_memory: true,
        };
        let extend_trace_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("extend-trace-pipeline"),
                layout: Some(&pl_layout),
                module: &extend_mod,
                entry_point: Some("evaluate_line_twiddle_per_poly32"),
                cache: None,
                compilation_options: opts.clone(),
            });
        let composition_polynomial_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("composition-polynomial-pipeline"),
                layout: Some(&pl_layout),
                module: &comp_mod,
                entry_point: Some("compute_composition_polynomial"),
                cache: None,
                compilation_options: opts,
            });

        Self {
            instance,
            adapter,
            device,
            queue,
            input_buffer: input,
            output_buffer: output,
            staging_buffer: staging,
            bind_group,
            extend_trace_pipeline,
            composition_polynomial_pipeline,
        }
    }

    #[inline]
    fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    /// Encode both compute passes and copy result into CPU‑visible staging buffer.
    fn encode_compute(&self) -> wgpu::CommandEncoder {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("comp-poly-encoder"),
            });

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("comp-poly-pass"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);

            // Pass 1: extend trace
            pass.set_pipeline(&self.extend_trace_pipeline);
            pass.dispatch_workgroups(1, (N_ORIGINAL_TRACE_COLUMNS + 15) / 16, 1);

            // Pass 2: composition polynomial
            pass.set_pipeline(&self.composition_polynomial_pipeline);
            pass.dispatch_workgroups(N_WORKGROUPS, 1, 1);
        }

        enc.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffer,
            0,
            self.staging_buffer.size(),
        );
        enc
    }
}

#[allow(dead_code)]
pub async fn compute_composition_polynomial_wgpu(
    input: Box<ComputeCompositionPolynomialInput>,
    gpu: &GpuContext,
) -> Box<ComputeCompositionPolynomialOutput> {
    // Upload input
    gpu.queue
        .write_buffer(&gpu.input_buffer, 0, &input.as_bytes());

    let encoder = gpu.encode_compute();
    gpu.queue.submit(Some(encoder.finish()));

    // Wait for GPU completion and map the staging buffer for readback.
    let slice = gpu.staging_buffer.slice(..);
    let (tx, rx) = flume::bounded(1);
    slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    gpu.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    let _ = rx.recv_async().await.unwrap();
    let data = slice.get_mapped_range();
    let output = ComputeCompositionPolynomialOutput::from_bytes_box(&data);
    drop(data);
    gpu.staging_buffer.unmap();

    output
}

fn alloc_default_gpu_input() -> Box<ComputeCompositionPolynomialInput> {
    let mut boxed = Box::<MaybeUninit<ComputeCompositionPolynomialInput>>::new_uninit();
    let out: *mut ComputeCompositionPolynomialInput = boxed.as_mut_ptr().cast();
    unsafe {
        ptr::write_bytes(out, 0, 1);
        Box::from_raw(Box::into_raw(boxed).cast())
    }
}

pub fn build_gpu_input(
    eval: &mut WebDomainEvaluator<'_>,
    lookup_elements: &PoseidonElements,
) -> Box<ComputeCompositionPolynomialInput> {
    let mut inp = alloc_default_gpu_input();

    eval.trace_poly
        .iter()
        .flatten()
        .enumerate()
        .for_each(|(col, poly)| {
            inp.original_trace[col]
                .coeffs
                .iter_mut()
                .zip(poly.coeffs.to_cpu())
                .for_each(|(dst, src)| *dst = src.into());
        });

    let tw = CpuBackend::precompute_twiddles(eval.eval_domain.half_coset);
    let line_tw = domain_line_twiddles_from_tree(eval.eval_domain, &tw.twiddles);

    inp.twiddles.line_twiddles_layer_count = line_tw.len() as u32;
    let mut offset = 0usize;
    for (layer_idx, layer) in line_tw.iter().enumerate() {
        inp.twiddles.line_twiddles_sizes[layer_idx] = layer.len() as u32;
        inp.twiddles.line_twiddles_offsets[layer_idx] = offset as u32;

        for (j, &el) in layer.iter().enumerate() {
            inp.twiddles.line_twiddles_flat[offset + j] = GpuM31::from(el);
        }
        offset += layer.len();
    }

    let circle = circle_twiddles_from_line_twiddles(line_tw[0]);
    let circle_len = circle.try_len().unwrap();
    for (i, tw) in circle.enumerate() {
        inp.twiddles.circle_twiddles[i] = GpuM31::from(tw);
    }
    inp.twiddles.circle_twiddles_size = circle_len as u32;

    for i in 0..4 {
        inp.denom_inv[i] = GpuM31::from(eval.denom_inv[i]);
    }
    for (i, &p) in eval
        .random_coeff_powers
        .iter()
        .enumerate()
        .take(N_CONSTRAINTS as usize)
    {
        inp.random_coeff_powers[i] = GpuQM31::from(p);
    }

    inp.lookup_elements = GpuLookupElements::from(lookup_elements);
    inp.trace_domain_log_size = eval.trace_domain_log_size;
    inp.eval_domain_log_size = eval.eval_domain.log_size();
    inp.cumsum_shift =
        (eval.claimed_sum / BaseField::from_u32_unchecked(1 << eval.log_size)).into();

    inp
}
