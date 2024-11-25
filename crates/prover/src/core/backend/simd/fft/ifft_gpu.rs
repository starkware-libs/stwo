use std::mem::size_of;

use wgpu::util::DeviceExt;

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct IFFTParams {
    log_step: u32,
    offset: u32,
}

async fn execute_ifft(
    values: &[u32],
    twiddles_dbl0: &[u32; 4],
    twiddles_dbl1: &[u32; 2],
    twiddles_dbl2: &[u32; 1],
    params: &IFFTParams,
) -> Option<Vec<u32>> {
    // Create GPU instance
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;

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

    // Load shader module
    let cs_module = device.create_shader_module(wgpu::include_wgsl!("ifft3.wgsl"));

    // Calculate buffer sizes
    let values_size = size_of::<u32>() * values.len();
    // let twiddles_dbl0_size = size_of::<u32>() * 4;
    // let twiddles_dbl1_size = size_of::<u32>() * 2;
    // let twiddles_dbl2_size = size_of::<u32>() * 1;
    // let uniforms_size = size_of::<IFFTParams>();

    // Create staging buffer (for reading results)
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: values_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create uniform buffer
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: bytemuck::bytes_of(params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create storage buffers
    let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Values Buffer"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let twiddles_dbl0_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Twiddles Dbl0 Buffer"),
        contents: bytemuck::cast_slice(twiddles_dbl0),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let twiddles_dbl1_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Twiddles Dbl1 Buffer"),
        contents: bytemuck::cast_slice(twiddles_dbl1),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let twiddles_dbl2_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Twiddles Dbl2 Buffer"),
        contents: bytemuck::cast_slice(twiddles_dbl2),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create compute pipeline
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("IFFT Pipeline"),
        layout: None,
        module: &cs_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create bind group
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("IFFT Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: values_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: twiddles_dbl0_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: twiddles_dbl1_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: twiddles_dbl2_buffer.as_entire_binding(),
            },
        ],
    });

    // Create command encoder and record commands
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("IFFT Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("IFFT Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(1, 1, 1); // Adjust workgroup size according to shader
    }

    // Copy results to staging buffer
    encoder.copy_buffer_to_buffer(&values_buffer, 0, &staging_buffer, 0, values_size as u64);

    // Submit command
    queue.submit(Some(encoder.finish()));

    // Read results
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    if let Ok(Ok(())) = receiver.recv_async().await {
        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        Some(result)
    } else {
        None
    }
}

pub unsafe fn ifft3_gpu(
    values: *mut u32,
    offset: usize,
    log_step: usize,
    twiddles_dbl0: [u32; 4],
    twiddles_dbl1: [u32; 2],
    twiddles_dbl2: [u32; 1],
) {
    // Convert raw pointer to slice for GPU processing
    let slice_len = 8 << log_step;
    let values_slice = std::slice::from_raw_parts(values, slice_len);

    // Create parameters for GPU execution
    let params = IFFTParams {
        log_step: log_step as u32,
        offset: offset as u32,
    };

    // Run GPU computation using pollster to block on the async operation
    pollster::block_on(async {
        if let Some(result) = execute_ifft(
            values_slice,
            &twiddles_dbl0,
            &twiddles_dbl1,
            &twiddles_dbl2,
            &params,
        )
        .await
        {
            // Copy results back to the input pointer
            std::ptr::copy(result.as_ptr(), values, slice_len);
        } else {
            panic!("GPU IFFT execution failed");
        }
    });
}
