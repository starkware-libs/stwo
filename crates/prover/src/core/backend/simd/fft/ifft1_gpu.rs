use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    log_step: u32,
    offset: u32,
}

// Main IFFT function matching the original signature
pub unsafe fn ifft1_gpu(values: *mut u32, offset: usize, log_step: usize, twiddles_dbl0: [u32; 1]) {
    // Convert raw pointer to slice for GPU processing
    let slice_len = 2 << log_step; // Two vectors, each with size determined by log_step
    let values_slice = std::slice::from_raw_parts(values, slice_len);

    // Run GPU computation
    pollster::block_on(async {
        if let Some(result) = execute_ifft1(values_slice, offset, log_step, &twiddles_dbl0).await {
            // Copy results back to the input pointer
            std::ptr::copy(result.as_ptr(), values, slice_len);
        } else {
            panic!("GPU IFFT execution failed");
        }
    });
}

async fn execute_ifft1(
    values: &[u32],
    offset: usize,
    log_step: usize,
    twiddles_dbl0: &[u32; 1],
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
    let cs_module = device.create_shader_module(wgpu::include_wgsl!("ifft1.wgsl"));

    // Create uniform buffer
    let uniforms = Uniforms {
        log_step: log_step as u32,
        offset: offset as u32,
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create buffers
    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let twiddle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Twiddle Buffer"),
        contents: bytemuck::cast_slice(twiddles_dbl0),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (values.len() * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
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
                resource: storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: twiddle_buffer.as_entire_binding(),
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
        compute_pass.dispatch_workgroups(1, 1, 1);
    }

    // Copy results to staging buffer
    encoder.copy_buffer_to_buffer(
        &storage_buffer,
        0,
        &staging_buffer,
        0,
        (values.len() * std::mem::size_of::<u32>()) as u64,
    );

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
