use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
pub fn practice() {
    let dev = CudaDevice::new(0).unwrap();

    let input = dev.htod_copy(vec![1.0f32; 100]).unwrap();
    let mut out = dev.alloc_zeros::<f32>(100).unwrap();

    let ptx = compile_ptx(
    "extern \"C\" __global__ void sin_kernel(float *out, const float *inp, const size_t numel) {
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < numel) {
                out[i] = sin(inp[i]);
            }
        }"

        
    ).unwrap();

    dev.load_ptx(ptx, "test", &["sin_kernel"]).unwrap();

    let sin_kernel = dev.get_func("test", "sin_kernel").unwrap();
    let cfg = LaunchConfig::for_num_elems(100);
    unsafe { sin_kernel.launch(cfg, (&mut out, &input, 100usize)) }.unwrap();

    let out_host: Vec<f32> = dev.dtoh_sync_copy(&out).unwrap();
    assert_eq!(out_host, [1.0; 100].map(f32::sin));
}

#[cfg(test)]
mod test {
    // #[test]
    // pub fn test_cuda() {
    //     super::practice();
    // }
}
