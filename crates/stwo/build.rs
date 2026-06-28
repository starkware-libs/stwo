use std::path::PathBuf;

fn main() {
    // The NitrooZK device-resident CUDA backend (P4) is only built under `--features cuda`.
    // cargo sets CARGO_FEATURE_CUDA when the `cuda` feature is enabled.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    #[cfg(target_os = "macos")]
    std::process::exit(0);

    // Export the CUDA include root so a DOWNSTREAM crate that compiles its own circuit-specific
    // `.cu` constraint kernel (registered via `set_gpu_constraint_kernel`) can `#include` the
    // shared device headers. Because this package declares `links = "stwo_cuda"`, this surfaces to
    // direct dependents as `DEP_STWO_CUDA_INCLUDE`. Emitted in every path (incl. check-only).
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:include={manifest_dir}/src/stwo_cuda/cuda");

    // Allow type-checking the CUDA backend Rust without nvcc/CMake present (e.g. laptop
    // `cargo check`): STWO_CUDA_SKIP_BUILD=1 skips the native compile + link. `cargo check`
    // does not link, so the missing libstwo_cuda is fine. On the GPU box leave it unset.
    if std::env::var_os("STWO_CUDA_SKIP_BUILD").is_some() {
        println!("cargo:warning=STWO_CUDA_SKIP_BUILD set — skipping CUDA native build (check-only)");
        return;
    }

    let dst = cmake::Config::new("src/stwo_cuda/cuda")
        .profile("Release")
        .build_arg("--jobs=16")
        .build();
    // Export the dir holding libstwo_cuda.a to dependents as `DEP_STWO_CUDA_LIB_DIR`, so a
    // downstream circuit kernel can link against the shared device library it calls into.
    println!("cargo:lib_dir={}", dst.display());
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=stwo_cuda");
    let cuda_lib_path: PathBuf = PathBuf::from("/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
    println!("cargo:rustc-link-lib=cudart");
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=stdc++");
}
