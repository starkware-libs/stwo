use std::env;
use std::path::PathBuf;
//use std::io::Write;

fn main() {
    let nvcc = match env::var("NVCC") {
        Ok(var) => which::which(var),
        Err(_) => which::which("nvcc"),
    };
    if nvcc.is_ok() {
        let cuda_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap() + "/src/core/backend/gpu");
        let ptx_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap() + "/src/core/backend/gpu/ptx");

        let source_files = get_cuda_files(cuda_dir.clone().to_str().unwrap());
        for cuda_file in source_files {
            let mut ptx = cuda_file.file_stem().unwrap().to_owned();
            ptx.push(".ptx");
            let mut out_dir = ptx_dir.clone();
            out_dir.push(ptx.clone());

            println!("cargo:rerun-if-changed={}", cuda_file.to_str().unwrap());

            std::process::Command::new("nvcc")
                .arg("-ptx")
                .arg(&cuda_file)
                .arg("--output-file")
                .arg(&out_dir)
                .arg("--include-path")
                .arg(&cuda_dir)
                .output()
                .expect("Failed to execute nvcc");
        }
        //force_rebuild(&out_dir); 
    }
    println!("cargo:rerun-if-env-changed=NVCC");
}

// Grab path object to CUDA files
fn get_cuda_files(dir: &str) -> Vec<PathBuf> {
    let entries = std::fs::read_dir(dir).expect("Failed to read directory");
    let cuda_files: Vec<PathBuf> = entries
        .filter_map(|entry| {
            let entry = entry.expect("Failed to read directory entry");
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "cu") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    cuda_files
}


// // For Testing
// fn force_rebuild(out_dir: &PathBuf) {
//     let timestamp = std::path::Path::new(&out_dir).join("timestamp.txt");
//     let mut f = std::fs::File::create(&timestamp).unwrap();
//     write!(f, "{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()).unwrap();
//     println!("cargo:rerun-if-changed={}", timestamp.display());
// }