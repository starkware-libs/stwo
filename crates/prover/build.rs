use std::path::PathBuf;

fn main() {
    let _out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let files = get_cuda_files(
        &(std::env::current_dir()
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned()
            + "/src/core/backend/gpu"),
    );
    for cu in files {
        std::process::Command::new("nvcc")
            .arg("-ptx")
            .arg(cu)
            .output()
            .expect("Failed to execute nvcc");
    }
}

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
