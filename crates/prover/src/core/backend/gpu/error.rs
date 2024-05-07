use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Failed to find constant: {0}")]
    FindConstantError(String),
    #[error("Device Load Error")]
    DeviceLoadError(),
}
