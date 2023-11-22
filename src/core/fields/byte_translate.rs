use thiserror::Error;

pub trait ByteTranslate: Sized {
    fn to_le_bytes(self) -> Vec<u8>;
    fn to_be_bytes(self) -> Vec<u8>;
    fn write_le_bytes(&self, dst: &mut [u8]);
    fn write_be_bytes(&self, dst: &mut [u8]);
    fn read_le_bytes(src: &[u8]) -> Result<Self, ByteTranslateError>;
    fn read_be_bytes(src: &[u8]) -> Result<Self, ByteTranslateError>;
}

#[derive(Error, Debug)]
pub enum ByteTranslateError {
    #[error("invalid dst length")]
    InvalidLength,
    #[error("write failed")]
    WriteFailed,
}
