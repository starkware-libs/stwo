// TODO: refactor this to const generics
pub const N_ROWS: u32 = 1024;
pub const N_CONSTRAINTS: u32 = 1144;

pub const N_STATE: u32 = 16;
pub const N_LOG_INSTANCES_PER_ROW: u32 = 3;
pub const N_INSTANCES_PER_ROW: u32 = 1 << N_LOG_INSTANCES_PER_ROW;
pub const N_LANES: u32 = 16;
pub const N_EXTENDED_ROWS: u32 = N_ROWS * 4;
pub const N_ORIGINAL_ROWS: u32 = N_ROWS;
pub const N_COLUMNS: u32 = 1264;
pub const N_INTERACTION_COLUMNS: u32 = N_INSTANCES_PER_ROW * 4;
#[allow(dead_code)]
pub const N_WORKGROUPS: u32 = N_EXTENDED_ROWS * N_LANES / THREADS_PER_WORKGROUP;
#[allow(dead_code)]
pub const THREADS_PER_WORKGROUP: u32 = 256;

pub const N_LINE_TWIDDLES_SIZE: u32 = 32;
pub const N_LINE_TWIDDLES_FLAT_SIZE: u32 = N_EXTENDED_ROWS * N_LANES / 2;
pub const N_CIRCLE_TWIDDLES_SIZE: u32 = N_LINE_TWIDDLES_FLAT_SIZE + 1;
pub const N_ORIGINAL_TRACE_COLUMNS: u32 = N_COLUMNS + N_INTERACTION_COLUMNS;
