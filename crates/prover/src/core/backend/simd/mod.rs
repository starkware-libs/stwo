use serde::{Deserialize, Serialize};

use super::{Backend, BackendForChannel};
use crate::core::vcs::blake2_merkle::Blake2sMerkleChannel;

pub mod accumulation;
pub mod bit_reverse;
pub mod blake2s;
pub mod circle;
pub mod cm31;
pub mod column;
pub mod domain;
pub mod fft;
pub mod fri;
mod grind;
pub mod lookups;
pub mod m31;
pub mod prefix_sum;
pub mod qm31;
pub mod quotients;
mod utils;

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub struct SimdBackend;

impl Backend for SimdBackend {}
impl BackendForChannel<Blake2sMerkleChannel> for SimdBackend {}
