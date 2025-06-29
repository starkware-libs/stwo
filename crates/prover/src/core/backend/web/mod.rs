use serde::{Deserialize, Serialize};

use super::{Backend, BackendForChannel};
use crate::core::vcs::blake2_merkle::Blake2sMerkleChannel;
#[cfg(not(target_arch = "wasm32"))]
use crate::core::vcs::poseidon252_merkle::Poseidon252MerkleChannel;

pub mod accumulation;
pub mod bit_reverse;
pub mod blake2s;
pub mod circle;
// pub mod cm31;
pub mod column;
// pub mod conversion;
// pub mod domain;
// pub mod fft;
pub mod fri;
pub mod grind;
pub mod lookups;
// pub mod m31;
#[cfg(not(target_arch = "wasm32"))]
pub mod poseidon252;
// pub mod prefix_sum;
// pub mod qm31;
pub mod quotients;
pub mod utils;
// pub mod very_packed_m31;
// pub mod prove_poseidon;
pub mod webgpu;

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub struct WebBackend;

impl Backend for WebBackend {}
impl BackendForChannel<Blake2sMerkleChannel> for WebBackend {}
#[cfg(not(target_arch = "wasm32"))]
impl BackendForChannel<Poseidon252MerkleChannel> for WebBackend {}
