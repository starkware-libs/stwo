use serde::{Deserialize, Serialize};

use super::m31::M31;

#[derive(Copy, Clone, Debug, Serialize, Deserialize, Default, Eq, PartialEq, Hash)]
pub struct CasmState {
    pub pc: M31,
    pub ap: M31,
    pub fp: M31,
}

impl CasmState {
    pub const fn values(&self) -> [M31; 3] {
        [self.pc, self.ap, self.fp]
    }
}
