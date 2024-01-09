use core::fmt;

use crate::commitment_scheme::hasher::{self, Hasher, Name};
use crate::core::fields::m31::BaseField;
use crate::math::matrix::RowMajorMatrix;

const POSEIDON_WIDTH: usize = 24; // in BaseField elements.

pub struct PoseidonParams {
    pub rate: usize,
    pub capacity: usize,
    pub full_rounds: usize,
    pub partial_rounds: usize,
    pub mds: RowMajorMatrix<BaseField, POSEIDON_WIDTH>,
}

#[derive(Clone, Copy, PartialEq, Debug, Default, Eq)]
pub struct PoseidonHash {
    state: [BaseField; POSEIDON_WIDTH],
}

impl Name for PoseidonHash {
    const NAME: std::borrow::Cow<'static, str> = std::borrow::Cow::Borrowed("Poseidon");
}

impl fmt::Display for PoseidonHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            self.as_ref()
                .iter()
                .map(|x| format!("{}, ", x))
                .collect::<String>()
                .as_str(),
        )
    }
}

impl From<PoseidonHash> for Vec<BaseField> {
    fn from(_value: PoseidonHash) -> Self {
        unimplemented!("From<PoseidonHash> for Vec<u8>")
    }
}

impl From<Vec<BaseField>> for PoseidonHash {
    fn from(_value: Vec<BaseField>) -> Self {
        unimplemented!("From<Vec<BaseField>> for PoseidonHash")
    }
}

impl From<&[BaseField]> for PoseidonHash {
    fn from(_value: &[BaseField]) -> Self {
        unimplemented!("From<&[BaseField]> for PoseidonHash")
    }
}

impl AsRef<[BaseField]> for PoseidonHash {
    fn as_ref(&self) -> &[BaseField] {
        unimplemented!("AsRef<[BaseField]> for PoseidonHash")
    }
}

impl From<PoseidonHash> for [BaseField; POSEIDON_WIDTH] {
    fn from(_value: PoseidonHash) -> Self {
        unimplemented!("From<PoseidonHash> for [BaseField; 8]")
    }
}

impl hasher::Hash<BaseField> for PoseidonHash {}

pub struct PoseidonHasher {
    _params: PoseidonParams,
    _state: PoseidonHash,
}

impl Hasher for PoseidonHasher {
    type Hash = PoseidonHash;
    const BLOCK_SIZE: usize = POSEIDON_WIDTH * 4;
    const OUTPUT_SIZE: usize = POSEIDON_WIDTH * 4;
    type NativeType = BaseField;

    fn new() -> Self {
        unimplemented!("new for PoseidonHasher")
    }

    fn reset(&mut self) {
        unimplemented!("reset for PoseidonHasher")
    }

    fn update(&mut self, _data: &[BaseField]) {
        unimplemented!("update for PoseidonHasher")
    }

    fn finalize(self) -> PoseidonHash {
        unimplemented!("finalize for PoseidonHasher")
    }

    fn finalize_reset(&mut self) -> PoseidonHash {
        unimplemented!("finalize_reset for PoseidonHasher")
    }

    fn hash_one_in_place(_data: &[Self::NativeType], _dst: &mut [Self::NativeType]) {
        unimplemented!("hash_one_in_place for PoseidonHasher")
    }

    unsafe fn hash_many_in_place(
        _data: &[*const Self::NativeType],
        _single_input_length_bytes: usize,
        _dst: &[*mut Self::NativeType],
    ) {
        unimplemented!("hash_many_in_place for PoseidonHasher")
    }

    fn concat_and_hash(_v1: &PoseidonHash, _v2: &PoseidonHash) -> PoseidonHash {
        unimplemented!("concat_and_hash for PoseidonHasher")
    }
    fn hash_many_multi_src(_data: &[Vec<&[Self::NativeType]>]) -> Vec<PoseidonHash> {
        unimplemented!("hash_many_multi_src for PoseidonHasher")
    }

    fn hash_many_multi_src_in_place(_data: &[Vec<&[Self::NativeType]>], _dst: &mut [PoseidonHash]) {
        unimplemented!("hash_many_multi_src_in_place for PoseidonHasher")
    }
}

#[cfg(test)]
mod tests {
    use crate::core::fields::m31::{BaseField, M31};
    use crate::hash_functions::poseidon::PoseidonHash;
    use crate::m31;

    #[test]
    fn poseidon_state_display_test() {
        let values = (0..24)
            .map(|x| m31!(x))
            .collect::<Vec<BaseField>>()
            .try_into()
            .unwrap();
        let poseidon_state = PoseidonHash { state: values };

        println!("Poseidon State: {:?}", poseidon_state);
    }
}
