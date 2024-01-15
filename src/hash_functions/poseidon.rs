use core::fmt;

use num_traits::Zero;

use crate::commitment_scheme::hasher::{self, Hasher, Name};
use crate::core::fields::m31::BaseField;
use crate::core::fields::Field;
use crate::math::matrix::{RowMajorMatrix, SquareMatrix};

const POSEIDON_WIDTH: usize = 24; // in BaseField elements.
const POSEIDON_CAPACITY: usize = 8; // in BaseField elements.
const POSEIDON_POWER: usize = 5;

/// Parameters for the Poseidon hash function.
/// For more info, see https://eprint.iacr.org/2019/458.pdf
pub struct PoseidonParams {
    pub rate: usize,
    pub capacity: usize,
    pub n_half_full_rounds: usize,
    pub n_partial_rounds: usize,
    pub mds: RowMajorMatrix<BaseField, POSEIDON_WIDTH>,
    // TODO(ShaharS): check if more constants are needed.
    pub constants: [BaseField; POSEIDON_WIDTH],
}

// TODO(ShaharS) Replace with real poseidon parameters.
impl Default for PoseidonParams {
    fn default() -> Self {
        Self {
            rate: POSEIDON_WIDTH - POSEIDON_CAPACITY,
            capacity: POSEIDON_CAPACITY,
            n_half_full_rounds: 1,
            n_partial_rounds: 1,
            mds: RowMajorMatrix::<BaseField, POSEIDON_WIDTH>::new(
                (0..POSEIDON_WIDTH * POSEIDON_WIDTH)
                    .map(|x| BaseField::from_u32_unchecked(x as u32))
                    .collect::<Vec<_>>(),
            ),
            constants: [BaseField::zero(); POSEIDON_WIDTH],
        }
    }
}

#[derive(PartialEq)]
pub enum HadesRoundType {
    Full,
    Partial,
}

#[derive(Clone, Copy, PartialEq, Debug, Default, Eq)]
pub struct PoseidonHash([BaseField; POSEIDON_WIDTH]);

impl Name for PoseidonHash {
    const NAME: std::borrow::Cow<'static, str> = std::borrow::Cow::Borrowed("Poseidon");
}

impl IntoIterator for PoseidonHash {
    type Item = BaseField;
    type IntoIter = std::array::IntoIter<BaseField, POSEIDON_WIDTH>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl fmt::Display for PoseidonHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for x in self.0 {
            f.write_str(&format!("{}, ", x))?;
        }
        Ok(())
    }
}

impl From<PoseidonHash> for Vec<BaseField> {
    fn from(value: PoseidonHash) -> Self {
        Self::from(value.0)
    }
}

impl From<Vec<BaseField>> for PoseidonHash {
    fn from(value: Vec<BaseField>) -> Self {
        Self::from(value.as_slice())
    }
}

impl From<&[BaseField]> for PoseidonHash {
    fn from(value: &[BaseField]) -> Self {
        Self(value.try_into().unwrap())
    }
}

impl AsRef<[BaseField]> for PoseidonHash {
    fn as_ref(&self) -> &[BaseField] {
        &self.0
    }
}

impl From<PoseidonHash> for [BaseField; POSEIDON_WIDTH] {
    fn from(value: PoseidonHash) -> Self {
        value.0
    }
}

impl hasher::Hash<BaseField> for PoseidonHash {}

pub struct PoseidonHasher {
    params: PoseidonParams,
    state: PoseidonHash,
}

impl PoseidonHasher {
    pub fn new(initial_state: PoseidonHash) -> Self {
        Self {
            params: PoseidonParams::default(),
            state: initial_state,
        }
    }

    fn set_state(&mut self, state: [BaseField; POSEIDON_WIDTH]) {
        self.state.0 = state;
    }

    // Setter the first element of the state.
    fn set_state_0(&mut self, val: BaseField) {
        self.state.0[0] = val;
    }

    pub fn hades_round(&mut self, round_type: HadesRoundType) {
        self.set_state(
            self.state
                .into_iter()
                .zip(self.params.constants.iter())
                .map(|(val, constant)| val + *constant)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        );
        if round_type == HadesRoundType::Partial {
            self.set_state_0(self.state.as_ref()[0].pow(POSEIDON_POWER as u128));
        } else {
            self.set_state(self.params.mds.mul(self.state.as_ref().try_into().unwrap()));
        }

        self.set_state(self.params.mds.mul(self.state.as_ref().try_into().unwrap()));
    }

    pub fn hades_permutation(&mut self) {
        for _ in 0..self.params.n_half_full_rounds {
            self.hades_round(HadesRoundType::Full);
        }
        for _ in 0..self.params.n_partial_rounds {
            self.hades_round(HadesRoundType::Partial);
        }
        for _ in 0..self.params.n_half_full_rounds {
            self.hades_round(HadesRoundType::Full);
        }
    }
}

impl Hasher for PoseidonHasher {
    type Hash = PoseidonHash;
    const BLOCK_SIZE: usize = POSEIDON_WIDTH;
    const OUTPUT_SIZE: usize = POSEIDON_CAPACITY;
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
    use super::{PoseidonHasher, POSEIDON_WIDTH};
    use crate::core::fields::m31::{BaseField, M31};
    use crate::hash_functions::poseidon::PoseidonHash;
    use crate::m31;

    #[test]
    fn poseidon_hash_debug_test() {
        let values = (0..24).map(|x| m31!(x)).collect::<Vec<BaseField>>();
        let poseidon_state = PoseidonHash::from(values);

        println!("Poseidon State: {:?}", poseidon_state);
    }

    #[test]
    fn poseidon_hash_iter_test() {
        let values = (0..24).map(|x| m31!(x)).collect::<Vec<BaseField>>();
        let poseidon_state = PoseidonHash::from(values);

        for (i, x) in poseidon_state.into_iter().enumerate() {
            assert_eq!(x, m31!(i as u32));
        }
    }

    #[test]
    fn poseidon_hasher_set_state_test() {
        let mut hasher = PoseidonHasher::new(PoseidonHash([m31!(0); POSEIDON_WIDTH]));
        let values = (0..24).map(|x| m31!(x)).collect::<Vec<BaseField>>();

        hasher.set_state(values.try_into().unwrap());
        hasher.set_state_0(m31!(100));

        for (i, x) in hasher.state.into_iter().enumerate() {
            if i == 0 {
                assert_eq!(x, m31!(100));
            } else {
                assert_eq!(x, m31!(i as u32));
            }
        }
    }

    #[test]
    fn hades_permutation_regression_test() {
        let mut hasher = PoseidonHasher::new(PoseidonHash([m31!(1); POSEIDON_WIDTH]));
        let expected_state = PoseidonHash([
            m31!(1866001008),
            m31!(1778447110),
            m31!(1690893212),
            m31!(1603339314),
            m31!(1515785416),
            m31!(1428231518),
            m31!(1340677620),
            m31!(1253123722),
            m31!(1165569824),
            m31!(1078015926),
            m31!(990462028),
            m31!(902908130),
            m31!(815354232),
            m31!(727800334),
            m31!(640246436),
            m31!(552692538),
            m31!(465138640),
            m31!(377584742),
            m31!(290030844),
            m31!(202476946),
            m31!(114923048),
            m31!(27369150),
            m31!(2087298899),
            m31!(1999745001),
        ]);

        hasher.hades_permutation();

        assert_eq!(hasher.state, expected_state);
    }
}
