use core::fmt;

use num_traits::One;

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
            constants: [BaseField::one(); POSEIDON_WIDTH],
        }
    }
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
    pub fn from_state(initial_state: PoseidonHash) -> Self {
        Self {
            params: PoseidonParams::default(),
            state: initial_state,
        }
    }

    // Setter for the prefix of the state.
    fn set_state_prefix(&mut self, prefix: &[BaseField]) {
        self.state.0[..prefix.len()].copy_from_slice(prefix);
    }

    fn add_param_constants(&mut self) {
        self.set_state_prefix(
            self.state
                .into_iter()
                .zip(self.params.constants.iter())
                .map(|(val, constant)| val + *constant)
                .collect::<Vec<_>>()
                .as_slice(),
        );
    }

    fn hades_partial_round(&mut self) {
        self.add_param_constants();
        self.set_state_prefix(&[self.state.as_ref()[0].pow(POSEIDON_POWER as u128)]);
        self.set_state_prefix(
            self.params
                .mds
                .mul(self.state.as_ref().try_into().unwrap())
                .as_slice(),
        );
    }

    fn hades_full_round(&mut self) {
        self.add_param_constants();
        self.set_state_prefix(
            self.state
                .as_ref()
                .iter()
                .map(|x| x.pow(POSEIDON_POWER as u128))
                .collect::<Vec<_>>()
                .as_slice(),
        );
        self.set_state_prefix(
            self.params
                .mds
                .mul(self.state.as_ref().try_into().unwrap())
                .as_slice(),
        );
    }

    pub fn hades_permutation(&mut self) {
        for _ in 0..self.params.n_half_full_rounds {
            self.hades_full_round();
        }
        for _ in 0..self.params.n_partial_rounds {
            self.hades_partial_round();
        }
        for _ in 0..self.params.n_half_full_rounds {
            self.hades_full_round();
        }
    }
}

impl Hasher for PoseidonHasher {
    type Hash = PoseidonHash;
    const BLOCK_SIZE: usize = POSEIDON_WIDTH;
    const OUTPUT_SIZE: usize = POSEIDON_CAPACITY;
    type NativeType = BaseField;

    fn new() -> Self {
        Self::from_state(PoseidonHash::default())
    }

    fn reset(&mut self) {
        self.state = PoseidonHash::default()
    }

    fn update(&mut self, _data: &[BaseField]) {
        unimplemented!("update for PoseidonHasher")
    }

    fn finalize(mut self) -> PoseidonHash {
        self.hades_permutation();
        self.state
    }

    fn finalize_reset(&mut self) -> PoseidonHash {
        self.hades_permutation();
        let res = self.state;
        self.reset();
        res
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
    use crate::commitment_scheme::hasher::Hasher;
    use crate::core::fields::m31::{BaseField, M31};
    use crate::hash_functions::poseidon::PoseidonHash;
    use crate::m31;

    const ZERO_HASH_RESULT: [BaseField; POSEIDON_WIDTH] = [
        m31!(1783652178),
        m31!(1273199544),
        m31!(762746910),
        m31!(252294276),
        m31!(1889325289),
        m31!(1378872655),
        m31!(868420021),
        m31!(357967387),
        m31!(1994998400),
        m31!(1484545766),
        m31!(974093132),
        m31!(463640498),
        m31!(2100671511),
        m31!(1590218877),
        m31!(1079766243),
        m31!(569313609),
        m31!(58860975),
        m31!(1695891988),
        m31!(1185439354),
        m31!(674986720),
        m31!(164534086),
        m31!(1801565099),
        m31!(1291112465),
        m31!(780659831),
    ];

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
        let values = (0..24).map(|x| m31!(x)).collect::<Vec<BaseField>>();
        let mut hasher = PoseidonHasher::from_state(PoseidonHash::from(values));

        hasher.set_state_prefix(&[m31!(100)]);

        for (i, x) in hasher.state.into_iter().enumerate() {
            if i == 0 {
                assert_eq!(x, m31!(100));
            } else {
                assert_eq!(x, m31!(i as u32));
            }
        }
    }

    #[test]
    fn finalize_reset_test() {
        let mut hasher = PoseidonHasher::new();

        let res = hasher.finalize_reset();

        assert_eq!(res, PoseidonHash(ZERO_HASH_RESULT));
        assert_eq!(hasher.state, PoseidonHash::default());
    }
}
