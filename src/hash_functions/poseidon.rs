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

    fn add_param_constants(&mut self) {
        self.set_state(
            self.state
                .into_iter()
                .zip(self.params.constants.iter())
                .map(|(val, constant)| val + *constant)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        );
    }

    fn hades_partial_round(&mut self) {
        self.add_param_constants();
        self.set_state_0(self.state.as_ref()[0].pow(POSEIDON_POWER as u128));
        self.set_state(self.params.mds.mul(self.state.as_ref().try_into().unwrap()));
    }

    fn hades_full_round(&mut self) {
        self.add_param_constants();
        self.set_state(
            self.state
                .as_ref()
                .iter()
                .map(|x| x.pow(POSEIDON_POWER as u128))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        );
        self.set_state(self.params.mds.mul(self.state.as_ref().try_into().unwrap()));
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
            m31!(945942046),
            m31!(1014745611),
            m31!(1083549176),
            m31!(1152352741),
            m31!(1221156306),
            m31!(1289959871),
            m31!(1358763436),
            m31!(1427567001),
            m31!(1496370566),
            m31!(1565174131),
            m31!(1633977696),
            m31!(1702781261),
            m31!(1771584826),
            m31!(1840388391),
            m31!(1909191956),
            m31!(1977995521),
            m31!(2046799086),
            m31!(2115602651),
            m31!(36922569),
            m31!(105726134),
            m31!(174529699),
            m31!(243333264),
            m31!(312136829),
            m31!(380940394),
        ]);

        hasher.hades_permutation();

        assert_eq!(hasher.state, expected_state);
    }
}
