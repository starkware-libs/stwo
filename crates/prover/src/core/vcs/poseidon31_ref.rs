use num_traits::Zero;
use sha2::Digest;

use crate::core::fields::m31::M31;
use crate::core::fields::Field;

/// Generated using https://github.com/HorizenLabs/poseidon2/blob/main/poseidon2_rust_params.sage
/// with p = 2^31 - 1 and t = 16

pub static MAT_DIAG16_M_1: [M31; 16] = [
    M31::from_u32_unchecked(0x07b80ac4),
    M31::from_u32_unchecked(0x6bd9cb33),
    M31::from_u32_unchecked(0x48ee3f9f),
    M31::from_u32_unchecked(0x4f63dd19),
    M31::from_u32_unchecked(0x18c546b3),
    M31::from_u32_unchecked(0x5af89e8b),
    M31::from_u32_unchecked(0x4ff23de8),
    M31::from_u32_unchecked(0x4f78aaf6),
    M31::from_u32_unchecked(0x53bdc6d4),
    M31::from_u32_unchecked(0x5c59823e),
    M31::from_u32_unchecked(0x2a471c72),
    M31::from_u32_unchecked(0x4c975e79),
    M31::from_u32_unchecked(0x58dc64d4),
    M31::from_u32_unchecked(0x06e9315d),
    M31::from_u32_unchecked(0x2cf32286),
    M31::from_u32_unchecked(0x2fb6755d),
];

pub static FIRST_FOUR_ROUND_RC: [[M31; 16]; 4] = [
    [
        M31::from_u32_unchecked(0x768bab52),
        M31::from_u32_unchecked(0x70e0ab7d),
        M31::from_u32_unchecked(0x3d266c8a),
        M31::from_u32_unchecked(0x6da42045),
        M31::from_u32_unchecked(0x600fef22),
        M31::from_u32_unchecked(0x41dace6b),
        M31::from_u32_unchecked(0x64f9bdd4),
        M31::from_u32_unchecked(0x5d42d4fe),
        M31::from_u32_unchecked(0x76b1516d),
        M31::from_u32_unchecked(0x6fc9a717),
        M31::from_u32_unchecked(0x70ac4fb6),
        M31::from_u32_unchecked(0x00194ef6),
        M31::from_u32_unchecked(0x22b644e2),
        M31::from_u32_unchecked(0x1f7916d5),
        M31::from_u32_unchecked(0x47581be2),
        M31::from_u32_unchecked(0x2710a123),
    ],
    [
        M31::from_u32_unchecked(0x6284e867),
        M31::from_u32_unchecked(0x018d3afe),
        M31::from_u32_unchecked(0x5df99ef3),
        M31::from_u32_unchecked(0x4c1e467b),
        M31::from_u32_unchecked(0x566f6abc),
        M31::from_u32_unchecked(0x2994e427),
        M31::from_u32_unchecked(0x538a6d42),
        M31::from_u32_unchecked(0x5d7bf2cf),
        M31::from_u32_unchecked(0x7fda2dab),
        M31::from_u32_unchecked(0x0fd854c4),
        M31::from_u32_unchecked(0x46922fca),
        M31::from_u32_unchecked(0x3d7763a1),
        M31::from_u32_unchecked(0x19fd05ca),
        M31::from_u32_unchecked(0x0a4bbb43),
        M31::from_u32_unchecked(0x15075851),
        M31::from_u32_unchecked(0x3d903d76),
    ],
    [
        M31::from_u32_unchecked(0x2d290ff7),
        M31::from_u32_unchecked(0x40809fa0),
        M31::from_u32_unchecked(0x59dac6ec),
        M31::from_u32_unchecked(0x127927a2),
        M31::from_u32_unchecked(0x6bbf0ea0),
        M31::from_u32_unchecked(0x0294140f),
        M31::from_u32_unchecked(0x24742976),
        M31::from_u32_unchecked(0x6e84c081),
        M31::from_u32_unchecked(0x22484f4a),
        M31::from_u32_unchecked(0x354cae59),
        M31::from_u32_unchecked(0x0453ffe1),
        M31::from_u32_unchecked(0x3f47a3cc),
        M31::from_u32_unchecked(0x0088204e),
        M31::from_u32_unchecked(0x6066e109),
        M31::from_u32_unchecked(0x3b7c4b80),
        M31::from_u32_unchecked(0x6b55665d),
    ],
    [
        M31::from_u32_unchecked(0x3bc4b897),
        M31::from_u32_unchecked(0x735bf378),
        M31::from_u32_unchecked(0x508daf42),
        M31::from_u32_unchecked(0x1884fc2b),
        M31::from_u32_unchecked(0x7214f24c),
        M31::from_u32_unchecked(0x7498be0a),
        M31::from_u32_unchecked(0x1a60e640),
        M31::from_u32_unchecked(0x3303f928),
        M31::from_u32_unchecked(0x29b46376),
        M31::from_u32_unchecked(0x5c96bb68),
        M31::from_u32_unchecked(0x65d097a5),
        M31::from_u32_unchecked(0x1d358e9f),
        M31::from_u32_unchecked(0x4a9a9017),
        M31::from_u32_unchecked(0x4724cf76),
        M31::from_u32_unchecked(0x347af70f),
        M31::from_u32_unchecked(0x1e77e59a),
    ],
];

pub static PARTIAL_ROUNDS_RC: [M31; 14] = [
    M31::from_u32_unchecked(0x7f7ec4bf),
    M31::from_u32_unchecked(0x0421926f),
    M31::from_u32_unchecked(0x5198e669),
    M31::from_u32_unchecked(0x34db3148),
    M31::from_u32_unchecked(0x4368bafd),
    M31::from_u32_unchecked(0x66685c7f),
    M31::from_u32_unchecked(0x78d3249a),
    M31::from_u32_unchecked(0x60187881),
    M31::from_u32_unchecked(0x76dad67a),
    M31::from_u32_unchecked(0x0690b437),
    M31::from_u32_unchecked(0x1ea95311),
    M31::from_u32_unchecked(0x40e5369a),
    M31::from_u32_unchecked(0x38f103fc),
    M31::from_u32_unchecked(0x1d226a21),
];

pub static LAST_FOUR_ROUNDS_RC: [[M31; 16]; 4] = [
    [
        M31::from_u32_unchecked(0x57090613),
        M31::from_u32_unchecked(0x1fa42108),
        M31::from_u32_unchecked(0x17bbef50),
        M31::from_u32_unchecked(0x1ff7e11c),
        M31::from_u32_unchecked(0x047b24ca),
        M31::from_u32_unchecked(0x4e140275),
        M31::from_u32_unchecked(0x4fa086f5),
        M31::from_u32_unchecked(0x079b309c),
        M31::from_u32_unchecked(0x1159bd47),
        M31::from_u32_unchecked(0x6d37e4e5),
        M31::from_u32_unchecked(0x075d8dce),
        M31::from_u32_unchecked(0x12121ca0),
        M31::from_u32_unchecked(0x7f6a7c40),
        M31::from_u32_unchecked(0x68e182ba),
        M31::from_u32_unchecked(0x5493201b),
        M31::from_u32_unchecked(0x0444a80e),
    ],
    [
        M31::from_u32_unchecked(0x0064f4c6),
        M31::from_u32_unchecked(0x6467abe6),
        M31::from_u32_unchecked(0x66975762),
        M31::from_u32_unchecked(0x2af68f9b),
        M31::from_u32_unchecked(0x345b33be),
        M31::from_u32_unchecked(0x1b70d47f),
        M31::from_u32_unchecked(0x053db717),
        M31::from_u32_unchecked(0x381189cb),
        M31::from_u32_unchecked(0x43b915f8),
        M31::from_u32_unchecked(0x20df3694),
        M31::from_u32_unchecked(0x0f459d26),
        M31::from_u32_unchecked(0x77a0e97b),
        M31::from_u32_unchecked(0x2f73e739),
        M31::from_u32_unchecked(0x1876c2f9),
        M31::from_u32_unchecked(0x65a0e29a),
        M31::from_u32_unchecked(0x4cabefbe),
    ],
    [
        M31::from_u32_unchecked(0x5abd1268),
        M31::from_u32_unchecked(0x4d34a760),
        M31::from_u32_unchecked(0x12771799),
        M31::from_u32_unchecked(0x69a0c9ac),
        M31::from_u32_unchecked(0x39091e55),
        M31::from_u32_unchecked(0x7f611cd0),
        M31::from_u32_unchecked(0x3af055da),
        M31::from_u32_unchecked(0x7ac0bbdf),
        M31::from_u32_unchecked(0x6e0f3a24),
        M31::from_u32_unchecked(0x41e3b6f7),
        M31::from_u32_unchecked(0x49b3756d),
        M31::from_u32_unchecked(0x568bc538),
        M31::from_u32_unchecked(0x20c079d8),
        M31::from_u32_unchecked(0x1701c72c),
        M31::from_u32_unchecked(0x7670dc6c),
        M31::from_u32_unchecked(0x5a439035),
    ],
    [
        M31::from_u32_unchecked(0x7c93e00e),
        M31::from_u32_unchecked(0x561fbb4d),
        M31::from_u32_unchecked(0x1178907b),
        M31::from_u32_unchecked(0x02737406),
        M31::from_u32_unchecked(0x32fb24f1),
        M31::from_u32_unchecked(0x6323b60a),
        M31::from_u32_unchecked(0x6ab12418),
        M31::from_u32_unchecked(0x42c99cea),
        M31::from_u32_unchecked(0x155a0b97),
        M31::from_u32_unchecked(0x53d1c6aa),
        M31::from_u32_unchecked(0x2bd20347),
        M31::from_u32_unchecked(0x279b3d73),
        M31::from_u32_unchecked(0x4f5f3c70),
        M31::from_u32_unchecked(0x0245af6c),
        M31::from_u32_unchecked(0x238359d3),
        M31::from_u32_unchecked(0x49966a59),
    ],
];

fn apply_4x4_mds_matrix(x0: M31, x1: M31, x2: M31, x3: M31) -> (M31, M31, M31, M31) {
    let t0 = x0 + x1;
    let t1 = x2 + x3;
    let t2 = x1.double() + t1;
    let t3 = x3.double() + t0;
    let t4 = t1.double().double() + t3;
    let t5 = t0.double().double() + t2;
    let t6 = t3 + t5;
    let t7 = t2 + t4;

    (t6, t5, t7, t4)
}

fn apply_16x16_mds_matrix(state: [M31; 16]) -> [M31; 16] {
    let p1 = apply_4x4_mds_matrix(state[0], state[1], state[2], state[3]);
    let p2 = apply_4x4_mds_matrix(state[4], state[5], state[6], state[7]);
    let p3 = apply_4x4_mds_matrix(state[8], state[9], state[10], state[11]);
    let p4 = apply_4x4_mds_matrix(state[12], state[13], state[14], state[15]);

    let t = [
        p1.0, p1.1, p1.2, p1.3, p2.0, p2.1, p2.2, p2.3, p3.0, p3.1, p3.2, p3.3, p4.0, p4.1, p4.2,
        p4.3,
    ];

    let mut state = t.clone();
    for i in 0..16 {
        state[i] = state[i].double();
    }

    for i in 0..4 {
        state[i] += t[i + 4];
        state[i] += t[i + 8];
        state[i] += t[i + 12];
    }
    for i in 4..8 {
        state[i] += t[i - 4];
        state[i] += t[i + 4];
        state[i] += t[i + 8];
    }
    for i in 8..12 {
        state[i] += t[i - 8];
        state[i] += t[i - 4];
        state[i] += t[i + 4];
    }
    for i in 12..16 {
        state[i] += t[i - 12];
        state[i] += t[i - 8];
        state[i] += t[i - 4];
    }

    state
}

#[inline(always)]
fn pow5(a: M31) -> M31 {
    let b = a * a;
    b * b * a
}

pub fn poseidon2_permute(p_state: &mut [M31; 16]) {
    let mut state = p_state.clone();
    state = apply_16x16_mds_matrix(state);

    for r in 0..4 {
        for i in 0..16 {
            state[i] += FIRST_FOUR_ROUND_RC[r][i];
        }
        for i in 0..16 {
            state[i] = pow5(state[i]);
        }

        state = apply_16x16_mds_matrix(state);
    }

    for r in 0..14 {
        state[0] += PARTIAL_ROUNDS_RC[r];
        state[0] = pow5(state[0]);

        let mut sum = state[0];
        for i in 1..16 {
            sum += state[i];
        }

        for i in 0..16 {
            state[i] = sum + state[i] * MAT_DIAG16_M_1[i];
        }
    }

    for r in 0..4 {
        for i in 0..16 {
            state[i] += LAST_FOUR_ROUNDS_RC[r][i];
        }
        for i in 0..16 {
            state[i] = pow5(state[i]);
        }

        state = apply_16x16_mds_matrix(state);
    }

    *p_state = state;
}

fn compute_iv_values(domain_separator: &[u8]) -> [M31; 8] {
    let mut sha256 = sha2::Sha256::new();
    Digest::update(&mut sha256, domain_separator);

    let bytes = sha256.finalize().to_vec();

    [
        M31::from(u32::from_be_bytes(bytes[0..4].try_into().unwrap())),
        M31::from(u32::from_be_bytes(bytes[4..8].try_into().unwrap())),
        M31::from(u32::from_be_bytes(bytes[8..12].try_into().unwrap())),
        M31::from(u32::from_be_bytes(bytes[12..16].try_into().unwrap())),
        M31::from(u32::from_be_bytes(bytes[16..20].try_into().unwrap())),
        M31::from(u32::from_be_bytes(bytes[20..24].try_into().unwrap())),
        M31::from(u32::from_be_bytes(bytes[24..28].try_into().unwrap())),
        M31::from(u32::from_be_bytes(bytes[28..32].try_into().unwrap())),
    ]
}

pub struct Poseidon31Hasher {
    pub buffer: Vec<M31>,
}

impl Poseidon31Hasher {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub fn update(&mut self, v: impl AsRef<[M31]>) {
        self.buffer.extend_from_slice(v.as_ref())
    }

    pub fn finalize(&self, description: impl ToString) -> [M31; 8] {
        let mut input = self.buffer.clone();
        let l = input.len();

        let iv = compute_iv_values(
            format!(
                "Poseidon2 M31 hashing {} M31 elements for {}",
                l,
                description.to_string()
            )
            .as_bytes(),
        );

        let zero = M31::zero();
        let mut state = [
            zero, zero, zero, zero, zero, zero, zero, zero, iv[0], iv[1], iv[2], iv[3], iv[4],
            iv[5], iv[6], iv[7],
        ];
        input.resize(l.div_ceil(8) * 8, zero);

        for chunk in input.chunks(8) {
            state[0..8].copy_from_slice(chunk);
            poseidon2_permute(&mut state);
        }

        [
            state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
        ]
    }

    pub fn finalize_reset(&mut self, description: impl ToString) -> [M31; 8] {
        let res = self.finalize(description);
        self.buffer.clear();
        res
    }
}

pub struct Poseidon31CRH;

impl Poseidon31CRH {
    pub fn compress(data: &[M31]) -> [M31; 8] {
        assert_eq!(data.len(), 16);

        let zero = M31::zero();
        let mut cur = [zero; 16];
        cur.copy_from_slice(data);

        poseidon2_permute(&mut cur);
        for i in 0..8 {
            cur[i] += data[i];
        }
        *cur.first_chunk().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use super::*;

    fn to_u32_array(state: [M31; 16]) -> [u32; 16] {
        [
            state[0].0,
            state[1].0,
            state[2].0,
            state[3].0,
            state[4].0,
            state[5].0,
            state[6].0,
            state[7].0,
            state[8].0,
            state[9].0,
            state[10].0,
            state[11].0,
            state[12].0,
            state[13].0,
            state[14].0,
            state[15].0,
        ]
    }

    #[test]
    fn test_poseidon2_permute() {
        let mut state = [M31::zero(); 16];
        for i in 0..16 {
            state[i] = M31::from(i);
        }

        poseidon2_permute(&mut state);

        assert_eq!(
            to_u32_array(state),
            [
                1348310665, 996460804, 2044919169, 1269301599, 615961333, 595876573, 1377780500,
                1776267289, 715842585, 1823756332, 1870636634, 1979645732, 311256455, 1364752356,
                58674647, 323699327,
            ]
        );
    }
}
