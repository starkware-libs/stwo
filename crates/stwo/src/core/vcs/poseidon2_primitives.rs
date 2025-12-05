// num_traits::Zero removed as it is not used directly, implied by BaseField ops?
// Actually BaseField::zero() is used in apply_internal_round_matrix? No, sum() is used.
// If not used, good.

use crate::core::fields::m31::BaseField;

pub const N_STATE: usize = 16;
// External round constants
pub const N_EXTERNAL_ROUNDS: usize = 8;
pub const EXTERNAL_ROUND_CONSTS: [[BaseField; N_STATE]; N_EXTERNAL_ROUNDS] = [
    [BaseField::from_u32_unchecked(1323103696), BaseField::from_u32_unchecked(32820862), BaseField::from_u32_unchecked(1980729053), BaseField::from_u32_unchecked(317622338), BaseField::from_u32_unchecked(50263984), BaseField::from_u32_unchecked(427303566), BaseField::from_u32_unchecked(476470815), BaseField::from_u32_unchecked(1873216103), BaseField::from_u32_unchecked(1013492029), BaseField::from_u32_unchecked(1876243821), BaseField::from_u32_unchecked(1423021976), BaseField::from_u32_unchecked(1034880506), BaseField::from_u32_unchecked(255516447), BaseField::from_u32_unchecked(1751710500), BaseField::from_u32_unchecked(1772458188), BaseField::from_u32_unchecked(1905707724)],
    [BaseField::from_u32_unchecked(2146357039), BaseField::from_u32_unchecked(300477280), BaseField::from_u32_unchecked(1303317487), BaseField::from_u32_unchecked(1896371959), BaseField::from_u32_unchecked(1077911909), BaseField::from_u32_unchecked(1623307068), BaseField::from_u32_unchecked(1716928924), BaseField::from_u32_unchecked(1899262763), BaseField::from_u32_unchecked(561896200), BaseField::from_u32_unchecked(2147059615), BaseField::from_u32_unchecked(262690381), BaseField::from_u32_unchecked(2144164168), BaseField::from_u32_unchecked(1245079228), BaseField::from_u32_unchecked(715189338), BaseField::from_u32_unchecked(588134996), BaseField::from_u32_unchecked(1875961624)],
    [BaseField::from_u32_unchecked(727635773), BaseField::from_u32_unchecked(1044882765), BaseField::from_u32_unchecked(1256399791), BaseField::from_u32_unchecked(170160872), BaseField::from_u32_unchecked(776522156), BaseField::from_u32_unchecked(1947778522), BaseField::from_u32_unchecked(1540706240), BaseField::from_u32_unchecked(1368992253), BaseField::from_u32_unchecked(412370089), BaseField::from_u32_unchecked(1562388559), BaseField::from_u32_unchecked(1199766382), BaseField::from_u32_unchecked(257896456), BaseField::from_u32_unchecked(931242721), BaseField::from_u32_unchecked(266356162), BaseField::from_u32_unchecked(1661329514), BaseField::from_u32_unchecked(1750311239)],
    [BaseField::from_u32_unchecked(818000640), BaseField::from_u32_unchecked(1603533679), BaseField::from_u32_unchecked(1930399982), BaseField::from_u32_unchecked(1297369576), BaseField::from_u32_unchecked(725793885), BaseField::from_u32_unchecked(1909393024), BaseField::from_u32_unchecked(542194279), BaseField::from_u32_unchecked(835590442), BaseField::from_u32_unchecked(118405644), BaseField::from_u32_unchecked(363245886), BaseField::from_u32_unchecked(306379271), BaseField::from_u32_unchecked(1859125274), BaseField::from_u32_unchecked(907155627), BaseField::from_u32_unchecked(728473679), BaseField::from_u32_unchecked(68216888), BaseField::from_u32_unchecked(955416744)],
    [BaseField::from_u32_unchecked(1460405014), BaseField::from_u32_unchecked(1954678784), BaseField::from_u32_unchecked(1737828686), BaseField::from_u32_unchecked(1054416209), BaseField::from_u32_unchecked(404011322), BaseField::from_u32_unchecked(887173471), BaseField::from_u32_unchecked(2106282024), BaseField::from_u32_unchecked(89192021), BaseField::from_u32_unchecked(1805308905), BaseField::from_u32_unchecked(731574445), BaseField::from_u32_unchecked(1689910155), BaseField::from_u32_unchecked(2010105078), BaseField::from_u32_unchecked(1592067770), BaseField::from_u32_unchecked(2053284731), BaseField::from_u32_unchecked(1704275285), BaseField::from_u32_unchecked(1622667542)],
    [BaseField::from_u32_unchecked(1496650353), BaseField::from_u32_unchecked(1129998437), BaseField::from_u32_unchecked(94975783), BaseField::from_u32_unchecked(1405456603), BaseField::from_u32_unchecked(1491473593), BaseField::from_u32_unchecked(1152648986), BaseField::from_u32_unchecked(1745698830), BaseField::from_u32_unchecked(786137366), BaseField::from_u32_unchecked(1273851054), BaseField::from_u32_unchecked(46867306), BaseField::from_u32_unchecked(1106872977), BaseField::from_u32_unchecked(1239847504), BaseField::from_u32_unchecked(1618342387), BaseField::from_u32_unchecked(767578938), BaseField::from_u32_unchecked(988319243), BaseField::from_u32_unchecked(1608609998)],
    [BaseField::from_u32_unchecked(1259045680), BaseField::from_u32_unchecked(1943647915), BaseField::from_u32_unchecked(1878170765), BaseField::from_u32_unchecked(1617904628), BaseField::from_u32_unchecked(77215054), BaseField::from_u32_unchecked(1172823114), BaseField::from_u32_unchecked(270899505), BaseField::from_u32_unchecked(648507064), BaseField::from_u32_unchecked(1275491737), BaseField::from_u32_unchecked(1639546117), BaseField::from_u32_unchecked(1743480048), BaseField::from_u32_unchecked(452460390), BaseField::from_u32_unchecked(8777006), BaseField::from_u32_unchecked(137880181), BaseField::from_u32_unchecked(1299964759), BaseField::from_u32_unchecked(932562216)],
    [BaseField::from_u32_unchecked(795180932), BaseField::from_u32_unchecked(178810366), BaseField::from_u32_unchecked(104268930), BaseField::from_u32_unchecked(86930848), BaseField::from_u32_unchecked(1965844883), BaseField::from_u32_unchecked(1574834033), BaseField::from_u32_unchecked(1529304802), BaseField::from_u32_unchecked(2046056540), BaseField::from_u32_unchecked(1725752411), BaseField::from_u32_unchecked(1791806377), BaseField::from_u32_unchecked(178907537), BaseField::from_u32_unchecked(2097766673), BaseField::from_u32_unchecked(1024197625), BaseField::from_u32_unchecked(1683581695), BaseField::from_u32_unchecked(1760930095), BaseField::from_u32_unchecked(1350479555)],
];

// Internal round constants
pub const N_PARTIAL_ROUNDS: usize = 26;
pub const INTERNAL_ROUND_CONSTS: [BaseField; N_PARTIAL_ROUNDS] = [
    BaseField::from_u32_unchecked(2059409277),
    BaseField::from_u32_unchecked(1595326017),
    BaseField::from_u32_unchecked(729019563),
    BaseField::from_u32_unchecked(821223358),
    BaseField::from_u32_unchecked(821187094),
    BaseField::from_u32_unchecked(1018226477),
    BaseField::from_u32_unchecked(446527941),
    BaseField::from_u32_unchecked(1373425565),
    BaseField::from_u32_unchecked(1207007119),
    BaseField::from_u32_unchecked(810524052),
    BaseField::from_u32_unchecked(613105743),
    BaseField::from_u32_unchecked(340008665),
    BaseField::from_u32_unchecked(112809736),
    BaseField::from_u32_unchecked(418771749),
    BaseField::from_u32_unchecked(1786887756),
    BaseField::from_u32_unchecked(406920982),
    BaseField::from_u32_unchecked(458308628),
    BaseField::from_u32_unchecked(501550214),
    BaseField::from_u32_unchecked(873604502),
    BaseField::from_u32_unchecked(2101098514),
    BaseField::from_u32_unchecked(1717274910),
    BaseField::from_u32_unchecked(1611916122),
    BaseField::from_u32_unchecked(368379723),
    BaseField::from_u32_unchecked(1530763479),
    BaseField::from_u32_unchecked(1570467377),
    BaseField::from_u32_unchecked(1796879066),
];

// Internal matrix diagonal (for apply_internal_round_matrix)
pub const INTERNAL_MATRIX_DIAGONAL: [u32; N_STATE] = [4, 5, 9, 17, 33, 65, 129, 257, 513, 1025, 2049, 4097, 8193, 16385, 32769, 65537];

#[inline(always)]
fn pow5(x: BaseField) -> BaseField {
    let x2 = x * x;
    let x4 = x2 * x2;
    x4 * x
}

#[inline(always)]
fn apply_m4(x: [BaseField; 4]) -> [BaseField; 4] {
    let t0 = x[0] + x[1];
    let t02 = t0 + t0;
    let t1 = x[2] + x[3];
    let t12 = t1 + t1;
    let t2 = x[1] + x[1] + t1;
    let t3 = x[3] + x[3] + t0;
    let t4 = t12 + t12 + t3;
    let t5 = t02 + t02 + t2;
    let t6 = t3 + t5;
    let t7 = t2 + t4;
    [t6, t5, t7, t4]
}

fn apply_external_round_matrix(state: &mut [BaseField; 16]) {
    // Applies circ(2M4, M4, M4, M4).
    for i in 0..4 {
        let chunk = [state[4 * i], state[4 * i + 1], state[4 * i + 2], state[4 * i + 3]];
        let transformed = apply_m4(chunk);
        state[4 * i] = transformed[0];
        state[4 * i + 1] = transformed[1];
        state[4 * i + 2] = transformed[2];
        state[4 * i + 3] = transformed[3];
    }
    for j in 0..4 {
        let s = state[j] + state[j + 4] + state[j + 8] + state[j + 12];
        for i in 0..4 {
            state[4 * i + j] += s;
        }
    }
}

fn apply_internal_round_matrix(state: &mut [BaseField; 16]) {
    let sum: BaseField = state.iter().sum();
    for (i, s) in state.iter_mut().enumerate() {
        *s = *s * BaseField::from_u32_unchecked(INTERNAL_MATRIX_DIAGONAL[i]) + sum;
    }
}

pub fn poseidon2_permute(state: &mut [BaseField; 16]) {
    let half_full_rounds = N_EXTERNAL_ROUNDS / 2;

    // First half of full rounds
    for round in 0..half_full_rounds {
        for (i, s) in state.iter_mut().enumerate() {
            *s += EXTERNAL_ROUND_CONSTS[round][i];
        }
        for s in state.iter_mut() {
            *s = pow5(*s);
        }
        apply_external_round_matrix(state);
    }

    // Partial rounds
    for round in 0..N_PARTIAL_ROUNDS {
        state[0] += INTERNAL_ROUND_CONSTS[round];
        state[0] = pow5(state[0]);
        apply_internal_round_matrix(state);
    }

    // Second half of full rounds
    for round in 0..half_full_rounds {
        for (i, s) in state.iter_mut().enumerate() {
            *s += EXTERNAL_ROUND_CONSTS[round + half_full_rounds][i];
        }
        for s in state.iter_mut() {
            *s = pow5(*s);
        }
        apply_external_round_matrix(state);
    }
}
