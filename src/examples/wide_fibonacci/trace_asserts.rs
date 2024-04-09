use num_traits::Zero;

use crate::core::fields::m31::BaseField;

pub fn assert_constraints_on_row(row: &[BaseField]) {
    assert_eq!(
        (row[2] - ((row[0] * row[0]) + (row[1] * row[1]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[3] - ((row[1] * row[1]) + (row[2] * row[2]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[4] - ((row[2] * row[2]) + (row[3] * row[3]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[5] - ((row[3] * row[3]) + (row[4] * row[4]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[6] - ((row[4] * row[4]) + (row[5] * row[5]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[7] - ((row[5] * row[5]) + (row[6] * row[6]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[8] - ((row[6] * row[6]) + (row[7] * row[7]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[9] - ((row[7] * row[7]) + (row[8] * row[8]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[10] - ((row[8] * row[8]) + (row[9] * row[9]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[11] - ((row[9] * row[9]) + (row[10] * row[10]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[12] - ((row[10] * row[10]) + (row[11] * row[11]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[13] - ((row[11] * row[11]) + (row[12] * row[12]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[14] - ((row[12] * row[12]) + (row[13] * row[13]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[15] - ((row[13] * row[13]) + (row[14] * row[14]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[16] - ((row[14] * row[14]) + (row[15] * row[15]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[17] - ((row[15] * row[15]) + (row[16] * row[16]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[18] - ((row[16] * row[16]) + (row[17] * row[17]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[19] - ((row[17] * row[17]) + (row[18] * row[18]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[20] - ((row[18] * row[18]) + (row[19] * row[19]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[21] - ((row[19] * row[19]) + (row[20] * row[20]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[22] - ((row[20] * row[20]) + (row[21] * row[21]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[23] - ((row[21] * row[21]) + (row[22] * row[22]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[24] - ((row[22] * row[22]) + (row[23] * row[23]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[25] - ((row[23] * row[23]) + (row[24] * row[24]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[26] - ((row[24] * row[24]) + (row[25] * row[25]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[27] - ((row[25] * row[25]) + (row[26] * row[26]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[28] - ((row[26] * row[26]) + (row[27] * row[27]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[29] - ((row[27] * row[27]) + (row[28] * row[28]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[30] - ((row[28] * row[28]) + (row[29] * row[29]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[31] - ((row[29] * row[29]) + (row[30] * row[30]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[32] - ((row[30] * row[30]) + (row[31] * row[31]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[33] - ((row[31] * row[31]) + (row[32] * row[32]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[34] - ((row[32] * row[32]) + (row[33] * row[33]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[35] - ((row[33] * row[33]) + (row[34] * row[34]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[36] - ((row[34] * row[34]) + (row[35] * row[35]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[37] - ((row[35] * row[35]) + (row[36] * row[36]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[38] - ((row[36] * row[36]) + (row[37] * row[37]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[39] - ((row[37] * row[37]) + (row[38] * row[38]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[40] - ((row[38] * row[38]) + (row[39] * row[39]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[41] - ((row[39] * row[39]) + (row[40] * row[40]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[42] - ((row[40] * row[40]) + (row[41] * row[41]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[43] - ((row[41] * row[41]) + (row[42] * row[42]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[44] - ((row[42] * row[42]) + (row[43] * row[43]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[45] - ((row[43] * row[43]) + (row[44] * row[44]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[46] - ((row[44] * row[44]) + (row[45] * row[45]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[47] - ((row[45] * row[45]) + (row[46] * row[46]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[48] - ((row[46] * row[46]) + (row[47] * row[47]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[49] - ((row[47] * row[47]) + (row[48] * row[48]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[50] - ((row[48] * row[48]) + (row[49] * row[49]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[51] - ((row[49] * row[49]) + (row[50] * row[50]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[52] - ((row[50] * row[50]) + (row[51] * row[51]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[53] - ((row[51] * row[51]) + (row[52] * row[52]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[54] - ((row[52] * row[52]) + (row[53] * row[53]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[55] - ((row[53] * row[53]) + (row[54] * row[54]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[56] - ((row[54] * row[54]) + (row[55] * row[55]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[57] - ((row[55] * row[55]) + (row[56] * row[56]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[58] - ((row[56] * row[56]) + (row[57] * row[57]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[59] - ((row[57] * row[57]) + (row[58] * row[58]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[60] - ((row[58] * row[58]) + (row[59] * row[59]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[61] - ((row[59] * row[59]) + (row[60] * row[60]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[62] - ((row[60] * row[60]) + (row[61] * row[61]))),
        BaseField::zero()
    );
    assert_eq!(
        (row[63] - ((row[61] * row[61]) + (row[62] * row[62]))),
        BaseField::zero()
    );
}

pub fn assert_constraints_on_lookup_column(
    columns: &[Vec<BaseField>],
    input_trace: &[Vec<BaseField>],
    alpha: BaseField,
    z: BaseField,
) {
    assert_eq!(
        (columns[0][0] - (input_trace[0][0] + alpha * input_trace[1][0] - z)),
        BaseField::zero()
    );
    assert_eq!(
        (columns[0][1] - ((input_trace[0][1] + alpha * input_trace[1][1] - z) * columns[0][0])),
        BaseField::zero()
    );
    assert_eq!(
        (columns[0][2] - ((input_trace[0][2] + alpha * input_trace[1][2] - z) * columns[0][1])),
        BaseField::zero()
    );
    assert_eq!(
        (columns[0][3] - ((input_trace[0][3] + alpha * input_trace[1][3] - z) * columns[0][2])),
        BaseField::zero()
    );
    assert_eq!(
        (columns[1][0] - (input_trace[62][0] + alpha * input_trace[63][0] - z)),
        BaseField::zero()
    );
    assert_eq!(
        (columns[1][1] - ((input_trace[62][1] + alpha * input_trace[63][1] - z) * columns[1][0])),
        BaseField::zero()
    );
    assert_eq!(
        (columns[1][2] - ((input_trace[62][2] + alpha * input_trace[63][2] - z) * columns[1][1])),
        BaseField::zero()
    );
    assert_eq!(
        (columns[1][3] - ((input_trace[62][3] + alpha * input_trace[63][3] - z) * columns[1][2])),
        BaseField::zero()
    );
    assert_eq!(
        (input_trace[0][0] + alpha * input_trace[1][0] - z) * columns[1][3]
            - (input_trace[62][3] + alpha * input_trace[63][3] - z) * columns[0][3],
        BaseField::zero()
    );
}
