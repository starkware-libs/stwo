use super::structs::Input;
use crate::core::fields::m31::BaseField;

// TODO(ShaharS), try to make it into a for loop and use intermiddiate variables to save
// computation.
/// Given a private input, write the trace row for the wide Fibonacci example to dst.
pub fn write_trace_row(
    dst: &mut [Vec<BaseField>],
    private_input: &Input,
    row_offset: usize,
) -> (BaseField, BaseField) {
    let a = private_input.a;
    let b = private_input.b;
    let col0 = a;
    dst[0][row_offset] = col0;
    let col1 = b;
    dst[1][row_offset] = col1;
    let col2 = col0 * col0 + col1 * col1;
    dst[2][row_offset] = col2;
    let col3 = col1 * col1 + col2 * col2;
    dst[3][row_offset] = col3;
    let col4 = col2 * col2 + col3 * col3;
    dst[4][row_offset] = col4;
    let col5 = col3 * col3 + col4 * col4;
    dst[5][row_offset] = col5;
    let col6 = col4 * col4 + col5 * col5;
    dst[6][row_offset] = col6;
    let col7 = col5 * col5 + col6 * col6;
    dst[7][row_offset] = col7;
    let col8 = col6 * col6 + col7 * col7;
    dst[8][row_offset] = col8;
    let col9 = col7 * col7 + col8 * col8;
    dst[9][row_offset] = col9;
    let col10 = col8 * col8 + col9 * col9;
    dst[10][row_offset] = col10;
    let col11 = col9 * col9 + col10 * col10;
    dst[11][row_offset] = col11;
    let col12 = col10 * col10 + col11 * col11;
    dst[12][row_offset] = col12;
    let col13 = col11 * col11 + col12 * col12;
    dst[13][row_offset] = col13;
    let col14 = col12 * col12 + col13 * col13;
    dst[14][row_offset] = col14;
    let col15 = col13 * col13 + col14 * col14;
    dst[15][row_offset] = col15;
    let col16 = col14 * col14 + col15 * col15;
    dst[16][row_offset] = col16;
    let col17 = col15 * col15 + col16 * col16;
    dst[17][row_offset] = col17;
    let col18 = col16 * col16 + col17 * col17;
    dst[18][row_offset] = col18;
    let col19 = col17 * col17 + col18 * col18;
    dst[19][row_offset] = col19;
    let col20 = col18 * col18 + col19 * col19;
    dst[20][row_offset] = col20;
    let col21 = col19 * col19 + col20 * col20;
    dst[21][row_offset] = col21;
    let col22 = col20 * col20 + col21 * col21;
    dst[22][row_offset] = col22;
    let col23 = col21 * col21 + col22 * col22;
    dst[23][row_offset] = col23;
    let col24 = col22 * col22 + col23 * col23;
    dst[24][row_offset] = col24;
    let col25 = col23 * col23 + col24 * col24;
    dst[25][row_offset] = col25;
    let col26 = col24 * col24 + col25 * col25;
    dst[26][row_offset] = col26;
    let col27 = col25 * col25 + col26 * col26;
    dst[27][row_offset] = col27;
    let col28 = col26 * col26 + col27 * col27;
    dst[28][row_offset] = col28;
    let col29 = col27 * col27 + col28 * col28;
    dst[29][row_offset] = col29;
    let col30 = col28 * col28 + col29 * col29;
    dst[30][row_offset] = col30;
    let col31 = col29 * col29 + col30 * col30;
    dst[31][row_offset] = col31;
    let col32 = col30 * col30 + col31 * col31;
    dst[32][row_offset] = col32;
    let col33 = col31 * col31 + col32 * col32;
    dst[33][row_offset] = col33;
    let col34 = col32 * col32 + col33 * col33;
    dst[34][row_offset] = col34;
    let col35 = col33 * col33 + col34 * col34;
    dst[35][row_offset] = col35;
    let col36 = col34 * col34 + col35 * col35;
    dst[36][row_offset] = col36;
    let col37 = col35 * col35 + col36 * col36;
    dst[37][row_offset] = col37;
    let col38 = col36 * col36 + col37 * col37;
    dst[38][row_offset] = col38;
    let col39 = col37 * col37 + col38 * col38;
    dst[39][row_offset] = col39;
    let col40 = col38 * col38 + col39 * col39;
    dst[40][row_offset] = col40;
    let col41 = col39 * col39 + col40 * col40;
    dst[41][row_offset] = col41;
    let col42 = col40 * col40 + col41 * col41;
    dst[42][row_offset] = col42;
    let col43 = col41 * col41 + col42 * col42;
    dst[43][row_offset] = col43;
    let col44 = col42 * col42 + col43 * col43;
    dst[44][row_offset] = col44;
    let col45 = col43 * col43 + col44 * col44;
    dst[45][row_offset] = col45;
    let col46 = col44 * col44 + col45 * col45;
    dst[46][row_offset] = col46;
    let col47 = col45 * col45 + col46 * col46;
    dst[47][row_offset] = col47;
    let col48 = col46 * col46 + col47 * col47;
    dst[48][row_offset] = col48;
    let col49 = col47 * col47 + col48 * col48;
    dst[49][row_offset] = col49;
    let col50 = col48 * col48 + col49 * col49;
    dst[50][row_offset] = col50;
    let col51 = col49 * col49 + col50 * col50;
    dst[51][row_offset] = col51;
    let col52 = col50 * col50 + col51 * col51;
    dst[52][row_offset] = col52;
    let col53 = col51 * col51 + col52 * col52;
    dst[53][row_offset] = col53;
    let col54 = col52 * col52 + col53 * col53;
    dst[54][row_offset] = col54;
    let col55 = col53 * col53 + col54 * col54;
    dst[55][row_offset] = col55;
    let col56 = col54 * col54 + col55 * col55;
    dst[56][row_offset] = col56;
    let col57 = col55 * col55 + col56 * col56;
    dst[57][row_offset] = col57;
    let col58 = col56 * col56 + col57 * col57;
    dst[58][row_offset] = col58;
    let col59 = col57 * col57 + col58 * col58;
    dst[59][row_offset] = col59;
    let col60 = col58 * col58 + col59 * col59;
    dst[60][row_offset] = col60;
    let col61 = col59 * col59 + col60 * col60;
    dst[61][row_offset] = col61;
    let col62 = col60 * col60 + col61 * col61;
    dst[62][row_offset] = col62;
    let col63 = col61 * col61 + col62 * col62;
    dst[63][row_offset] = col63;

    (dst[62][row_offset], dst[63][row_offset])
}

pub fn write_lookup_column(
    dst: &mut [BaseField],
    input_trace: &[Vec<BaseField>],
    column_offset: usize,
    alpha: BaseField,
    z: BaseField,
) {
    let row_0_a = input_trace[column_offset][0];
    let row_0_b = input_trace[column_offset + 1][0];
    dst[0] = row_0_a + alpha * row_0_b - z;
    let row_1_a = input_trace[column_offset][1];
    let row_1_b = input_trace[column_offset + 1][1];
    dst[1] = (row_1_a + alpha * row_1_b - z) * dst[0];
    let row_2_a = input_trace[column_offset][2];
    let row_2_b = input_trace[column_offset + 1][2];
    dst[2] = (row_2_a + alpha * row_2_b - z) * dst[1];
    let row_3_a = input_trace[column_offset][3];
    let row_3_b = input_trace[column_offset + 1][3];
    dst[3] = (row_3_a + alpha * row_3_b - z) * dst[2];
}
