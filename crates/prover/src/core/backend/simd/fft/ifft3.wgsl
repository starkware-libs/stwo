struct Uniforms {
    log_step: u32,
    offset: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> values: array<u32>;
@group(0) @binding(2) var<storage, read> twiddles_dbl0: array<u32, 4>;
@group(0) @binding(3) var<storage, read> twiddles_dbl1: array<u32, 2>;
@group(0) @binding(4) var<storage, read> twiddles_dbl2: array<u32, 1>;

fn complex_multiply(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

fn simd_ibutterfly(a: vec4<u32>, b: vec4<u32>, twiddle: u32) -> array<vec4<u32>, 2> {
    let twiddle_vec = vec2<u32>(
        twiddle & 0xFFFFu,
        (twiddle >> 16u) & 0xFFFFu
    );
    
    let b_twisted_1 = complex_multiply(
        vec2<u32>(b.x, b.y),
        twiddle_vec
    );
    let b_twisted_2 = complex_multiply(
        vec2<u32>(b.z, b.w),
        twiddle_vec
    );
    
    let b_twisted = vec4<u32>(
        b_twisted_1.x, b_twisted_1.y,
        b_twisted_2.x, b_twisted_2.y
    );
    
    return array<vec4<u32>, 2>(
        a + b_twisted,
        a - b_twisted
    );
}

@compute @workgroup_size(8, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let offset = uniforms.offset;
    let log_step = uniforms.log_step;
    
    // Load values
    var val0 = vec4<u32>(
        values[offset + (0u << log_step)],
        values[offset + (0u << log_step) + 1u],
        values[offset + (0u << log_step) + 2u],
        values[offset + (0u << log_step) + 3u]
    );
    
    var val1 = vec4<u32>(
        values[offset + (1u << log_step)],
        values[offset + (1u << log_step) + 1u],
        values[offset + (1u << log_step) + 2u],
        values[offset + (1u << log_step) + 3u]
    );
    
    var val2 = vec4<u32>(
        values[offset + (2u << log_step)],
        values[offset + (2u << log_step) + 1u],
        values[offset + (2u << log_step) + 2u],
        values[offset + (2u << log_step) + 3u]
    );
    
    var val3 = vec4<u32>(
        values[offset + (3u << log_step)],
        values[offset + (3u << log_step) + 1u],
        values[offset + (3u << log_step) + 2u],
        values[offset + (3u << log_step) + 3u]
    );
    
    var val4 = vec4<u32>(
        values[offset + (4u << log_step)],
        values[offset + (4u << log_step) + 1u],
        values[offset + (4u << log_step) + 2u],
        values[offset + (4u << log_step) + 3u]
    );
    
    var val5 = vec4<u32>(
        values[offset + (5u << log_step)],
        values[offset + (5u << log_step) + 1u],
        values[offset + (5u << log_step) + 2u],
        values[offset + (5u << log_step) + 3u]
    );
    
    var val6 = vec4<u32>(
        values[offset + (6u << log_step)],
        values[offset + (6u << log_step) + 1u],
        values[offset + (6u << log_step) + 2u],
        values[offset + (6u << log_step) + 3u]
    );
    
    var val7 = vec4<u32>(
        values[offset + (7u << log_step)],
        values[offset + (7u << log_step) + 1u],
        values[offset + (7u << log_step) + 2u],
        values[offset + (7u << log_step) + 3u]
    );

    // First layer of ibutterflies
    let temp0 = simd_ibutterfly(val0, val1, twiddles_dbl0[0]);
    val0 = temp0[0];
    val1 = temp0[1];

    let temp1 = simd_ibutterfly(val2, val3, twiddles_dbl0[1]);
    val2 = temp1[0];
    val3 = temp1[1];

    let temp2 = simd_ibutterfly(val4, val5, twiddles_dbl0[2]);
    val4 = temp2[0];
    val5 = temp2[1];

    let temp3 = simd_ibutterfly(val6, val7, twiddles_dbl0[3]);
    val6 = temp3[0];
    val7 = temp3[1];

    // Second layer of ibutterflies
    let temp4 = simd_ibutterfly(val0, val2, twiddles_dbl1[0]);
    val0 = temp4[0];
    val2 = temp4[1];

    let temp5 = simd_ibutterfly(val1, val3, twiddles_dbl1[0]);
    val1 = temp5[0];
    val3 = temp5[1];

    let temp6 = simd_ibutterfly(val4, val6, twiddles_dbl1[1]);
    val4 = temp6[0];
    val6 = temp6[1];

    let temp7 = simd_ibutterfly(val5, val7, twiddles_dbl1[1]);
    val5 = temp7[0];
    val7 = temp7[1];

    // Third layer of ibutterflies
    let temp8 = simd_ibutterfly(val0, val4, twiddles_dbl2[0]);
    val0 = temp8[0];
    val4 = temp8[1];

    let temp9 = simd_ibutterfly(val1, val5, twiddles_dbl2[0]);
    val1 = temp9[0];
    val5 = temp9[1];

    let temp10 = simd_ibutterfly(val2, val6, twiddles_dbl2[0]);
    val2 = temp10[0];
    val6 = temp10[1];

    let temp11 = simd_ibutterfly(val3, val7, twiddles_dbl2[0]);
    val3 = temp11[0];
    val7 = temp11[1];

    // Store results back
    values[offset + (0u << log_step)] = val0.x;
    values[offset + (0u << log_step) + 1u] = val0.y;
    values[offset + (0u << log_step) + 2u] = val0.z;
    values[offset + (0u << log_step) + 3u] = val0.w;

    values[offset + (1u << log_step)] = val1.x;
    values[offset + (1u << log_step) + 1u] = val1.y;
    values[offset + (1u << log_step) + 2u] = val1.z;
    values[offset + (1u << log_step) + 3u] = val1.w;

    values[offset + (2u << log_step)] = val2.x;
    values[offset + (2u << log_step) + 1u] = val2.y;
    values[offset + (2u << log_step) + 2u] = val2.z;
    values[offset + (2u << log_step) + 3u] = val2.w;

    values[offset + (3u << log_step)] = val3.x;
    values[offset + (3u << log_step) + 1u] = val3.y;
    values[offset + (3u << log_step) + 2u] = val3.z;
    values[offset + (3u << log_step) + 3u] = val3.w;

    values[offset + (4u << log_step)] = val4.x;
    values[offset + (4u << log_step) + 1u] = val4.y;
    values[offset + (4u << log_step) + 2u] = val4.z;
    values[offset + (4u << log_step) + 3u] = val4.w;

    values[offset + (5u << log_step)] = val5.x;
    values[offset + (5u << log_step) + 1u] = val5.y;
    values[offset + (5u << log_step) + 2u] = val5.z;
    values[offset + (5u << log_step) + 3u] = val5.w;

    values[offset + (6u << log_step)] = val6.x;
    values[offset + (6u << log_step) + 1u] = val6.y;
    values[offset + (6u << log_step) + 2u] = val6.z;
    values[offset + (6u << log_step) + 3u] = val6.w;

    values[offset + (7u << log_step)] = val7.x;
    values[offset + (7u << log_step) + 1u] = val7.y;
    values[offset + (7u << log_step) + 2u] = val7.z;
    values[offset + (7u << log_step) + 3u] = val7.w;
}
