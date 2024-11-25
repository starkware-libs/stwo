struct Uniforms {
    log_step: u32,
    offset: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> values: array<u32>;
@group(0) @binding(2) var<storage, read> twiddles_dbl0: array<u32, 1>;

// fn complex_multiply(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
//     return vec2<u32>(
//         a.x * b.x - a.y * b.y,
//         a.x * b.y + a.y * b.x
//     );
// }

// fn simd_ibutterfly(a: vec4<u32>, b: vec4<u32>, twiddle: u32) -> array<vec4<u32>, 2> {
//     let twiddle_vec = vec2<u32>(
//         twiddle & 0xFFFFu,
//         (twiddle >> 16u) & 0xFFFFu
//     );
    
//     // Process two complex numbers at once
//     let b_twisted_1 = complex_multiply(
//         vec2<u32>(b.x, b.y),
//         twiddle_vec
//     );
//     let b_twisted_2 = complex_multiply(
//         vec2<u32>(b.z, b.w),
//         twiddle_vec
//     );
    
//     let b_twisted = vec4<u32>(
//         b_twisted_1.x, b_twisted_1.y,
//         b_twisted_2.x, b_twisted_2.y
//     );
    
//     return array<vec4<u32>, 2>(
//         a + b_twisted,
//         a - b_twisted
//     );
// }

fn ibutterfly(a: u32, b: u32, twiddle: u32) -> vec2<u32> {
    let tmp = a;           // Store original value of a
    let sum = tmp + b;     // First output
    let diff = tmp - b;    // Calculate difference
    let twisted = diff * twiddle;  // Second output
    
    return vec2<u32>(
        sum,       // v0 = v0 + v1
        twisted    // v1 = (v0_original - v1) * twiddle
    );
}

fn simd_ibutterfly(a: vec4<u32>, b: vec4<u32>, twiddle: u32) -> array<vec4<u32>, 2> {
    // Process each element individually
    let result0 = ibutterfly(a.x, b.x, twiddle);
    let result1 = ibutterfly(a.y, b.y, twiddle);
    let result2 = ibutterfly(a.z, b.z, twiddle);
    let result3 = ibutterfly(a.w, b.w, twiddle);
    
    return array<vec4<u32>, 2>(
        vec4<u32>(result0.x, result1.x, result2.x, result3.x),  // v0 results (sums)
        vec4<u32>(result0.y, result1.y, result2.y, result3.y)   // v1 results (twisted differences)
    );
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let offset = uniforms.offset;
    let log_step = uniforms.log_step;
    
    // Load values
    let val0_offset = offset + (0u << log_step);
    let val1_offset = offset + (1u << log_step);
    
    var val0 = vec4<u32>(
        values[val0_offset],
        values[val0_offset + 1u],
        values[val0_offset + 2u],
        values[val0_offset + 3u]
    );
    
    var val1 = vec4<u32>(
        values[val1_offset],
        values[val1_offset + 1u],
        values[val1_offset + 2u],
        values[val1_offset + 3u]
    );
    
    // // Apply butterfly operation
    let result = simd_ibutterfly(val0, val1, twiddles_dbl0[0]);
    val0 = result[0];
    val1 = result[1];
    // do nothing
    
    // Store results back
    values[val0_offset] = val0.x;
    values[val0_offset + 1u] = val0.y;
    values[val0_offset + 2u] = val0.z;
    values[val0_offset + 3u] = val0.w;
    
    values[val1_offset] = val1.x;
    values[val1_offset + 1u] = val1.y;
    values[val1_offset + 2u] = val1.z;
    values[val1_offset + 3u] = val1.w;
}
