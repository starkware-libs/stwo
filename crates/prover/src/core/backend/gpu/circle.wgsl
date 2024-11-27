const MODULUS_BITS: u32 = 31u;
const P: u32 = 2147483647u;
const MAX_ARRAY_LOG_SIZE: u32 = 22;
const MAX_ARRAY_SIZE: u32 = 1u << MAX_ARRAY_LOG_SIZE;

fn partial_reduce(val: u32) -> u32 {
    let reduced = val - P;
    return select(val, reduced, reduced < val);
}

fn full_reduce(val: u64) -> u32 {
    let first_shift = val >> MODULUS_BITS;
    let first_sum = first_shift + val + 1;
    let second_shift = first_sum >> MODULUS_BITS;
    let final_sum = second_shift + val;
    return u32(final_sum & u64(P));
}

fn ibutterfly(v0: ptr<function, u32>, v1: ptr<function, u32>, itwid: u32) {
    let tmp = *v0;
    *v0 = partial_reduce(tmp + *v1);
    *v1 = full_reduce(u64(partial_reduce(tmp + P - *v1)) * u64(itwid));
}

fn fft_layer_loop(i: u32, h: u32, t: u32) {
    let step = 1u << i;
    
    var l = 0u;
    loop {
        if (l >= step) { break; }
        let idx0 = (h << (i + 1u)) + l;
        let idx1 = idx0 + step;
        
        var val0 = output.values[idx0];
        var val1 = output.values[idx1];
        
        ibutterfly(&val0, &val1, t);
        
        output.values[idx0] = val0;
        output.values[idx1] = val1;
        
        l = l + 1u;
    }
}

fn calculate_modular_inverse(val: u32) -> u32 {
    var xyn_inv = val;
    var power = P - 2u;
    var result = 1u;

    while (power > 0u) {
        if ((power & 1u) == 1u) {
            result = full_reduce(u64(result) * u64(xyn_inv));
        }
        xyn_inv = full_reduce(u64(xyn_inv) * u64(xyn_inv));
        power = power >> 1u;
    }
    return result;
}

struct InterpolateData {
    values: array<u32, MAX_ARRAY_SIZE>,
    initial_x: u32,
    initial_y: u32,
    log_size: u32,
    circle_twiddles: array<u32, MAX_ARRAY_SIZE>,
    circle_twiddles_size: u32,
    line_twiddles_flat: array<u32, MAX_ARRAY_SIZE>,
    line_twiddles_layer_count: u32,
    line_twiddles_sizes: array<u32, MAX_ARRAY_SIZE>,
    line_twiddles_offsets: array<u32, MAX_ARRAY_SIZE>,
}

struct Results {
    values: array<u32, MAX_ARRAY_SIZE>,
}

struct DebugData {
    index: array<u32, 16>,
    values: array<u32, 16>,
    counter: atomic<u32>,
}

@group(0) @binding(0) var<storage, read> input: InterpolateData;
@group(0) @binding(1) var<storage, read_write> output: Results;
@group(0) @binding(2) var<storage, read_write> debug_buffer: DebugData;

fn store_debug_value(index: u32, value: u32) {
    let debug_idx = atomicAdd(&debug_buffer.counter, 1u);
    debug_buffer.index[debug_idx] = index;
    debug_buffer.values[debug_idx] = value;
}

@compute @workgroup_size(256)
fn interpolate_compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x != 0u) {
        return;
    }

    let size = 1u << input.log_size;

    if (input.log_size <= 2u) {
        var small_values: array<u32, 4>;
        for (var i = 0u; i < 4u; i = i + 1u) {
            small_values[i] = input.values[i];
        }
        
        let result = interpolate_small(input.log_size, small_values, input.initial_x, input.initial_y);
        
        for (var i = 0u; i < 4u; i = i + 1u) {
            output.values[i] = result[i];
        }
    } else {
        interpolate_large(size);
    }
}

fn interpolate_large(size: u32) {
    for (var i = 0u; i < size; i = i + 1u) {
        output.values[i] = input.values[i];
    }

    // Process circle_twiddles
    for (var h = 0u; h < input.circle_twiddles_size; h = h + 1u) {
        let t = input.circle_twiddles[h];
        fft_layer_loop(0u, h, t);
    }

    // Process line_twiddles
    var layer = 0u;
    loop {
        let layer_size = input.line_twiddles_sizes[layer];
        let layer_offset = input.line_twiddles_offsets[layer];
        
        for (var h = 0u; h < layer_size; h = h + 1u) {
            let t = input.line_twiddles_flat[layer_offset + h];
            store_debug_value(layer + 1u, t);
            fft_layer_loop(layer + 1u, h, t);
        }

        layer = layer + 1u;
        if (layer >= input.line_twiddles_layer_count) { break; }
    }

    // divide all values by 2^log_size
    let inv = calculate_modular_inverse(1u << input.log_size);
    for (var i = 0u; i < size; i = i + 1u) {
        output.values[i] = full_reduce(u64(output.values[i]) * u64(inv));
    }

}

fn interpolate_small(log_size: u32, values: array<u32, 4>, x: u32, y: u32) -> array<u32, 4> {
    var result: array<u32, 4>;
    
    if (log_size == 1u) {
        var v0 = values[0];
        var v1 = values[1];
        
        let n = 2u;
        let yn = full_reduce(u64(y) * u64(n));
        let yn_inv = calculate_modular_inverse(yn);
        
        let y_inv = full_reduce(u64(yn_inv) * u64(n));
        let n_inv = full_reduce(u64(yn_inv) * u64(y));
        
        ibutterfly(&v0, &v1, y_inv);
        
        result[0] = full_reduce(u64(v0) * u64(n_inv));
        result[1] = full_reduce(u64(v1) * u64(n_inv));
        result[2] = 0u;
        result[3] = 0u;
    } else if (log_size == 2u) {
        var v0 = values[0];
        var v1 = values[1];
        var v2 = values[2];
        var v3 = values[3];
        
        let n = 4u;
        let xy_mult = full_reduce(u64(x) * u64(y));
        let xyn = full_reduce(u64(xy_mult) * u64(n));
        let xyn_inv = calculate_modular_inverse(xyn);
        
        let yn = full_reduce(u64(y) * u64(n));
        let xn = full_reduce(u64(x) * u64(n));
        let x_inv = full_reduce(u64(xyn_inv) * u64(yn));
        let y_inv = full_reduce(u64(xyn_inv) * u64(xn));
        let n_inv = full_reduce(u64(xyn_inv) * u64(x) * u64(y));
        
        let neg_y_inv = P - y_inv;
        
        ibutterfly(&v0, &v1, y_inv);
        ibutterfly(&v2, &v3, neg_y_inv);
        ibutterfly(&v0, &v2, x_inv);
        ibutterfly(&v1, &v3, x_inv);
        
        result[0] = full_reduce(u64(v0) * u64(n_inv));
        result[1] = full_reduce(u64(v1) * u64(n_inv));
        result[2] = full_reduce(u64(v2) * u64(n_inv));
        result[3] = full_reduce(u64(v3) * u64(n_inv));
    }
    
    return result;
}