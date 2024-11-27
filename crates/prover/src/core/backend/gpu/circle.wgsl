const MODULUS_BITS: u32 = 31u;
const P: u32 = 2147483647u;
const MAX_ARRAY_LOG_SIZE: u32 = 12;
const MAX_ARRAY_SIZE: u32 = 1u << MAX_ARRAY_LOG_SIZE;
const MAX_DEBUG_SIZE: u32 = 16;

fn partial_reduce(val: u32) -> u32 {
    let reduced = val - P;
    return select(val, reduced, reduced < val);
}

fn ff_multiply(a: u32, b: u32) -> u32 {
    store_debug_value(996u, a);
    store_debug_value(995u, b);
    let ab = a * b;
    store_debug_value(997u, ab);
    return partial_reduce(ab);
}

fn mod_mul(a: u32, b: u32) -> u32 {
    var result: u32 = 0u;
    var current_a: u32 = a % P;
    var current_b: u32 = b;
    
    while (current_b > 0u) {
        if ((current_b & 1u) == 1u) {
            // Add current_a to result, but handle potential overflow
            let temp = result + current_a;
            result = select(temp, temp - P, temp >= P);
        }
        
        // Double current_a for next bit, but handle potential overflow
        current_a = select(current_a * 2u, (current_a * 2u) - P, current_a >= P/2u);
        current_b = current_b >> 1u;
    }
    
    return result;
}

// fn full_reduce(val: u64) -> u32 {
//     let first_shift = val >> MODULUS_BITS;
//     let first_sum = first_shift + val + 1;
//     let second_shift = first_sum >> MODULUS_BITS;
//     let final_sum = second_shift + val;
//     return u32(final_sum & u64(P));
// }

fn ibutterfly(v0: ptr<function, u32>, v1: ptr<function, u32>, itwid: u32) {
    let tmp = *v0;
    *v0 = partial_reduce(tmp + *v1);
    *v1 = mod_mul(partial_reduce(tmp + P - *v1), itwid);
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
            result = mod_mul(result, xyn_inv);
        }
        xyn_inv = mod_mul(xyn_inv, xyn_inv);
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
    index: array<u32, MAX_DEBUG_SIZE>,
    values: array<u32, MAX_DEBUG_SIZE>,
    counter: atomic<u32>,
}

@group(0) @binding(0) var<storage, read> input: InterpolateData;
@group(0) @binding(1) var<storage, read_write> output: Results;
@group(0) @binding(2) var<storage, read_write> debug_buffer: DebugData;

var<workgroup> shared_values: array<u32, MAX_DEBUG_SIZE>;

fn store_debug_value(index: u32, value: u32) {
    let debug_idx = atomicAdd(&debug_buffer.counter, 1u);
    debug_buffer.index[debug_idx] = index;
    debug_buffer.values[debug_idx] = value;
}

@compute @workgroup_size(1)
fn interpolate_compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_size = 1u;

    store_debug_value(777u, thread_size);
    //store_debug_value(888u, ff_multiply(10u, 10u));
    if (global_id.x >= thread_size) {
        return;
    }

    let size = 1u << input.log_size;
    let thread_id = global_id.x;
    for (var i = thread_id; i < size; i = i + thread_size) {
        output.values[i] = input.values[i];
       // store_debug_value(i, output.values[i]);
    }

    //storageBarrier();

    // Process circle_twiddles
    for (var h = thread_id; h < input.circle_twiddles_size; h = h + thread_size) {
        let t = input.circle_twiddles[h];
        let step = 1u << 0;
        var l = 0u;
        loop {
            if (l >= step) { break; }
            let idx0 = (h << 1u) + l;
            let idx1 = idx0 + step;
            
            var val0 = output.values[idx0];
            var val1 = output.values[idx1];
            
            ibutterfly(&val0, &val1, t);
            
            output.values[idx0] = val0;
            output.values[idx1] = val1;
            
            l = l + 1u;
        }
    }

    //storageBarrier();

    // Process line_twiddles
    var layer = 0u;
    loop {
        let layer_size = input.line_twiddles_sizes[layer];
        let layer_offset = input.line_twiddles_offsets[layer];
        
        for (var h = 0u; h < layer_size; h = h + 1u) {
            let t = input.line_twiddles_flat[layer_offset + h];
            let step = 1u << (layer + 1u);
            
            for (var l = thread_id; l < step; l = l + thread_size) {
                let idx0 = (h << (layer + 2u)) + l;
                let idx1 = idx0 + step;
                
                var val0 = output.values[idx0];
                var val1 = output.values[idx1];
                
                ibutterfly(&val0, &val1, t);
                
                output.values[idx0] = val0;
                output.values[idx1] = val1;
            }
            
            //storageBarrier();
        }

        layer = layer + 1u;
        if (layer >= input.line_twiddles_layer_count) { break; }
    }

    //storageBarrier();
    // divide all values by 2^log_size
    let inv = calculate_modular_inverse(1u << input.log_size);
    //store_debug_value(997u, output.values[3]);
    //store_debug_value(998u, inv);
    //store_debug_value(999u, full_reduce(u64(output.values[3]) * u64(inv)));
    //store_debug_value(1000u, mod_mul(output.values[3], inv));
    for (var i = thread_id; i < size; i = i + thread_size) {
        //output.values[i] = full_reduce(u64(output.values[i]) * u64(inv));
        //output.values[i] = full_reduce32(multiply32(output.values[i], inv));
        output.values[i] = mod_mul(output.values[i], inv);
    }
}
