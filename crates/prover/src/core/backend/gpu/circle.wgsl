const MODULUS_BITS: u32 = 31u;
const HALF_BITS: u32 = 16u;
// Mersenne prime P = 2^31 - 1
const P: u32 = 2147483647u;
const MAX_ARRAY_LOG_SIZE: u32 = 22;
const MAX_ARRAY_SIZE: u32 = 1u << MAX_ARRAY_LOG_SIZE;
const MAX_DEBUG_SIZE: u32 = 16;

fn partial_reduce(val: u32) -> u32 {
    let reduced = val - P;
    return select(val, reduced, reduced < val);
}

fn mod_mul(a: u32, b: u32) -> u32 {
    // Reduce inputs first
    let a_reduced = partial_reduce(a);
    let b_reduced = partial_reduce(b);
    
    // Split into 16-bit parts
    let a1 = a_reduced >> HALF_BITS;
    let a0 = a_reduced & 0xFFFFu;
    let b1 = b_reduced >> HALF_BITS;
    let b0 = b_reduced & 0xFFFFu;
    
    // Compute partial products
    let m0 = partial_reduce(a0 * b0);
    let m1 = partial_reduce(a0 * b1);
    let m2 = partial_reduce(a1 * b0);
    let m3 = partial_reduce(a1 * b1);
    
    // Combine middle terms with reduction
    let mid = partial_reduce(m1 + m2);
    
    // Combine parts with partial reduction
    let shifted_mid = partial_reduce(mid << HALF_BITS);
    let low = partial_reduce(m0 + shifted_mid);
    
    let high_part = partial_reduce(m3 + (mid >> HALF_BITS));
    
    // Final combination using Mersenne prime property
    let result = partial_reduce(
        partial_reduce((high_part << 1u)) + 
        partial_reduce((low >> MODULUS_BITS)) + 
        partial_reduce(low & P)
    );
    
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

@compute @workgroup_size(256)
fn interpolate_compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_size = 256u;

    let size = 1u << input.log_size;
    let thread_id = global_id.x;
    store_debug_value(thread_id, thread_id);
    for (var i = thread_id; i < size; i = i + thread_size) {
        output.values[i] = input.values[i];
       // store_debug_value(i, output.values[i]);
    }

    storageBarrier();

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

    storageBarrier();

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
            
            storageBarrier();
        }

        layer = layer + 1u;
        if (layer >= input.line_twiddles_layer_count) { break; }
    }

    storageBarrier();
    // divide all values by 2^log_size
    let inv = calculate_modular_inverse(1u << input.log_size);
    for (var i = thread_id; i < size; i = i + thread_size) {
        output.values[i] = mod_mul(output.values[i], inv);
    }
}
