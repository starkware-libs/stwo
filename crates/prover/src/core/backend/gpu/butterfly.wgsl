const MODULUS_BITS: u32 = 31u;
const P: u32 = 2147483647u;

// Partial reduce for values in [0, 2P)
fn partial_reduce(val: u32) -> u32 {
    let reduced = val - P;
    return select(val, reduced, reduced < val);
}

fn full_reduce(val: u64) -> u32 {
    // Step 1 & 2: (val >> MODULUS_BITS) + val + 1
    let first_shift = val >> MODULUS_BITS;
    let first_sum = first_shift + val + 1;
    
    // Step 3 & 4: (first_sum >> MODULUS_BITS) + val
    let second_shift = first_sum >> MODULUS_BITS;
    let final_sum = second_shift + val;
    
    // Step 5 & 6: & P and cast to u32
    return u32(final_sum & u64(P));
}

struct Operands {
    v0: u32,
    v1: u32,
    twiddle: u32,
}

struct Results {
    v0: u32,
    v1: u32,
}

@group(0) @binding(0) var<storage, read> operands: Operands;
@group(0) @binding(1) var<storage, read_write> results: Results;

// Basic butterfly operation
@compute @workgroup_size(1)
fn butterfly_compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x != 0u) {
        return;
    }

    var v0 = operands.v0;
    var v1 = operands.v1;
    let twid = operands.twiddle;

    // Butterfly computation
    let tmp = full_reduce(u64(v1) * u64(twid));
    v1 = partial_reduce(v0 + P - tmp);  // v0 - tmp
    v0 = partial_reduce(v0 + tmp);      // v0 + tmp

    results.v0 = v0;
    results.v1 = v1;
}

// Basic inverse butterfly operation
@compute @workgroup_size(1)
fn ibutterfly_compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x != 0u) {
        return;
    }

    var v0 = operands.v0;
    var v1 = operands.v1;
    let itwid = operands.twiddle;

    // Inverse butterfly computation
    let tmp = v0;
    v0 = partial_reduce(tmp + v1);
    v1 = full_reduce(u64(partial_reduce(tmp + P - v1)) * u64(itwid));

    results.v0 = v0;
    results.v1 = v1;
}
