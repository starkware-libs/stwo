const MODULUS_BITS: u32 = 31u;
const P: u32 = 2147483647u;

// Partial reduce for values in [0, 2P)
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

struct InterpolateData {
    values: array<u32, 2>,
    initial_y: u32,
}

struct Results {
    values: array<u32, 2>,
}

@group(0) @binding(0) var<storage, read> input: InterpolateData;
@group(0) @binding(1) var<storage, read_write> output: Results;

fn ibutterfly(v0: ptr<function, u32>, v1: ptr<function, u32>, itwid: u32) {
    let tmp = *v0;
    *v0 = partial_reduce(tmp + *v1);
    *v1 = full_reduce(u64(partial_reduce(tmp + P - *v1)) * u64(itwid));
}

@compute @workgroup_size(1)
fn interpolate_compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x != 0u) {
        return;
    }

    // Get input values
    var v0 = input.values[0];
    var v1 = input.values[1];
    let y = input.initial_y;
    
    // Calculate required constants
    let n = 2u;  // BaseField::from(2)
    let yn = full_reduce(u64(y) * u64(n));
    
    // Calculate yn_inv using Fermat's little theorem: a^(p-1) ≡ 1 (mod p)
    // So, a^(p-2) gives multiplicative inverse
    var yn_inv = yn;
    var power = P - 2u;
    var result = 1u;
    
    while (power > 0u) {
        if ((power & 1u) == 1u) {
            result = full_reduce(u64(result) * u64(yn_inv));
        }
        yn_inv = full_reduce(u64(yn_inv) * u64(yn_inv));
        power = power >> 1u;
    }
    yn_inv = result;
    
    // Calculate y_inv and n_inv
    let y_inv = full_reduce(u64(yn_inv) * u64(n));
    let n_inv = full_reduce(u64(yn_inv) * u64(y));
    
    // Perform ibutterfly operation
    ibutterfly(&v0, &v1, y_inv);
    
    // Multiply results by n_inv
    output.values[0] = full_reduce(u64(v0) * u64(n_inv));
    output.values[1] = full_reduce(u64(v1) * u64(n_inv));
}
