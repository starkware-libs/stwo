const MODULUS_BITS: u32 = 31u;
const P: u32 = 2147483647u;

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

struct InterpolateData {
    values: array<u32, 4>,
    initial_x: u32,
    initial_y: u32,
    log_size: u32,
}

struct Results {
    values: array<u32, 4>,
}

@group(0) @binding(0) var<storage, read> input: InterpolateData;
@group(0) @binding(1) var<storage, read_write> output: Results;

@compute @workgroup_size(1)
fn interpolate_compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x != 0u) {
        return;
    }

    if (input.log_size == 1u) {
        // Original log_size == 1 implementation
        var v0 = input.values[0];
        var v1 = input.values[1];
        let y = input.initial_y;
        
        let n = 2u;
        let yn = full_reduce(u64(y) * u64(n));
        
        // Calculate yn_inv using Fermat's little theorem
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
        
        let y_inv = full_reduce(u64(yn_inv) * u64(n));
        let n_inv = full_reduce(u64(yn_inv) * u64(y));
        
        ibutterfly(&v0, &v1, y_inv);
        
        output.values[0] = full_reduce(u64(v0) * u64(n_inv));
        output.values[1] = full_reduce(u64(v1) * u64(n_inv));
        output.values[2] = 0u;
        output.values[3] = 0u;
    } else if (input.log_size == 2u) {
        var v0 = input.values[0];
        var v1 = input.values[1];
        var v2 = input.values[2];
        var v3 = input.values[3];
        
        let x = input.initial_x;
        let y = input.initial_y;
        let n = 4u;

        // Calculate xyn_inv using Fermat's little theorem
        let xy_mult = full_reduce(u64(x) * u64(y));
        let xyn = full_reduce(u64(xy_mult) * u64(n));
        var xyn_inv = xyn;
        var power = P - 2u;
        var result = 1u;
        
        while (power > 0u) {
            if ((power & 1u) == 1u) {
                result = full_reduce(u64(result) * u64(xyn_inv));
            }
            xyn_inv = full_reduce(u64(xyn_inv) * u64(xyn_inv));
            power = power >> 1u;
        }
        xyn_inv = result;
        
        // Calculate inverse values
        let yn = full_reduce(u64(y) * u64(n));
        let xn = full_reduce(u64(x) * u64(n));
        let x_inv = full_reduce(u64(xyn_inv) * u64(yn));
        let y_inv = full_reduce(u64(xyn_inv) * u64(xn));
        let n_inv = full_reduce(u64(xyn_inv) * u64(x) * u64(y));
        
        // Calculate -y_inv
        let neg_y_inv = P - y_inv;
        
        // Perform ibutterfly operations in correct order
        ibutterfly(&v0, &v1, y_inv);
        ibutterfly(&v2, &v3, neg_y_inv);
        ibutterfly(&v0, &v2, x_inv);
        ibutterfly(&v1, &v3, x_inv);
        
        // Multiply all values by n_inv
        output.values[0] = full_reduce(u64(v0) * u64(n_inv));
        output.values[1] = full_reduce(u64(v1) * u64(n_inv));
        output.values[2] = full_reduce(u64(v2) * u64(n_inv));
        output.values[3] = full_reduce(u64(v3) * u64(n_inv));
    }
}
