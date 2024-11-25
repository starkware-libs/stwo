struct LayerParams {
    layer: u32,
    height: u32,
    size: u32,
    is_inverse: u32,
};

@group(0) @binding(0) var<storage, read_write> values: array<u32>;
@group(0) @binding(1) var<storage, read> twiddles: array<u32>;
@group(0) @binding(2) var<uniform> params: LayerParams;

// M31 field operations
fn m31_add(a: u32, b: u32) -> u32 {
    let sum = a + b;
    return select(sum, sum - 0x7FFFFFFF, sum >= 0x7FFFFFFF);
}

fn m31_sub(a: u32, b: u32) -> u32 {
    return select(a + 0x7FFFFFFF - b, a - b, a < b);
}

fn m31_mul(a: u32, b: u32) -> u32 {
    let prod = a * b;
    let q = (prod * 0x7FB10D6F) >> 31;
    let r = prod - q * 0x7FFFFFFF;
    return select(r, r - 0x7FFFFFFF, r >= 0x7FFFFFFF);
}

@compute @workgroup_size(256)
fn fft_layer_compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let layer_size = 1u << params.layer;
    let pair_distance = 1u << (params.layer - 1u);
    let group = idx >> params.layer;
    let pair_idx = idx & (pair_distance - 1u);
    
    let i0 = group * layer_size + pair_idx;
    let i1 = i0 + pair_distance;
    
    if (i1 >= params.size) {
        return;
    }

    let twiddle_idx = (pair_idx << (params.height - params.layer)) & (params.size - 1u);
    let twiddle = twiddles[twiddle_idx];
    
    let v0 = values[i0];
    let v1 = values[i1];
    
    // Butterfly computation
    if (params.is_inverse == 0u) {
        let temp = m31_mul(v1, twiddle);
        values[i0] = m31_add(v0, temp);
        values[i1] = m31_sub(v0, temp);
    } else {
        let temp = m31_mul(v1, twiddle);
        values[i0] = m31_add(v0, temp);
        values[i1] = m31_sub(v0, temp);
        
        // Apply normalization if this is the last layer
        if (params.layer == params.height - 1u) {
            let n_inv = 0x7FB10D6F;  // Multiplicative inverse of size mod 2^31-1
            values[i0] = m31_mul(values[i0], n_inv);
            values[i1] = m31_mul(values[i1], n_inv);
        }
    }
}